package simt

import (
	"errors"
	"fmt"
	"io/fs"
	"math"

	"github.com/CWBudde/gocuda/cuda"
	"github.com/CWBudde/gocuda/internal/jit"
	"github.com/CWBudde/gocuda/internal/lower"
)

// Kernel is a Go kernel that has been transpiled, compiled and loaded.
type Kernel struct {
	Name   string // the Go function's name, also the C entry point
	Source string // the generated CUDA C
	PTX    []byte // what was loaded, whether NVRTC or the generator produced it
	Log    string // NVRTC's compiler log, usually empty

	// Prebuilt says the PTX came from a registration rather than from NVRTC,
	// so nothing was compiled here. It is what a test asserts on to prove the
	// ahead-of-time path is being taken, and the honest signal that Log is
	// empty because there was no compilation, not because it was clean.
	Prebuilt bool

	// Arch is the virtual architecture the PTX was built for. On the JIT path
	// that is the device's own; on the prebuilt path it may be older, since
	// PTX is forward compatible and the driver JITs it. Either way the field
	// answers the same question: what this PTX targets.
	Arch string

	// BoundsChecks says this kernel was built with WithBoundsChecks, and is
	// what turns a faulted launch into a TrapError rather than a bare driver
	// error. It is also the honest signal that this build went through NVRTC:
	// a bounds-checked kernel never matches a prebuilt.
	BoundsChecks bool

	// RequiredBlock, SharedBytes, DynSharedWidth and Params are what the
	// kernel's source demands of a launch; see Unit.
	RequiredBlock  int
	SharedBytes    int
	DynSharedWidth int
	Params         []Param

	// maxShared is what the device this kernel was built for offers per block,
	// or 0 when there was nobody to ask. It is kept because a dynamic tile is
	// sized at the launch rather than at Build, so that is the only place the
	// total can be checked against the limit at all.
	maxShared int

	fn *cuda.Function
}

// A BuildOption adjusts what Build does. The options are variadic so that the
// common call stays two arguments and a name.
type BuildOption func(*buildOptions)

type buildOptions struct {
	cacheDir     string
	noPrebuilt   bool
	noDiskCache  bool
	boundsChecks bool
}

// lowerOptions is what of a build's options the lowering is allowed to see.
// Only WithBoundsChecks changes the generated C; WithCacheDir, WithoutPrebuilt
// and WithoutDiskCache all concern the loading that comes after it and are
// simply not consulted here.
func (o buildOptions) lowerOptions() []lower.Option {
	if o.boundsChecks {
		return []lower.Option{lower.WithBoundsChecks()}
	}
	return nil
}

// WithCacheDir chooses where the generated .cu and the PTX that was loaded are
// written for inspection. Empty disables the dump. The default is
// ".gocuda-cache".
//
// This replaces a package-level setter, which wrote a package-level variable
// unsynchronised, for the whole process, and reached the tile track as well: a
// debugging preference expressed in one corner of a program silently changed
// where an unrelated library wrote files.
func WithCacheDir(dir string) BuildOption {
	return func(o *buildOptions) { o.cacheDir = dir }
}

// WithoutPrebuilt ignores any registered PTX and compiles with NVRTC.
//
// It is the negative control: a test that asserts the ahead-of-time path was
// taken proves nothing unless the same test can also demand the other one and
// see the difference. It is also the way to check that a kernel behaves
// identically whichever path produced it.
func WithoutPrebuilt() BuildOption {
	return func(o *buildOptions) { o.noPrebuilt = true }
}

// WithoutDiskCache compiles with NVRTC rather than reading the persistent PTX
// cache, and stores nothing in it.
//
// That cache is on by default, because an opt-in one does not do the thing it
// is for: taking NVRTC out of process start. This is the way to measure what
// it saves, and the way for a caller who would rather not have a library
// writing to their user cache directory to say so. GOCUDA_PTX_CACHE moves the
// directory; only this turns the cache off, because whether a compiler runs is
// not something one corner of a program should decide for the rest of it.
func WithoutDiskCache() BuildOption {
	return func(o *buildOptions) { o.noDiskCache = true }
}

// WithBoundsChecks builds the kernel with a range check at every slice, array
// and shared-tile subscript. An index outside its buffer calls __trap()
// instead of reading or writing memory that belongs to something else.
//
// This is the debug build, and it is a build option rather than something a
// kernel says about itself precisely so that the released path pays nothing:
// without it not one character of the generated C changes. It also means a
// debug build never loads a prebuilt artifact -- the checks and a marker line
// are both in the source the hash is taken over -- so it always goes through
// NVRTC and needs a toolkit.
//
// What the device can tell you afterwards is that it trapped, and not which
// index did it: a trap carries no payload and the context does not survive
// it. Running the same kernel under gpu.RunCPU is what names the index, the
// length and the thread, because there a kernel is ordinary Go.
func WithBoundsChecks() BuildOption {
	return func(o *buildOptions) { o.boundsChecks = true }
}

// Build transpiles the named Go kernel from fsys and loads it into dev: the
// whole SIMT pipeline, Go source to CUDA C to PTX.
//
// The kernel is transpiled even when a prebuilt image is available, and that
// is the point rather than an oversight. The generated CUDA C is what produces
// the hash the registry is keyed on, so there is no way to find a prebuilt
// without lowering the kernel first; and having it means Source, RequiredBlock
// and SharedBytes -- and therefore the shared-memory check below and the block
// check at launch -- are derived from the source in hand on both paths instead
// of being trusted from an artifact. Lowering one function is a fraction of a
// millisecond. What the ahead-of-time path buys out is NVRTC, which is the
// tens of milliseconds.
func Build(dev *cuda.Context, fsys fs.FS, name string, opts ...BuildOption) (*Kernel, error) {
	o := buildOptions{cacheDir: jit.DefaultCacheDir}
	for _, opt := range opts {
		opt(&o)
	}

	u, err := Transpile(fsys, name, opts...)
	if err != nil {
		return nil, err
	}
	// A tile larger than the device's limit is rejected by cuModuleLoadData
	// with nothing but a generic error code, so it is worth catching here,
	// where both the total and the kernel's name are still at hand.
	if limit := dev.MaxSharedMemPerBlock(); limit > 0 && u.SharedBytes > limit {
		return nil, &SharedMemoryError{Kernel: name, Bytes: u.SharedBytes, Limit: limit}
	}

	req := jit.Request{
		Src:         u.Source,
		Name:        name,
		CacheDir:    o.cacheDir,
		FastMath:    u.FastMath,
		NoDiskCache: o.noDiskCache,
	}
	var pre Prebuilt
	if !o.noPrebuilt {
		major, minor := dev.ComputeCapability()
		if p, ok := pickPrebuilt(u.SourceHash, major, minor); ok {
			pre = p
			req.PTX, req.PTXArch = p.PTX, p.Arch
		}
	}

	res, err := jit.Load(dev, req)
	if err != nil {
		return nil, err
	}
	k := &Kernel{
		Name:           name,
		Source:         u.Source,
		PTX:            res.PTX,
		Log:            res.Log,
		Prebuilt:       res.Prebuilt,
		Arch:           res.Arch,
		BoundsChecks:   u.BoundsChecks,
		RequiredBlock:  u.RequiredBlock,
		SharedBytes:    u.SharedBytes,
		DynSharedWidth: u.DynSharedWidth,
		Params:         u.Params,
		maxShared:      dev.MaxSharedMemPerBlock(),
		fn:             res.Func,
	}
	if res.Prebuilt {
		// Log is documented as NVRTC's compiler log, so on this path it holds
		// whatever NVRTC said when the artifact was generated, and nothing at
		// all when that was a clean compile. Writing a note about the load
		// into it instead would make every caller that prints a non-empty log
		// as a warning print one.
		k.Log = pre.Log
	}
	return k, nil
}

// SharedMemoryError is a kernel whose statically declared shared memory does
// not fit the device it was built for.
type SharedMemoryError struct {
	Kernel string
	Bytes  int // what the kernel declares
	Limit  int // what the device offers per block
}

func (e *SharedMemoryError) Error() string {
	return fmt.Sprintf("simt: %s declares %d bytes of shared memory, but this device offers %d per block",
		e.Kernel, e.Bytes, e.Limit)
}

// BlockSizeError is a launch at a block size the kernel cannot run at.
type BlockSizeError struct {
	Kernel string
	Want   int // the size the kernel declared with AssumeBlockDim
	Got    int // the size the launch asked for
}

func (e *BlockSizeError) Error() string {
	return fmt.Sprintf("simt: %s requires block == %d, got %d", e.Kernel, e.Want, e.Got)
}

// DynamicSharedError is a launch whose spelling does not match how the kernel
// declares its shared memory.
//
// The two are separate calls rather than one with an optional size because the
// mismatch is silent either way round: a kernel with a dynamic tile launched
// through Launch gets zero bytes of it and reads off the end of nothing, and a
// kernel without one launched through LaunchShared is handed a size for a tile
// it does not have, plus an argument its signature has no parameter for. Both
// are refused here, where the kernel's own answer is at hand.
type DynamicSharedError struct {
	Kernel  string
	Dynamic bool // whether the kernel declares a dynamic tile
}

func (e *DynamicSharedError) Error() string {
	if e.Dynamic {
		return fmt.Sprintf("simt: %s declares a dynamically sized shared tile; launch it with LaunchShared, which sizes it", e.Kernel)
	}
	return fmt.Sprintf("simt: %s declares no dynamically sized shared tile; launch it with Launch", e.Kernel)
}

// AliasError is a launch that bound two of a kernel's slice parameters to
// overlapping device memory while the kernel writes through one of them.
type AliasError struct {
	Kernel string
	Write  string // the parameter the kernel writes through
	Other  string // the parameter overlapping it
}

func (e *AliasError) Error() string {
	return fmt.Sprintf("simt: %s: parameters %s and %s were given overlapping device memory, "+
		"and the kernel writes through %s; the generated C declares every pointer __restrict__, "+
		"which promises that they do not overlap",
		e.Kernel, e.Write, e.Other, e.Write)
}

// ArgCountError is a launch that supplied a different number of arguments than
// the kernel has parameters.
//
// Nothing downstream catches this. cuda.BuildArgs marshals whatever it is
// given, and cuLaunchKernel reads as many parameters as the loaded function
// declares from the array it was handed -- so too few arguments has the kernel
// reading whatever follows the array, and too many is silently dropped. Both
// are undefined behaviour that surfaces as wrong numbers rather than as an
// error, which is why the count is checked against the signature that lowering
// recorded.
type ArgCountError struct {
	Kernel string
	Want   int // parameters the kernel declares, after the gpu.Ctx
	Got    int // arguments the launch supplied
}

func (e *ArgCountError) Error() string {
	return fmt.Sprintf("simt: %s takes %d launch argument(s), got %d", e.Kernel, e.Want, e.Got)
}

// checkArgs refuses a launch whose argument count does not match the kernel's.
//
// args is what reaches the driver, which for a kernel with a dynamically sized
// shared tile is one longer than the caller wrote: LaunchSharedDim appends the
// tile's length, and Params -- which describes the Go signature -- does not
// include it. The extra is subtracted again before the numbers are reported,
// so the error states the count the caller can actually see.
func (k *Kernel) checkArgs(args []any) error {
	extra := 0
	if k.DynSharedWidth != 0 {
		extra = 1
	}
	if got := len(args) - extra; got != len(k.Params) {
		return &ArgCountError{Kernel: k.Name, Want: len(k.Params), Got: got}
	}
	return nil
}

// checkAliasing refuses a launch that breaks the __restrict__ promise.
//
// The Go source cannot make that promise: VecAdd(ctx, c, a, b) is three
// parameters and may be handed one buffer three times. The qualifier is worth
// having anyway -- it is most of what const/__restrict__ buys on a device --
// so the promise is checked here, against the buffers the caller actually
// bound, rather than left as undefined behaviour that shows up as wrong
// numbers on one architecture.
//
// Two read-only parameters may share memory, and that is not an oversight:
// what __restrict__ forbids is an object being modified through one pointer
// and reached through another, so an overlap matters only when one side of it
// is written. An argument that cannot report its range -- a raw cuda.Arg
// holding a pointer -- is skipped, because there is nothing to compare.
//
// The CPU emulator makes no such check. RunCPU takes a closure, so the kernel's
// slices never pass through it and it has no way to see that two of them are
// the same Go slice; there the kernel is ordinary Go, where aliasing is
// defined, so the two backends agree on everything except the diagnosis.
func (k *Kernel) checkAliasing(args []any) error {
	if len(args) < len(k.Params) {
		// checkArgs has already refused this at every launch; pairing the
		// arguments off anyway would be a guess about which is which.
		return nil
	}
	type span struct {
		base  cuda.DevPtr
		bytes int
		param int
	}
	var spans []span
	// Over the parameters rather than over the arguments, because args may
	// carry one more than Params describes: the length LaunchSharedDim
	// appends for a dynamically sized tile. It is a scalar and always last,
	// so the positions the two share still line up.
	for i, p := range k.Params {
		if !p.Slice {
			continue
		}
		r, ok := args[i].(cuda.Ranger)
		if !ok {
			continue
		}
		base, bytes := r.DeviceRange()
		if bytes == 0 {
			continue
		}
		spans = append(spans, span{base: base, bytes: bytes, param: i})
	}
	for i, a := range spans {
		for _, b := range spans[i+1:] {
			if a.base >= b.base+cuda.DevPtr(b.bytes) || b.base >= a.base+cuda.DevPtr(a.bytes) {
				continue
			}
			write, other := a.param, b.param
			if k.Params[write].ReadOnly {
				write, other = other, write
			}
			if k.Params[write].ReadOnly {
				continue // both only read it, which restrict allows
			}
			return &AliasError{Kernel: k.Name, Write: k.Params[write].Name, Other: k.Params[other].Name}
		}
	}
	return nil
}

// Launch runs the kernel over grid blocks of block threads. Arguments are
// given as Go values -- device slices, float32, int32 -- in the same order as
// the Go kernel's parameters, minus the gpu.Ctx.
//
// A kernel that declared its block size with gpu.Ctx.AssumeBlockDim is refused
// at any other size: its shared tiles are sized for that one geometry, so a
// different block would stage the wrong number of samples and read past them.
// A launch that binds one device buffer to two parameters is refused too, when
// the kernel writes through either of them; see checkAliasing. So is one that
// supplies the wrong number of arguments, which the driver would otherwise
// read off the end of the parameter array; see checkArgs.
func (k *Kernel) Launch(grid, block int, args ...any) error {
	return k.LaunchDim(cuda.D1(grid), cuda.D1(block), args...)
}

// LaunchDim is Launch over a grid of any rank, for a kernel that reads more
// than one axis of its position.
//
// The block-size contract counts threads per block across all three axes:
// AssumeBlockDim says how many threads fill a shared tile, not how they are
// arranged, so a 16x16 block satisfies AssumeBlockDim(256).
func (k *Kernel) LaunchDim(grid, block cuda.Dim3, args ...any) error {
	if k.DynSharedWidth != 0 {
		return &DynamicSharedError{Kernel: k.Name, Dynamic: true}
	}
	return k.launch(nil, grid, block, 0, args)
}

// LaunchOn queues the kernel on a stream and returns without waiting for it.
//
// Every check Launch makes is made here too -- the block-size contract, the
// argument count, the aliasing rule -- because all of them are host-side and
// none of them need the kernel to have run. What changes is when a *device*
// fault is reported: not by this call, which returns before the kernel
// starts, but by the stream's Sync or Wait. See cuda.Stream.
//
// The buffers the arguments name must stay allocated until then. That is the
// part a synchronous launch did for free.
func (k *Kernel) LaunchOn(s *cuda.Stream, grid, block int, args ...any) error {
	return k.LaunchOnDim(s, cuda.D1(grid), cuda.D1(block), args...)
}

// LaunchOnDim is LaunchOn over a grid of any rank.
//
// There is deliberately no asynchronous twin of LaunchShared yet: a dynamic
// shared tile and a stream are two new things at once, and nothing has asked
// for both. A kernel that declares one is refused here rather than launched
// without its tile.
func (k *Kernel) LaunchOnDim(s *cuda.Stream, grid, block cuda.Dim3, args ...any) error {
	if s == nil {
		return fmt.Errorf("simt: %s: LaunchOn needs a stream; use Launch for the synchronous form", k.Name)
	}
	if k.DynSharedWidth != 0 {
		return &DynamicSharedError{Kernel: k.Name, Dynamic: true}
	}
	return k.launch(s, grid, block, 0, args)
}

// LaunchN runs the kernel over enough blocks to cover n threads. Kernels guard
// against the ragged tail themselves, exactly as in CUDA C.
func (k *Kernel) LaunchN(n, block int, args ...any) error {
	return k.Launch((n+block-1)/block, block, args...)
}

// LaunchShared runs a kernel that declares a dynamically sized shared tile,
// giving that tile n elements.
//
// n is an element count and not a byte count on purpose. How wide an element
// is was decided when the kernel was lowered -- it is the thing the emitter
// knows and the caller would have to look up -- so the multiplication happens
// here, against the width the Unit carries, rather than at every launch site.
// The same count reaches the kernel as the length its len() reads, which is
// why the two can never disagree.
func (k *Kernel) LaunchShared(grid, block, n int, args ...any) error {
	return k.LaunchSharedDim(cuda.D1(grid), cuda.D1(block), n, args...)
}

// LaunchSharedDim is LaunchShared over a grid of any rank.
func (k *Kernel) LaunchSharedDim(grid, block cuda.Dim3, n int, args ...any) error {
	if k.DynSharedWidth == 0 {
		return &DynamicSharedError{Kernel: k.Name}
	}
	if n < 0 {
		return fmt.Errorf("simt: %s: a shared tile of %d elements is not a size", k.Name, n)
	}
	// Both numbers derived from n are 32-bit on the device -- the byte count
	// the driver is handed, and the length the kernel reads -- so a count that
	// does not fit has to be refused here rather than wrapped. The limit check
	// below cannot stand in for this one: the product is computed in a host
	// int, so it is the truncation of n that comes first, and maxShared == 0,
	// which is what "there was no device to ask" looks like, disables the
	// limit check entirely.
	if n > math.MaxInt32/k.DynSharedWidth {
		return fmt.Errorf("simt: %s: a shared tile of %d elements of %d bytes does not fit a 32-bit size",
			k.Name, n, k.DynSharedWidth)
	}
	dynBytes := n * k.DynSharedWidth
	// The same check Build makes, and it has to be made again: this is the
	// first moment the dynamic half of the total exists. cuLaunchKernel would
	// refuse it too, with a generic error code and neither number in it.
	if total := k.SharedBytes + dynBytes; k.maxShared > 0 && total > k.maxShared {
		return &SharedMemoryError{Kernel: k.Name, Bytes: total, Limit: k.maxShared}
	}
	// The length is the last parameter of the generated signature, which is
	// where the emitter appends it, so it goes last here too. Into a slice of
	// its own, because appending to args would write into the caller's backing
	// array whenever they passed one with room to spare.
	full := make([]any, 0, len(args)+1)
	full = append(append(full, args...), int32(n))
	return k.launch(nil, grid, block, dynBytes, full)
}

// launch is the half every spelling shares: the block-size contract, the
// argument marshalling, and the launch itself.
//
// A nil stream means the synchronous launch, which is what every form but
// LaunchOn asks for.
func (k *Kernel) launch(s *cuda.Stream, grid, block cuda.Dim3, dynBytes int, args []any) error {
	threads := int(block.X) * int(block.Y) * int(block.Z)
	if k.RequiredBlock != 0 && threads != k.RequiredBlock {
		return &BlockSizeError{Kernel: k.Name, Want: k.RequiredBlock, Got: threads}
	}
	if err := k.checkArgs(args); err != nil {
		return err
	}
	if err := k.checkAliasing(args); err != nil {
		return err
	}
	flat, err := cuda.BuildArgs(args...)
	if err != nil {
		return err
	}
	if s == nil {
		return k.diagnose(k.fn.LaunchSync(grid, block, dynBytes, flat...))
	}
	// diagnose gets much less to work with here, and that is not a gap that
	// can be closed at this level: an asynchronous launch returns before the
	// kernel runs, so a device fault it causes is reported by whatever
	// synchronises next -- the stream's Sync or Wait, or any later call on
	// the context once the fault has poisoned it. What this still catches is
	// the launch refusing outright, which is a host-side failure and happens
	// before the return.
	return k.diagnose(k.fn.Launch(s, grid, block, dynBytes, flat...))
}

// diagnose turns a failed launch into a TrapError, but only for the failures
// a __trap() can actually have produced.
//
// Wrapping every error was wrong in two directions, and both of them point
// the reader at the wrong thing. A launch that never started -- the wrong
// argument count, a block the device cannot fit, too few registers -- is not
// a fault at all, so "its launch faulted, which an index outside a slice
// would do" sends somebody hunting an index that was never read. And once a
// context is poisoned, every later launch fails in Context.bind before the
// kernel is dispatched: naming *that* kernel accuses one that did not run,
// and buries the ContextPoisonedError that says which one did.
//
// So: the poisoned case passes through untouched, because it already carries
// a better sentence than this one could write, and everything else has to
// name a result code a device-side fault produces.
func (k *Kernel) diagnose(err error) error {
	if err == nil || !k.BoundsChecks {
		return err
	}
	var poisoned *cuda.ContextPoisonedError
	if errors.As(err, &poisoned) {
		return err
	}
	var e *cuda.Error
	if !errors.As(err, &e) || !faulted(e.Code) {
		return err
	}
	return &TrapError{Kernel: k.Name, Err: err}
}

// faulted reports whether a result code means the kernel ran and the device
// faulted, as opposed to the launch being refused before it started.
//
// 719 is what a __trap() was measured to surface as here
// (docs/toolchain.md#what-the-host-sees-after-a-trap). 700 is on the list
// because it is what an out-of-range access that the checks did *not* cover
// lands on -- a raw overrun in a helper the emitter had no length for -- and
// under a bounds-checked build that is the same mistake wearing a different
// code, which is exactly what TrapError says out loud. Every other code,
// ErrLaunchOutOfResources and the argument errors among them, is left alone.
func faulted(r cuda.Result) bool {
	switch r {
	case cuda.ErrLaunchFailed, cuda.ErrIllegalAddress:
		return true
	}
	return false
}

// TrapError is a bounds-checked launch that faulted.
//
// It exists because the device cannot say anything useful about the fault
// itself. __trap() carries no payload, so the driver reports an unspecified
// launch failure and the context does not survive it -- which is
// indistinguishable, from the error alone, from a genuine illegal address in
// code the checks never covered. What the caller does know, and what this
// says, is that this build asked for the checks, so an index outside its
// buffer is by far the most likely cause and there is a way to find out which
// one.
//
// It is raised only for those two codes; see diagnose. A launch that was
// refused before it started, or one failing because an earlier trap poisoned
// the context, is not evidence of anything a bounds check saw.
type TrapError struct {
	Kernel string
	Err    error
}

func (e *TrapError) Error() string {
	return fmt.Sprintf("simt: %s was built with bounds checks and its launch faulted, "+
		"which an index outside a slice or a shared tile would do: %v. "+
		"Run the same kernel under gpu.RunCPU to find out which index -- there a "+
		"kernel is ordinary Go, and the panic names the index, the length and the thread",
		e.Kernel, e.Err)
}

func (e *TrapError) Unwrap() error { return e.Err }
