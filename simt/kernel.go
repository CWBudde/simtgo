package simt

import (
	"fmt"
	"io/fs"

	"github.com/CWBudde/gocuda/cuda"
	"github.com/CWBudde/gocuda/internal/jit"
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

	// RequiredBlock and SharedBytes are what the kernel's source demands of a
	// launch; see Unit.
	RequiredBlock int
	SharedBytes   int

	fn *cuda.Function
}

// A BuildOption adjusts what Build does. The options are variadic so that the
// common call stays two arguments and a name.
type BuildOption func(*buildOptions)

type buildOptions struct {
	cacheDir   string
	noPrebuilt bool
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

	u, err := Transpile(fsys, name)
	if err != nil {
		return nil, err
	}
	// A tile larger than the device's limit is rejected by cuModuleLoadData
	// with nothing but a generic error code, so it is worth catching here,
	// where both the total and the kernel's name are still at hand.
	if limit := dev.MaxSharedMemPerBlock(); limit > 0 && u.SharedBytes > limit {
		return nil, &SharedMemoryError{Kernel: name, Bytes: u.SharedBytes, Limit: limit}
	}

	req := jit.Request{Src: u.Source, Name: name, CacheDir: o.cacheDir}
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
		Name:          name,
		Source:        u.Source,
		PTX:           res.PTX,
		Log:           res.Log,
		Prebuilt:      res.Prebuilt,
		Arch:          res.Arch,
		RequiredBlock: u.RequiredBlock,
		SharedBytes:   u.SharedBytes,
		fn:            res.Func,
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

// Launch runs the kernel over grid blocks of block threads. Arguments are
// given as Go values -- device slices, float32, int32 -- in the same order as
// the Go kernel's parameters, minus the gpu.Ctx.
//
// A kernel that declared its block size with gpu.Ctx.AssumeBlockDim is refused
// at any other size: its shared tiles are sized for that one geometry, so a
// different block would stage the wrong number of samples and read past them.
func (k *Kernel) Launch(grid, block int, args ...any) error {
	if k.RequiredBlock != 0 && block != k.RequiredBlock {
		return &BlockSizeError{Kernel: k.Name, Want: k.RequiredBlock, Got: block}
	}
	flat, err := cuda.BuildArgs(args...)
	if err != nil {
		return err
	}
	return k.fn.LaunchSync(cuda.D1(grid), cuda.D1(block), 0, flat...)
}

// LaunchN runs the kernel over enough blocks to cover n threads. Kernels guard
// against the ragged tail themselves, exactly as in CUDA C.
func (k *Kernel) LaunchN(n, block int, args ...any) error {
	return k.Launch((n+block-1)/block, block, args...)
}
