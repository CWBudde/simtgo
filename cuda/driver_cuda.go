//go:build cuda

package cuda

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"
)

// This file is the CUDA driver API and NVRTC, without cgo. The entry points
// are resolved from the shared libraries at run time in loader_cuda.go, so the
// package compiles with CGO_ENABLED=0 and needs no CUDA toolkit to build --
// only a driver to run against.
//
// The constants below are the handful of enum values this package needs,
// copied from cuda.h rather than included from it, since there is no header to
// include any more. They are part of the driver ABI and do not change.
const (
	attrMaxSharedMemoryPerBlock = 8
	attrComputeCapabilityMajor  = 75
	attrComputeCapabilityMinor  = 76
)

var initOnce struct {
	sync.Once
	err error
}

// Init initialises the CUDA driver. It is safe to call repeatedly.
//
// It is also where a machine with no CUDA is discovered: opening libcuda is
// part of initialising, and fails with a *LibraryError naming what was tried.
func Init() error {
	initOnce.Do(func() {
		if err := loadDriver(); err != nil {
			initOnce.err = err
			return
		}
		initOnce.err = check(cuInit(0), "cuInit")
	})
	return initOnce.err
}

// Available reports whether a usable CUDA device is present.
func Available() bool {
	if Init() != nil {
		return false
	}
	var n int32
	if err := check(cuDeviceGetCount(&n), "cuDeviceGetCount"); err != nil {
		return false
	}
	return n > 0
}

// Context owns a device's primary context.
//
// # Goroutine safety
//
// A *Context is safe to share between goroutines. Every method that needs the
// context current goes through call, which makes it current and issues its
// driver call on one OS thread it holds for the duration, so a goroutine that
// migrates cannot leave a call behind on a thread the context was never made
// current on. See docs/decisions.md#a-driver-call-holds-its-os-thread.
//
// Name and Close are the two that do not, because they do not need to: both
// address the device by ordinal -- cuDeviceGetName and
// cuDevicePrimaryCtxRelease -- and neither reads the current context. (Close
// unloads the modules first, and each of those unloads does go through call.)
//
// What is *not* promised is ordering. The calls are synchronous, so each one
// has finished before it returns, but two goroutines allocating, copying and
// launching against one context interleave however the scheduler likes, and
// nothing here serialises them for you. A device buffer shared between
// goroutines needs the same care as any other shared memory.
//
// A context also owns every module loaded into it, including the compilation
// cache behind LoadPTXCached. A CUmodule is only meaningful inside the context
// it was loaded into, so that cache has to live here rather than in a
// package-level map: a process-wide cache outlives the contexts it caches for
// and would hand out handles belonging to a context that has already been
// released. Close unloads what is left, so no module can survive its context.
type Context struct {
	dev int32
	ctx uintptr

	// The compute capability is kept as the two numbers the driver reports
	// rather than as the "compute_75" string alone, because callers that gate
	// a feature on hardware ("needs 8.0 or better") have to compare numbers,
	// and re-parsing the string to get them back is a detour through a format
	// this package invented in the first place.
	ccMajor, ccMinor int

	maxShared int

	// mu guards the module bookkeeping below. It is also held across the
	// build callback of LoadPTXCached, so one source is compiled once
	// however many goroutines ask for it at the same moment.
	mu      sync.Mutex
	modules []*Module          // every module loaded here, in load order
	cache   map[string]*Module // the subset loaded through LoadPTXCached
	closed  bool

	// poisoned: see the comment below. It is shared with every other Context
	// wrapping the same device's primary context, so it is a pointer to the
	// cell rather than the cell itself, and NewContext fills it in.
	poisoned *atomic.Pointer[ContextPoisonedError]
}

// poisoned is the first sticky fault this device's primary context saw, set
// once and never cleared. Every later call reports it instead of the driver's
// own repetition of the same code with no explanation of where it came from
// -- the difference between "cuMemAlloc failed: unspecified launch failure"
// ten calls later and a sentence naming the kernel fault that caused it. See
// ContextPoisonedError.
//
// It is an atomic rather than a field under mu, and that is not a
// micro-optimisation. bind() reads it, and loadPTXLocked calls bind() with mu
// already held by LoadPTX or LoadPTXCached; a sync.Mutex is not reentrant, so
// reading this under mu deadlocks the first Build that ever runs. That was
// not reasoned out in advance -- it was written that way, and it hung.
//
// It lives in a package-level table rather than in the struct because the
// struct is not what the driver poisoned. NewContext(0) twice retains the
// same primary context and hands back two *Context values around one
// CUcontext; a marker on whichever wrapper happened to launch the faulting
// kernel left the other one handing out bare 719s, which is the obscurity
// ContextPoisonedError exists to replace. The entry outlives Close on
// purpose, and that is measured rather than assumed: after a trap
// cuDevicePrimaryCtxRetain fails too, so a replacement Context for that
// device is already unusable when it is made.
//
// Keyed by device ordinal and not process-wide, although the measurement here
// found the process finished with CUDA altogether. That machine has one GPU.
// Saying a fault on device 0 implies anything about device 1 would be a claim
// nothing has measured, and this file does not make those.
var (
	poisonMu sync.Mutex
	poison   = map[int32]*atomic.Pointer[ContextPoisonedError]{}
)

func poisonFor(dev int32) *atomic.Pointer[ContextPoisonedError] {
	poisonMu.Lock()
	defer poisonMu.Unlock()
	p, ok := poison[dev]
	if !ok {
		p = &atomic.Pointer[ContextPoisonedError]{}
		poison[dev] = p
	}
	return p
}

// NewContext retains the primary context of the given device ordinal.
func NewContext(device int) (*Context, error) {
	if err := Init(); err != nil {
		return nil, err
	}
	c := &Context{}
	if err := check(cuDeviceGet(&c.dev, int32(device)), "cuDeviceGet"); err != nil {
		return nil, err
	}
	// Before the retain, which is the first call here that can fault, and
	// before the bind at the bottom, which reads it.
	c.poisoned = poisonFor(c.dev)
	if err := c.fault(check(cuDevicePrimaryCtxRetain(&c.ctx, c.dev), "cuDevicePrimaryCtxRetain")); err != nil {
		// The driver's own error, not the poison marker, deliberately: on the
		// machine this was measured on the retain is what fails after a trap,
		// and that measurement is what docs/toolchain.md rests on. A caller
		// who just asked for a context needs no explanation of which call
		// went wrong.
		return nil, err
	}
	// The retain succeeding does not mean the device is usable -- another
	// wrapper may already have seen a sticky fault on this same primary
	// context. Refused here rather than at the bind below, which would hand
	// back a live-looking *Context beside its error.
	if p := c.poisoned.Load(); p != nil {
		return nil, p
	}
	var major, minor int32
	if err := check(cuDeviceGetAttribute(&major, attrComputeCapabilityMajor, c.dev), "cuDeviceGetAttribute"); err != nil {
		return nil, err
	}
	if err := check(cuDeviceGetAttribute(&minor, attrComputeCapabilityMinor, c.dev), "cuDeviceGetAttribute"); err != nil {
		return nil, err
	}
	c.ccMajor, c.ccMinor = int(major), int(minor)
	var shared int32
	if err := check(cuDeviceGetAttribute(&shared, attrMaxSharedMemoryPerBlock, c.dev), "cuDeviceGetAttribute"); err != nil {
		return nil, err
	}
	c.maxShared = int(shared)
	// One bind here as a check that the context can be made current at all,
	// so a caller learns about a context they cannot use now rather than at
	// their first allocation. It is not what makes the calls below work:
	// each of those binds on its own locked thread, because this one is
	// undone by the first goroutine migration.
	return c, c.bind()
}

// call runs one driver entry point with this context current on an OS thread
// that cannot change underneath it. Every method here goes through it, which
// is what makes it the one place a sticky fault has to be reported from.
//
// The lock is the whole point, and it is why bind and the call it protects
// cannot be two statements in the caller. cuCtxSetCurrent binds a context to
// the *calling OS thread*; Go may move a goroutine to another thread at any
// preemption point, and a driver call that lands on a thread this context was
// never made current on is answered with CUDA_ERROR_INVALID_CONTEXT. That is
// not a data race and the race detector cannot see it. See
// docs/decisions.md#a-driver-call-holds-its-os-thread.
//
// The cost is one thread pinned for the duration of one driver call. Those
// calls are microseconds to milliseconds and block anyway, so the scheduler
// loses nothing it could have used; LockOSThread itself is a couple of
// pointer writes.
func (c *Context) call(op string, fn func() Result) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := c.bind(); err != nil {
		return err
	}
	return c.fault(check(fn(), op))
}

// bind makes this context current on the calling thread.
//
// It is unexported and deliberately small: call is its only caller besides
// NewContext, because a bind that is not followed by the driver call on the
// same locked thread is a bind that may already have been undone.
func (c *Context) bind() error {
	if p := c.poisoned.Load(); p != nil {
		return p
	}
	return check(cuCtxSetCurrent(c.ctx), "cuCtxSetCurrent")
}

// sticky lists the result codes that leave the CUDA context unusable.
//
// The driver's word for these is "sticky": the fault happened on the device
// with no way to unwind it, so the driver fails every subsequent call in that
// context with the same code and offers no reset. What was measured here, on
// a T550 with driver 580, is stronger than that and is the reason
// ContextPoisonedError says what it says: after a __trap() not only does the
// poisoned context keep returning 719 -- from Sync, from cuMemAlloc, from
// cuMemcpyDtoH and from the cuModuleUnload inside Close -- but
// cuDevicePrimaryCtxRetain fails with 719 as well, so a *replacement* context
// cannot be made either. The process is finished with CUDA, not just the
// context.
//
// Only 719 was measured. The rest are on the list because the CUDA
// documentation describes them as sticky, and treating a sticky error as
// recoverable is the failure this exists to prevent -- the cost of being
// wrong in the other direction is one misleading sentence, not a hang.
// See docs/toolchain.md#what-the-host-sees-after-a-trap.
func sticky(r Result) bool {
	switch r {
	case ErrIllegalAddress, ErrLaunchTimeout, ErrIllegalInstruction,
		ErrMisalignedAddress, ErrLaunchFailed:
		return true
	}
	return false
}

// fault records a sticky error so that the calls after it say what happened
// rather than repeating the code, and passes every error through unchanged.
//
// Only the first one is kept. A sticky fault makes every subsequent call fail
// with the same code, so the second is a consequence of the first and saying
// so is the whole point.
//
// Every call that can be the one to observe a device-side fault goes through
// here, not just Sync and the launch. A host program that allocates or copies
// after launching is the ordinary shape, so cuMemAlloc, cuMemFree and both
// copies are just as likely to be where the driver first admits to 719 -- and
// a call that recognises the code but does not record it leaves the *next*
// one repeating the bare number.
func (c *Context) fault(err error) error {
	var e *Error
	if !errors.As(err, &e) || !sticky(e.Code) {
		return err
	}
	c.poisoned.CompareAndSwap(nil, &ContextPoisonedError{Op: e.Op, Code: e.Code})
	return err
}

// Arch reports the virtual architecture of the device, e.g. "compute_75". It
// is what NVRTC's --gpu-architecture expects, and ParseArch reads it back.
func (c *Context) Arch() string { return fmt.Sprintf("compute_%d%d", c.ccMajor, c.ccMinor) }

// ComputeCapability reports the device's compute capability, e.g. (7, 5) for a
// Turing card. It is the same fact Arch spells as a string, in the form a
// caller needs to decide whether a feature is available on this device.
func (c *Context) ComputeCapability() (major, minor int) { return c.ccMajor, c.ccMinor }

// MaxSharedMemPerBlock reports how much shared memory a single block may use,
// in bytes (CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK); 48 KiB on most
// devices, and the ceiling a generated kernel's tile buffers have to fit into.
//
// Zero means "unknown" and is what a build without the "cuda" tag always
// reports, since there is no device to ask. Callers that size or validate a
// kernel against this limit must treat zero as "skip the check" rather than as
// a device with no shared memory at all.
func (c *Context) MaxSharedMemPerBlock() int { return c.maxShared }

// Name reports the device name.
func (c *Context) Name() string {
	buf := make([]byte, 256)
	var pin runtime.Pinner
	pin.Pin(&buf[0])
	defer pin.Unpin()
	if err := check(cuDeviceGetName(&buf[0], int32(len(buf)), c.dev), "cuDeviceGetName"); err != nil {
		return "unknown"
	}
	return goString(&buf[0])
}

// Close unloads every module the context owns and then releases the primary
// context.
//
// The order matters: a CUmodule lives inside the context, so once the last
// reference to the primary context is gone there is nothing left to unload it
// from and the modules leak until the process exits.
//
// Close is idempotent. Tests release their context from t.Cleanup while other
// code may still defer a Close of its own, and cuDevicePrimaryCtxRelease
// decrements a reference count the driver shares with everything else in the
// process: releasing twice would tear down a primary context that another part
// of the program still believes it holds.
func (c *Context) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	c.closed = true

	// Unload in reverse load order, and keep going after a failure: every
	// module has to be dealt with before the context goes away, so the first
	// error is reported but none of the others are allowed to stop the walk.
	var firstErr error
	for i := len(c.modules) - 1; i >= 0; i-- {
		if err := c.modules[i].unloadLocked(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	c.modules, c.cache = nil, nil

	if err := check(cuDevicePrimaryCtxRelease(c.dev), "cuDevicePrimaryCtxRelease"); err != nil && firstErr == nil {
		firstErr = err
	}
	return firstErr
}

// Sync blocks until all work on the context has completed.
func (c *Context) Sync() error {
	return c.call("cuCtxSynchronize", cuCtxSynchronize)
}

// Alloc reserves n bytes of device memory.
func (c *Context) Alloc(n int) (DevPtr, error) {
	var p uint64
	if err := c.call("cuMemAlloc", func() Result { return cuMemAlloc(&p, uint64(n)) }); err != nil {
		return 0, err
	}
	return DevPtr(p), nil
}

// Free releases device memory.
func (c *Context) Free(p DevPtr) error {
	return c.call("cuMemFree", func() Result { return cuMemFree(uint64(p)) })
}

func (c *Context) copyHtoD(dst DevPtr, src unsafe.Pointer, n int) error {
	return c.call("cuMemcpyHtoD", func() Result {
		return cuMemcpyHtoD(uint64(dst), src, uint64(n))
	})
}

func (c *Context) copyDtoH(dst unsafe.Pointer, src DevPtr, n int) error {
	return c.call("cuMemcpyDtoH", func() Result {
		return cuMemcpyDtoH(dst, uint64(src), uint64(n))
	})
}

// Module is a loaded PTX module. It belongs to the context it was loaded into
// and becomes invalid when that context is closed.
type Module struct {
	c uintptr
	x *Context
}

// LoadPTX loads PTX (NUL-terminated on the driver's side) into the context.
//
// The context takes ownership of the module: Close unloads it, so a caller
// only needs Module.Unload to drop one earlier than that.
func (c *Context) LoadPTX(ptx []byte) (*Module, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.loadPTXLocked(ptx)
}

// loadPTXLocked does the driver call and the bookkeeping with c.mu held, so
// that LoadPTXCached can load a module without dropping the lock it compiled
// under.
func (c *Context) loadPTXLocked(ptx []byte) (*Module, error) {
	if c.closed {
		return nil, ErrContextClosed
	}

	// cuModuleLoadData reads a NUL-terminated image. The copy is not avoidable
	// by appending in place: ptx belongs to the caller, and a spare byte of
	// capacity would let append write into memory they still own.
	img := make([]byte, len(ptx)+1)
	copy(img, ptx)
	var pin runtime.Pinner
	pin.Pin(&img[0])
	defer pin.Unpin()

	m := &Module{x: c}
	if err := c.call("cuModuleLoadData", func() Result {
		return cuModuleLoadData(&m.c, unsafe.Pointer(&img[0]))
	}); err != nil {
		return nil, err
	}
	c.modules = append(c.modules, m)
	return m, nil
}

// LoadPTXCached loads a module into the context at most once per key, and is
// how a JIT layer avoids recompiling and reloading a source it has already
// seen without ever caching a module beyond the life of its context.
//
// build runs only on a miss, with the context's lock held, so concurrent
// callers asking for the same key compile once and share the result. What this
// cache is for is the module alone: a CUmodule dies with its context, which is
// the one thing a package-level map cannot get right. Everything else a caller
// wants to remember about a build -- the PTX, the compiler log, where the
// bytes came from -- is inert and belongs in the caller's own map, so it is
// not dragged through this API as an untyped value.
func (c *Context) LoadPTXCached(key string, build func() ([]byte, error)) (*Module, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil, ErrContextClosed
	}
	if m, ok := c.cache[key]; ok {
		return m, nil
	}
	ptx, err := build()
	if err != nil {
		return nil, err
	}
	m, err := c.loadPTXLocked(ptx)
	if err != nil {
		return nil, err
	}
	if c.cache == nil {
		c.cache = make(map[string]*Module)
	}
	c.cache[key] = m
	return m, nil
}

// Unload releases the module ahead of its context's Close, which is the only
// reason to call it: Close unloads whatever is still loaded anyway.
//
// It is idempotent, so unloading a module that Close has already dealt with
// (or unloading twice) is a no-op rather than a second cuModuleUnload on a
// stale handle. Functions looked up in the module do not survive it.
func (m *Module) Unload() error {
	m.x.mu.Lock()
	defer m.x.mu.Unlock()
	m.x.forgetLocked(m)
	return m.unloadLocked()
}

// unloadLocked unloads the module once, with its context's lock held. Clearing
// the handle first is what makes a second call, from Close or from the caller,
// harmless.
func (m *Module) unloadLocked() error {
	if m.c == 0 {
		return nil
	}
	h := m.c
	m.c = 0
	return m.x.call("cuModuleUnload", func() Result { return cuModuleUnload(h) })
}

// forgetLocked drops every reference the context holds to m, so that an
// explicitly unloaded module is neither unloaded again by Close nor served
// from the cache afterwards; the next request for that key rebuilds.
func (c *Context) forgetLocked(m *Module) {
	for i, other := range c.modules {
		if other == m {
			c.modules = append(c.modules[:i], c.modules[i+1:]...)
			break
		}
	}
	for key, cached := range c.cache {
		if cached == m {
			delete(c.cache, key)
		}
	}
}

// Function is an entry point inside a Module.
type Function struct {
	f uintptr
	x *Context
}

// Function looks up a kernel by its (extern "C") name.
//
// This one used to reach the driver with no bind at all, which worked only
// because every caller had just loaded the module and so had left the context
// current on that thread by luck. It goes through call like the rest now.
func (m *Module) Function(name string) (*Function, error) {
	f := &Function{x: m.x}
	if err := m.x.call("cuModuleGetFunction "+name, func() Result {
		return cuModuleGetFunction(&f.f, m.c, name)
	}); err != nil {
		return nil, err
	}
	return f, nil
}

// LaunchSync runs the kernel and waits for it to finish.
//
// The name says so because the wait is the surprising part: cuLaunchKernel is
// asynchronous, so the plain name belongs to the asynchronous launch that is
// still to come. Having both under one name for a while would be worse than
// renaming this once.
//
// The driver receives an array of pointers to the parameter values. Both the
// values and the array itself are Go memory here, pinned for the duration of
// the call: the rule that C must not be shown Go memory containing Go pointers
// is about the collector moving or freeing it, and that is what a Pinner
// suspends. The cgo implementation malloc'd both instead; this is a wash for
// cost -- two C allocations become two Go ones -- and is done this way because
// there is no C allocator left to call, not because it is faster.
func (f *Function) LaunchSync(grid, block Dim3, sharedBytes int, args ...Arg) error {
	// The null stream, which is what "synchronous" has always meant here.
	if err := f.launch(0, grid, block, sharedBytes, args); err != nil {
		return err
	}
	// A kernel fault is asynchronous: measured on a T550, cuLaunchKernel
	// returns success and the trap surfaces here. Sync records it. It locks a
	// thread of its own, which need not be this one -- cuCtxSynchronize waits
	// on the context, not on the thread that issued the launch.
	return f.x.Sync()
}

// launch marshals the parameters and issues cuLaunchKernel on one stream.
//
// LaunchSync and Function.Launch differ in the stream and in whether anything
// waits afterwards; everything above that is this, once.
func (f *Function) launch(stream uintptr, grid, block Dim3, sharedBytes int, args []Arg) error {
	// Each parameter gets its own width, rounded up to 8, rather than a fixed
	// 8-byte slot: a struct passed by value is as wide as the struct, and
	// copying it into an 8-byte slot would overwrite the parameters after it.
	// The rounding keeps every slot 8-aligned, which is enough because no type
	// the subset accepts is aligned more strictly than a double.
	const align = 8
	offsets := make([]int, len(args))
	total := 0
	for i, a := range args {
		offsets[i] = total
		total += (max(len(a.b), align) + align - 1) / align * align
	}
	// At least one slot, so &store[0] exists and the allocator gives it the
	// 8-byte alignment the driver reads these through.
	store := make([]byte, max(total, align))
	table := make([]unsafe.Pointer, max(len(args), 1))

	var pin runtime.Pinner
	pin.Pin(&store[0])
	pin.Pin(&table[0])
	defer pin.Unpin()

	for i, a := range args {
		copy(store[offsets[i]:], a.b)
		table[i] = unsafe.Pointer(&store[offsets[i]])
	}

	var params unsafe.Pointer
	if len(args) > 0 {
		params = unsafe.Pointer(&table[0])
	}
	// Marshalling and pinning happen above, off the locked thread: only the
	// driver call itself has to stay on one.
	//
	// The pin is released when this returns, which is correct even for an
	// asynchronous launch and is worth saying because it looks wrong. The
	// driver copies the parameter block out during cuLaunchKernel itself --
	// it has to, since the caller is free to reuse those variables on the
	// next line and CUDA promises that works. What outlives the call is the
	// *device* memory the parameters point at, which is not Go memory and is
	// the caller's to keep alive. TestLaunchAsyncSurvivesGC is the check.
	return f.x.call("cuLaunchKernel", func() Result {
		return cuLaunchKernel(f.f,
			grid.X, grid.Y, grid.Z,
			block.X, block.Y, block.Z,
			uint32(sharedBytes), stream, params, nil)
	})
}

// Compile JIT-compiles CUDA C source to PTX with NVRTC.
//
// The compiler log comes back on success and failure alike: on success it
// carries the warnings a generated kernel provoked, and on failure it is the
// only thing that explains what the kernel got wrong, which is why a refused
// compilation is a *CompileError holding the whole log rather than a code.
//
// arch is required; opts are the numerical settings, of which WithFastMath is
// currently the only one. Everything not passed is NVRTC's default, and
// NUMERICS.md says what those defaults are and how they were measured.
func Compile(src, name, arch string, opts ...CompileOption) (*PTX, error) {
	if err := loadNVRTC(); err != nil {
		return nil, err
	}

	var prog uintptr
	if r := nvrtcCreateProgram(&prog, src, name, 0, nil, nil); r != nvrtcSuccess {
		return nil, nvrtcError(r, "nvrtcCreateProgram")
	}
	// The result is discarded deliberately. This runs on the way out of a
	// function that already has its answer -- a PTX or a compile error -- and
	// failing to free a program handle is neither recoverable here nor
	// something the caller could act on.
	defer func() { _ = nvrtcDestroyProgram(&prog) }()

	// The options array is a char** of NUL-terminated strings. purego converts
	// a string *argument* for us, but this one is an array, so it is built by
	// hand and pinned like any other pointer-to-pointer handed to C.
	//
	// Every element has to stay pinned for the whole call, so the NUL-
	// terminated copies are kept in args until nvrtcCompileProgram returns:
	// pinning argv alone would leave the strings it points at free to move.
	var pin runtime.Pinner
	strs := nvrtcOptions(arch, opts)
	args := make([][]byte, len(strs))
	argv := make([]*byte, len(strs))
	for i, o := range strs {
		args[i] = append([]byte(o), 0)
		argv[i] = &args[i][0]
		pin.Pin(argv[i])
	}
	pin.Pin(&argv[0])
	cres := nvrtcCompileProgram(prog, int32(len(argv)), unsafe.Pointer(&argv[0]))
	pin.Unpin()

	// The log is read before the status is acted on, because it is the part of
	// a failed compilation worth keeping.
	var log string
	var logSize uint64
	if nvrtcGetProgramLogSize(prog, &logSize) == nvrtcSuccess && logSize > 1 {
		buf := make([]byte, logSize)
		var lpin runtime.Pinner
		lpin.Pin(&buf[0])
		ok := nvrtcGetProgramLog(prog, unsafe.Pointer(&buf[0])) == nvrtcSuccess
		lpin.Unpin()
		if ok {
			log = goString(&buf[0])
		}
	}
	if cres != nvrtcSuccess {
		return nil, &CompileError{Name: name, Arch: arch, Code: cres, Log: log}
	}

	var ptxSize uint64
	if r := nvrtcGetPTXSize(prog, &ptxSize); r != nvrtcSuccess {
		return nil, nvrtcError(r, "nvrtcGetPTXSize")
	}
	buf := make([]byte, ptxSize)
	var ppin runtime.Pinner
	ppin.Pin(&buf[0])
	r := nvrtcGetPTX(prog, unsafe.Pointer(&buf[0]))
	ppin.Unpin()
	if r != nvrtcSuccess {
		return nil, nvrtcError(r, "nvrtcGetPTX")
	}
	return &PTX{Bytes: []byte(goString(&buf[0])), Log: log, Arch: arch}, nil
}

// NVRTCVersion reports the version of the NVRTC library this process loaded.
// It decides which CUDA C a generated kernel may use and which architectures
// Compile accepts, so it is worth putting into a bug report next to the
// driver's own version.
func NVRTCVersion() (major, minor int, err error) {
	if err := loadNVRTC(); err != nil {
		return 0, 0, err
	}
	var cMajor, cMinor int32
	if r := nvrtcVersion(&cMajor, &cMinor); r != nvrtcSuccess {
		return 0, 0, nvrtcError(r, "nvrtcVersion")
	}
	return int(cMajor), int(cMinor), nil
}

// The two success codes, named here so the call sites read as they did when
// they came from the headers.
const (
	cudaSuccess  Result      = 0
	nvrtcSuccess NVRTCResult = 0
)

// check turns a driver status into a *Error that keeps the code, so that a
// caller can ask errors.Is which failure this was instead of matching on the
// message. The description is taken here rather than remembered for later
// because cuGetErrorString needs a driver, and these errors are routinely
// formatted somewhere that has none.
//
// It returns error rather than *Error on purpose: several callers hand the
// result straight back as an error, and a typed nil pointer travelling through
// that interface would be a non-nil error reporting success.
func check(r Result, op string) error {
	if r == cudaSuccess {
		return nil
	}
	// The driver owns the string it hands back, so it outlives the call and
	// needs no pinning; goString copies it into Go memory.
	var str *byte
	// The result is discarded deliberately: cuGetErrorString fails only for a
	// code it does not recognise, and leaves str nil when it does, which
	// goString turns into "". The code itself still prints through Error.Code
	// from the pure-Go name table, so a check here could add nothing.
	_ = cuGetErrorString(r, &str)
	return &Error{Op: op, Code: r, Desc: goString(str)}
}

// nvrtcError is check for NVRTC, and returns error for the same reason.
func nvrtcError(r NVRTCResult, op string) error {
	if r == nvrtcSuccess {
		return nil
	}
	return &NVRTCError{Op: op, Code: r, Desc: nvrtcGetErrorString(r)}
}
