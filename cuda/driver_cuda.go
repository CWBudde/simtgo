//go:build cuda

package cuda

/*
#cgo CFLAGS: -I/usr/local/cuda/include
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -lnvrtc -lcuda
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <nvrtc.h>
*/
import "C"

import (
	"fmt"
	"sync"
	"unsafe"
)

var initOnce struct {
	sync.Once
	err error
}

// Init initialises the CUDA driver. It is safe to call repeatedly.
func Init() error {
	initOnce.Do(func() { initOnce.err = check(C.cuInit(0), "cuInit") })
	return initOnce.err
}

// Available reports whether a usable CUDA device is present.
func Available() bool {
	if Init() != nil {
		return false
	}
	var n C.int
	if err := check(C.cuDeviceGetCount(&n), "cuDeviceGetCount"); err != nil {
		return false
	}
	return n > 0
}

// Context owns a device's primary context.
//
// The driver API binds contexts to OS threads, and goroutines migrate between
// them, so every method calls cuCtxSetCurrent before touching the driver
// rather than relying on runtime.LockOSThread.
//
// A context also owns every module loaded into it, including the compilation
// cache behind LoadPTXCached. A CUmodule is only meaningful inside the context
// it was loaded into, so that cache has to live here rather than in a
// package-level map: a process-wide cache outlives the contexts it caches for
// and would hand out handles belonging to a context that has already been
// released. Close unloads what is left, so no module can survive its context.
type Context struct {
	dev C.CUdevice
	ctx C.CUcontext

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
	modules []*Module             // every module loaded here, in load order
	cache   map[string]cacheEntry // the subset loaded through LoadPTXCached
	closed  bool
}

// cacheEntry is one cached module together with the value its builder attached
// to it. The extra value is opaque to this package; see LoadPTXCached.
type cacheEntry struct {
	mod   *Module
	extra any
}

// NewContext retains the primary context of the given device ordinal.
func NewContext(device int) (*Context, error) {
	if err := Init(); err != nil {
		return nil, err
	}
	c := &Context{}
	if err := check(C.cuDeviceGet(&c.dev, C.int(device)), "cuDeviceGet"); err != nil {
		return nil, err
	}
	if err := check(C.cuDevicePrimaryCtxRetain(&c.ctx, c.dev), "cuDevicePrimaryCtxRetain"); err != nil {
		return nil, err
	}
	var major, minor C.int
	if err := check(C.cuDeviceGetAttribute(&major, C.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, c.dev), "cuDeviceGetAttribute"); err != nil {
		return nil, err
	}
	if err := check(C.cuDeviceGetAttribute(&minor, C.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, c.dev), "cuDeviceGetAttribute"); err != nil {
		return nil, err
	}
	c.ccMajor, c.ccMinor = int(major), int(minor)
	var shared C.int
	if err := check(C.cuDeviceGetAttribute(&shared, C.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, c.dev), "cuDeviceGetAttribute"); err != nil {
		return nil, err
	}
	c.maxShared = int(shared)
	return c, c.bind()
}

func (c *Context) bind() error {
	return check(C.cuCtxSetCurrent(c.ctx), "cuCtxSetCurrent")
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
	buf := make([]C.char, 256)
	if err := check(C.cuDeviceGetName(&buf[0], C.int(len(buf)), c.dev), "cuDeviceGetName"); err != nil {
		return "unknown"
	}
	return C.GoString(&buf[0])
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

	if err := check(C.cuDevicePrimaryCtxRelease(c.dev), "cuDevicePrimaryCtxRelease"); err != nil && firstErr == nil {
		firstErr = err
	}
	return firstErr
}

// Sync blocks until all work on the context has completed.
func (c *Context) Sync() error {
	if err := c.bind(); err != nil {
		return err
	}
	return check(C.cuCtxSynchronize(), "cuCtxSynchronize")
}

// Alloc reserves n bytes of device memory.
func (c *Context) Alloc(n int) (DevPtr, error) {
	if err := c.bind(); err != nil {
		return 0, err
	}
	var p C.CUdeviceptr
	if err := check(C.cuMemAlloc(&p, C.size_t(n)), "cuMemAlloc"); err != nil {
		return 0, err
	}
	return DevPtr(p), nil
}

// Free releases device memory.
func (c *Context) Free(p DevPtr) error {
	if err := c.bind(); err != nil {
		return err
	}
	return check(C.cuMemFree(C.CUdeviceptr(p)), "cuMemFree")
}

func (c *Context) copyHtoD(dst DevPtr, src unsafe.Pointer, n int) error {
	if err := c.bind(); err != nil {
		return err
	}
	return check(C.cuMemcpyHtoD(C.CUdeviceptr(dst), src, C.size_t(n)), "cuMemcpyHtoD")
}

func (c *Context) copyDtoH(dst unsafe.Pointer, src DevPtr, n int) error {
	if err := c.bind(); err != nil {
		return err
	}
	return check(C.cuMemcpyDtoH(dst, C.CUdeviceptr(src), C.size_t(n)), "cuMemcpyDtoH")
}

// Module is a loaded PTX module. It belongs to the context it was loaded into
// and becomes invalid when that context is closed.
type Module struct {
	c C.CUmodule
	x *Context
}

// LoadPTX loads PTX (NUL-terminated on the C side) into the context.
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
	if err := c.bind(); err != nil {
		return nil, err
	}
	src := C.CString(string(ptx))
	defer C.free(unsafe.Pointer(src))
	m := &Module{x: c}
	if err := check(C.cuModuleLoadData(&m.c, unsafe.Pointer(src)), "cuModuleLoadData"); err != nil {
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
// callers asking for the same key compile once and share the result. Besides
// the PTX to load, build returns a value this package stores but never looks
// at and hands back on every subsequent hit: internal/jit uses it to carry the
// generated PTX and the NVRTC log, so a cache hit can report exactly what the
// original miss reported. The split is deliberate -- the caller keeps the
// compile and dump policy, the context keeps the module's lifetime.
func (c *Context) LoadPTXCached(key string, build func() (ptx []byte, extra any, err error)) (*Module, any, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil, nil, ErrContextClosed
	}
	if e, ok := c.cache[key]; ok {
		return e.mod, e.extra, nil
	}
	ptx, extra, err := build()
	if err != nil {
		return nil, nil, err
	}
	m, err := c.loadPTXLocked(ptx)
	if err != nil {
		return nil, nil, err
	}
	if c.cache == nil {
		c.cache = make(map[string]cacheEntry)
	}
	c.cache[key] = cacheEntry{mod: m, extra: extra}
	return m, extra, nil
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
	if m.c == nil {
		return nil
	}
	h := m.c
	m.c = nil
	if err := m.x.bind(); err != nil {
		return err
	}
	return check(C.cuModuleUnload(h), "cuModuleUnload")
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
	for key, e := range c.cache {
		if e.mod == m {
			delete(c.cache, key)
		}
	}
}

// Function is an entry point inside a Module.
type Function struct {
	f C.CUfunction
	x *Context
}

// Function looks up a kernel by its (extern "C") name.
func (m *Module) Function(name string) (*Function, error) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	f := &Function{x: m.x}
	if err := check(C.cuModuleGetFunction(&f.f, m.c, cname), "cuModuleGetFunction "+name); err != nil {
		return nil, err
	}
	return f, nil
}

// Launch runs the kernel and waits for it to finish.
//
// Kernel parameters are copied into C-allocated memory: the driver receives an
// array of pointers, and cgo forbids handing it an array that itself holds Go
// pointers.
func (f *Function) Launch(grid, block Dim3, sharedBytes int, args ...Arg) error {
	if err := f.x.bind(); err != nil {
		return err
	}

	const slot = 8 // one 8-byte aligned slot per parameter
	store := C.malloc(C.size_t(len(args) * slot))
	defer C.free(store)
	ptrs := C.malloc(C.size_t(len(args)) * C.size_t(unsafe.Sizeof(uintptr(0))))
	defer C.free(ptrs)

	table := unsafe.Slice((*unsafe.Pointer)(ptrs), max(len(args), 1))
	for i, a := range args {
		dst := unsafe.Add(store, i*slot)
		C.memcpy(dst, unsafe.Pointer(&a.b[0]), C.size_t(len(a.b)))
		table[i] = dst
	}

	var params *unsafe.Pointer
	if len(args) > 0 {
		params = (*unsafe.Pointer)(ptrs)
	}
	err := check(C.cuLaunchKernel(f.f,
		C.uint(grid.X), C.uint(grid.Y), C.uint(grid.Z),
		C.uint(block.X), C.uint(block.Y), C.uint(block.Z),
		C.uint(sharedBytes), nil, params, nil), "cuLaunchKernel")
	if err != nil {
		return err
	}
	return f.x.Sync()
}

// Compile JIT-compiles CUDA C source to PTX with NVRTC.
//
// The compiler log comes back on success and failure alike: on success it
// carries the warnings a generated kernel provoked, and on failure it is the
// only thing that explains what the kernel got wrong, which is why a refused
// compilation is a *CompileError holding the whole log rather than a code.
func Compile(src, name, arch string) (*PTX, error) {
	csrc, cname := C.CString(src), C.CString(name)
	defer C.free(unsafe.Pointer(csrc))
	defer C.free(unsafe.Pointer(cname))

	var prog C.nvrtcProgram
	if r := C.nvrtcCreateProgram(&prog, csrc, cname, 0, nil, nil); r != C.NVRTC_SUCCESS {
		return nil, nvrtcErr(r, "nvrtcCreateProgram")
	}
	defer C.nvrtcDestroyProgram(&prog)

	opt := C.CString("--gpu-architecture=" + arch)
	defer C.free(unsafe.Pointer(opt))
	opts := [1]*C.char{opt}
	cres := C.nvrtcCompileProgram(prog, 1, &opts[0])

	// The log is read before the status is acted on, because it is the part of
	// a failed compilation worth keeping.
	var log string
	var logSize C.size_t
	if C.nvrtcGetProgramLogSize(prog, &logSize) == C.NVRTC_SUCCESS && logSize > 1 {
		buf := make([]C.char, logSize)
		if C.nvrtcGetProgramLog(prog, &buf[0]) == C.NVRTC_SUCCESS {
			log = C.GoString(&buf[0])
		}
	}
	if cres != C.NVRTC_SUCCESS {
		return nil, &CompileError{Name: name, Arch: arch, Code: NVRTCResult(cres), Log: log}
	}

	var ptxSize C.size_t
	if r := C.nvrtcGetPTXSize(prog, &ptxSize); r != C.NVRTC_SUCCESS {
		return nil, nvrtcErr(r, "nvrtcGetPTXSize")
	}
	buf := make([]C.char, ptxSize)
	if r := C.nvrtcGetPTX(prog, &buf[0]); r != C.NVRTC_SUCCESS {
		return nil, nvrtcErr(r, "nvrtcGetPTX")
	}
	return &PTX{Bytes: []byte(C.GoString(&buf[0])), Log: log, Arch: arch}, nil
}

// NVRTCVersion reports the version of the NVRTC library this process is linked
// against. It decides which CUDA C a generated kernel may use and which
// architectures Compile accepts, so it is worth putting into a bug report next
// to the driver's own version.
func NVRTCVersion() (major, minor int, err error) {
	var maj, min C.int
	if r := C.nvrtcVersion(&maj, &min); r != C.NVRTC_SUCCESS {
		return 0, 0, nvrtcErr(r, "nvrtcVersion")
	}
	return int(maj), int(min), nil
}

// check turns a driver status into a *Error that keeps the code, so that a
// caller can ask errors.Is which failure this was instead of matching on the
// message. The description is taken here rather than remembered for later
// because cuGetErrorString needs a driver, and these errors are routinely
// formatted somewhere that has none.
//
// It returns error rather than *Error on purpose: several callers hand the
// result straight back as an error, and a typed nil pointer travelling through
// that interface would be a non-nil error reporting success.
func check(r C.CUresult, op string) error {
	if r == C.CUDA_SUCCESS {
		return nil
	}
	var str *C.char
	C.cuGetErrorString(r, &str)
	return &Error{Op: op, Code: Result(r), Desc: C.GoString(str)}
}

// nvrtcErr is check for NVRTC, and returns error for the same reason.
func nvrtcErr(r C.nvrtcResult, op string) error {
	if r == C.NVRTC_SUCCESS {
		return nil
	}
	return &NVRTCError{Op: op, Code: NVRTCResult(r), Desc: C.GoString(C.nvrtcGetErrorString(r))}
}
