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
type Context struct {
	dev  C.CUdevice
	ctx  C.CUcontext
	arch string
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
	c.arch = fmt.Sprintf("compute_%d%d", int(major), int(minor))
	return c, c.bind()
}

func (c *Context) bind() error {
	return check(C.cuCtxSetCurrent(c.ctx), "cuCtxSetCurrent")
}

// Arch reports the virtual architecture of the device, e.g. "compute_75".
func (c *Context) Arch() string { return c.arch }

// Name reports the device name.
func (c *Context) Name() string {
	buf := make([]C.char, 256)
	if err := check(C.cuDeviceGetName(&buf[0], C.int(len(buf)), c.dev), "cuDeviceGetName"); err != nil {
		return "unknown"
	}
	return C.GoString(&buf[0])
}

// Close releases the primary context.
func (c *Context) Close() error {
	return check(C.cuDevicePrimaryCtxRelease(c.dev), "cuDevicePrimaryCtxRelease")
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

// Module is a loaded PTX module.
type Module struct {
	c C.CUmodule
	x *Context
}

// LoadPTX loads PTX (NUL-terminated on the C side) into the context.
func (c *Context) LoadPTX(ptx []byte) (*Module, error) {
	if err := c.bind(); err != nil {
		return nil, err
	}
	src := C.CString(string(ptx))
	defer C.free(unsafe.Pointer(src))
	m := &Module{x: c}
	if err := check(C.cuModuleLoadData(&m.c, unsafe.Pointer(src)), "cuModuleLoadData"); err != nil {
		return nil, err
	}
	return m, nil
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

// Compile JIT-compiles CUDA C source to PTX with NVRTC. The compiler log is
// returned on success and failure alike; on failure it is the only thing that
// explains what the generated kernel got wrong.
func Compile(src, name, arch string) (ptx []byte, log string, err error) {
	csrc, cname := C.CString(src), C.CString(name)
	defer C.free(unsafe.Pointer(csrc))
	defer C.free(unsafe.Pointer(cname))

	var prog C.nvrtcProgram
	if r := C.nvrtcCreateProgram(&prog, csrc, cname, 0, nil, nil); r != C.NVRTC_SUCCESS {
		return nil, "", nvrtcErr(r, "nvrtcCreateProgram")
	}
	defer C.nvrtcDestroyProgram(&prog)

	opt := C.CString("--gpu-architecture=" + arch)
	defer C.free(unsafe.Pointer(opt))
	opts := [1]*C.char{opt}
	cres := C.nvrtcCompileProgram(prog, 1, &opts[0])

	var logSize C.size_t
	if C.nvrtcGetProgramLogSize(prog, &logSize) == C.NVRTC_SUCCESS && logSize > 1 {
		buf := make([]C.char, logSize)
		if C.nvrtcGetProgramLog(prog, &buf[0]) == C.NVRTC_SUCCESS {
			log = C.GoString(&buf[0])
		}
	}
	if cres != C.NVRTC_SUCCESS {
		return nil, log, fmt.Errorf("%w\n%s", nvrtcErr(cres, "nvrtcCompileProgram"), log)
	}

	var ptxSize C.size_t
	if r := C.nvrtcGetPTXSize(prog, &ptxSize); r != C.NVRTC_SUCCESS {
		return nil, log, nvrtcErr(r, "nvrtcGetPTXSize")
	}
	buf := make([]C.char, ptxSize)
	if r := C.nvrtcGetPTX(prog, &buf[0]); r != C.NVRTC_SUCCESS {
		return nil, log, nvrtcErr(r, "nvrtcGetPTX")
	}
	return []byte(C.GoString(&buf[0])), log, nil
}

func check(r C.CUresult, op string) error {
	if r == C.CUDA_SUCCESS {
		return nil
	}
	var name, str *C.char
	C.cuGetErrorName(r, &name)
	C.cuGetErrorString(r, &str)
	return fmt.Errorf("cuda: %s failed: %s (%s)", op, C.GoString(str), C.GoString(name))
}

func nvrtcErr(r C.nvrtcResult, op string) error {
	return fmt.Errorf("nvrtc: %s failed: %s", op, C.GoString(C.nvrtcGetErrorString(r)))
}
