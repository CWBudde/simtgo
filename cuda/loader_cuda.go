//go:build cuda

package cuda

import (
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// This file opens the two shared libraries and binds the entry points
// driver_cuda.go calls. Nothing here is cgo: purego resolves the symbols at
// run time, which is what lets this package build with CGO_ENABLED=0 on a
// machine that has never had a CUDA toolkit installed.
//
// Binding is lazy and happens at most once per library. A process that never
// touches a GPU never opens libcuda, and one with no CUDA at all gets a
// *LibraryError rather than a link failure it could not have recovered from.

// The driver entry points. The names on the right are the *exported* symbols,
// which are not always the names in cuda.h: the header #defines cuMemAlloc to
// cuMemAlloc_v2 and so on for everything whose signature changed when sizes
// went 64-bit. Both names are exported by libcuda, and the unsuffixed ones are
// the pre-CUDA-3.2 API that takes 32-bit sizes, so binding those would
// silently truncate every allocation and copy above 4 GiB.
//
// The rule is not "append _v2 and hope". Each name below was checked against
// nm -D on the local libcuda, and three of them are instructive:
// cuMemHostAlloc and cuMemFreeHost have no _v2 at all (cuMemAllocHost, a
// different and older call, does); and cuEventElapsedTime has one that only
// exists from CUDA 12.8. Binding that one would panic in bind() on every
// older driver -- deliberately, since a missing symbol here means the library
// is not libcuda -- so the unsuffixed name is the portable choice and the
// difference between them is not a width but an error-reporting nicety.
var (
	cuInit                    func(flags uint32) Result
	cuDeviceGetCount          func(count *int32) Result
	cuDeviceGet               func(dev *int32, ordinal int32) Result
	cuDeviceGetName           func(name *byte, length int32, dev int32) Result
	cuDeviceGetAttribute      func(value *int32, attrib uint32, dev int32) Result
	cuDevicePrimaryCtxRetain  func(ctx *uintptr, dev int32) Result
	cuDevicePrimaryCtxRelease func(dev int32) Result
	cuCtxSetCurrent           func(ctx uintptr) Result
	cuCtxSynchronize          func() Result
	cuMemAlloc                func(dptr *uint64, size uint64) Result
	cuMemFree                 func(dptr uint64) Result
	cuMemcpyHtoD              func(dst uint64, src unsafe.Pointer, size uint64) Result
	cuMemcpyDtoH              func(dst unsafe.Pointer, src uint64, size uint64) Result
	cuMemHostAlloc            func(pp *unsafe.Pointer, size uint64, flags uint32) Result
	cuMemFreeHost             func(p unsafe.Pointer) Result
	cuMemcpyHtoDAsync         func(dst uint64, src unsafe.Pointer, size uint64, stream uintptr) Result
	cuMemcpyDtoHAsync         func(dst unsafe.Pointer, src uint64, size uint64, stream uintptr) Result
	cuStreamCreate            func(stream *uintptr, flags uint32) Result
	cuStreamDestroy           func(stream uintptr) Result
	cuStreamSynchronize       func(stream uintptr) Result
	cuStreamQuery             func(stream uintptr) Result
	cuEventCreate             func(event *uintptr, flags uint32) Result
	cuEventDestroy            func(event uintptr) Result
	cuEventRecord             func(event, stream uintptr) Result
	cuEventSynchronize        func(event uintptr) Result
	cuEventElapsedTime        func(ms *float32, start, end uintptr) Result
	cuModuleLoadData          func(mod *uintptr, image unsafe.Pointer) Result
	cuModuleUnload            func(mod uintptr) Result
	cuModuleGetFunction       func(fn *uintptr, mod uintptr, name string) Result
	cuLaunchKernel            func(fn uintptr,
		gridX, gridY, gridZ uint32,
		blockX, blockY, blockZ uint32,
		sharedBytes uint32, stream uintptr,
		params unsafe.Pointer, extra unsafe.Pointer) Result
	cuGetErrorString func(code Result, str **byte) Result
)

// The NVRTC entry points.
var (
	nvrtcCreateProgram func(prog *uintptr, src, name string,
		numHeaders int32, headers, includeNames unsafe.Pointer) NVRTCResult
	nvrtcDestroyProgram    func(prog *uintptr) NVRTCResult
	nvrtcCompileProgram    func(prog uintptr, numOptions int32, options unsafe.Pointer) NVRTCResult
	nvrtcGetProgramLogSize func(prog uintptr, size *uint64) NVRTCResult
	nvrtcGetProgramLog     func(prog uintptr, buf unsafe.Pointer) NVRTCResult
	nvrtcGetPTXSize        func(prog uintptr, size *uint64) NVRTCResult
	nvrtcGetPTX            func(prog uintptr, buf unsafe.Pointer) NVRTCResult
	nvrtcVersion           func(major, minor *int32) NVRTCResult
	nvrtcGetErrorString    func(code NVRTCResult) string
)

var (
	driverOnce sync.Once
	driverErr  error

	nvrtcOnce    sync.Once
	nvrtcLoadErr error
)

// loadDriver opens libcuda and binds the driver API, at most once per process.
func loadDriver() error {
	driverOnce.Do(func() {
		h, err := openFirst("libcuda", envLibCUDA, driverCandidates())
		if err != nil {
			driverErr = err
			return
		}
		bind(h, map[string]any{
			"cuInit":                       &cuInit,
			"cuDeviceGetCount":             &cuDeviceGetCount,
			"cuDeviceGet":                  &cuDeviceGet,
			"cuDeviceGetName":              &cuDeviceGetName,
			"cuDeviceGetAttribute":         &cuDeviceGetAttribute,
			"cuDevicePrimaryCtxRetain":     &cuDevicePrimaryCtxRetain,
			"cuDevicePrimaryCtxRelease_v2": &cuDevicePrimaryCtxRelease,
			"cuCtxSetCurrent":              &cuCtxSetCurrent,
			"cuCtxSynchronize":             &cuCtxSynchronize,
			"cuMemAlloc_v2":                &cuMemAlloc,
			"cuMemFree_v2":                 &cuMemFree,
			"cuMemcpyHtoD_v2":              &cuMemcpyHtoD,
			"cuMemcpyDtoH_v2":              &cuMemcpyDtoH,
			"cuMemHostAlloc":               &cuMemHostAlloc,
			"cuMemFreeHost":                &cuMemFreeHost,
			"cuMemcpyHtoDAsync_v2":         &cuMemcpyHtoDAsync,
			"cuMemcpyDtoHAsync_v2":         &cuMemcpyDtoHAsync,
			"cuStreamCreate":               &cuStreamCreate,
			"cuStreamDestroy_v2":           &cuStreamDestroy,
			"cuStreamSynchronize":          &cuStreamSynchronize,
			"cuStreamQuery":                &cuStreamQuery,
			"cuEventCreate":                &cuEventCreate,
			"cuEventDestroy_v2":            &cuEventDestroy,
			"cuEventRecord":                &cuEventRecord,
			"cuEventSynchronize":           &cuEventSynchronize,
			"cuEventElapsedTime":           &cuEventElapsedTime,
			"cuModuleLoadData":             &cuModuleLoadData,
			"cuModuleUnload":               &cuModuleUnload,
			"cuModuleGetFunction":          &cuModuleGetFunction,
			"cuLaunchKernel":               &cuLaunchKernel,
			"cuGetErrorString":             &cuGetErrorString,
		})
	})
	return driverErr
}

// loadNVRTC opens libnvrtc and binds the compiler API, at most once per
// process. It is separate from loadDriver because the two libraries come from
// different packages: a machine can have a driver and no toolkit, which is
// exactly the case where prebuilt PTX is the whole point.
func loadNVRTC() error {
	nvrtcOnce.Do(func() {
		h, err := openFirst("libnvrtc", envLibNVRTC, nvrtcCandidates())
		if err != nil {
			nvrtcLoadErr = err
			return
		}
		bind(h, map[string]any{
			"nvrtcCreateProgram":     &nvrtcCreateProgram,
			"nvrtcDestroyProgram":    &nvrtcDestroyProgram,
			"nvrtcCompileProgram":    &nvrtcCompileProgram,
			"nvrtcGetProgramLogSize": &nvrtcGetProgramLogSize,
			"nvrtcGetProgramLog":     &nvrtcGetProgramLog,
			"nvrtcGetPTXSize":        &nvrtcGetPTXSize,
			"nvrtcGetPTX":            &nvrtcGetPTX,
			"nvrtcVersion":           &nvrtcVersion,
			"nvrtcGetErrorString":    &nvrtcGetErrorString,
		})
	})
	return nvrtcLoadErr
}

// openFirst dlopens the first candidate that opens, and reports every one it
// tried if none does.
func openFirst(lib, envVar string, candidates []string) (uintptr, error) {
	for _, path := range candidates {
		if h, err := purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_GLOBAL); err == nil {
			return h, nil
		}
	}
	return 0, &LibraryError{Lib: lib, Tried: candidates, EnvVar: envVar}
}

// bind resolves every symbol into the function variable that calls it.
//
// A missing symbol panics, via purego. That is deliberate and not a judgement
// call left to run time: every name here is part of the CUDA driver or NVRTC
// ABI and has been since CUDA 3.2, so a library that opens but does not export
// one of them is not an older CUDA, it is not libcuda at all. Reporting that
// as a recoverable error would only defer the crash to a nil call later.
func bind(lib uintptr, syms map[string]any) {
	for name, fptr := range syms {
		purego.RegisterLibFunc(fptr, lib, name)
	}
}

// goString copies a NUL-terminated C string into Go memory. purego converts a
// char* return value automatically, but not one handed back through a pointer
// argument, which is how cuGetErrorString reports.
func goString(p *byte) string {
	if p == nil {
		return ""
	}
	var n int
	for q := unsafe.Add(unsafe.Pointer(p), 0); *(*byte)(q) != 0; q = unsafe.Add(q, 1) {
		n++
	}
	if n == 0 {
		return ""
	}
	return string(unsafe.Slice(p, n))
}
