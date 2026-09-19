package cuda

import (
	"fmt"
	"strconv"
)

// Result is a CUDA driver status code (CUresult). It is itself an error, so a
// sentinel can be compared with errors.Is: a failed driver call reports a
// *Error that unwraps to its Result, and errors.Is(err, ErrIllegalAddress)
// then answers the question a caller actually has.
//
// This file carries no build tag on purpose. The sentinels have to exist, and
// compare equal, in a build without the "cuda" tag as well, exactly like
// ErrNoCUDA: code that reacts to a particular driver failure is written once
// and has to compile on a machine with no CUDA toolchain.
type Result int32

// The codes this binding can actually run into. CUresult has well over a
// hundred values; the rest stay unnamed and print as their number, which is
// all a caller can do with them anyway.
const (
	Success                  Result = 0
	ErrInvalidValue          Result = 1
	ErrOutOfMemory           Result = 2
	ErrNotInitialized        Result = 3
	ErrDeinitialized         Result = 4
	ErrNoDevice              Result = 100
	ErrInvalidDevice         Result = 101
	ErrInvalidImage          Result = 200
	ErrInvalidContext        Result = 201
	ErrNoBinaryForGPU        Result = 209
	ErrInvalidPTX            Result = 218
	ErrUnsupportedPTXVersion Result = 222
	ErrNotFound              Result = 500
	ErrIllegalAddress        Result = 700
	ErrLaunchOutOfResources  Result = 701
	ErrLaunchTimeout         Result = 702
	ErrIllegalInstruction    Result = 715
	ErrMisalignedAddress     Result = 716
	ErrLaunchFailed          Result = 719
	ErrSystemDriverMismatch  Result = 803
)

// resultNames is written out in Go rather than fetched from cuGetErrorName
// because a Result outlives the call that produced it: it is formatted by a
// test, by a caller in a package that has no cgo, or in a build without the
// "cuda" tag, where there is no driver to ask.
var resultNames = map[Result]string{
	Success:                  "CUDA_SUCCESS",
	ErrInvalidValue:          "CUDA_ERROR_INVALID_VALUE",
	ErrOutOfMemory:           "CUDA_ERROR_OUT_OF_MEMORY",
	ErrNotInitialized:        "CUDA_ERROR_NOT_INITIALIZED",
	ErrDeinitialized:         "CUDA_ERROR_DEINITIALIZED",
	ErrNoDevice:              "CUDA_ERROR_NO_DEVICE",
	ErrInvalidDevice:         "CUDA_ERROR_INVALID_DEVICE",
	ErrInvalidImage:          "CUDA_ERROR_INVALID_IMAGE",
	ErrInvalidContext:        "CUDA_ERROR_INVALID_CONTEXT",
	ErrNoBinaryForGPU:        "CUDA_ERROR_NO_BINARY_FOR_GPU",
	ErrInvalidPTX:            "CUDA_ERROR_INVALID_PTX",
	ErrUnsupportedPTXVersion: "CUDA_ERROR_UNSUPPORTED_PTX_VERSION",
	ErrNotFound:              "CUDA_ERROR_NOT_FOUND",
	ErrIllegalAddress:        "CUDA_ERROR_ILLEGAL_ADDRESS",
	ErrLaunchOutOfResources:  "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES",
	ErrLaunchTimeout:         "CUDA_ERROR_LAUNCH_TIMEOUT",
	ErrIllegalInstruction:    "CUDA_ERROR_ILLEGAL_INSTRUCTION",
	ErrMisalignedAddress:     "CUDA_ERROR_MISALIGNED_ADDRESS",
	ErrLaunchFailed:          "CUDA_ERROR_LAUNCH_FAILED",
	ErrSystemDriverMismatch:  "CUDA_ERROR_SYSTEM_DRIVER_MISMATCH",
}

// Name returns the driver's own spelling of the code, e.g.
// "CUDA_ERROR_INVALID_CONTEXT", so that a message can be matched against the
// CUDA documentation. An unnamed code keeps its number.
func (r Result) Name() string {
	if name, ok := resultNames[r]; ok {
		return name
	}
	return "CUresult(" + strconv.Itoa(int(r)) + ")"
}

// Error makes a bare code usable as a sentinel. A code on its own says nothing
// about which call failed, so the message a user sees normally comes from the
// *Error wrapping it.
func (r Result) Error() string { return "cuda: " + r.Name() }

// Error is a failed CUDA driver call.
type Error struct {
	Op   string // the driver entry point, e.g. "cuMemAlloc"
	Code Result
	Desc string // cuGetErrorString's text, captured at the call site
}

// Error keeps the wording the package has always used, so log scrapers and
// test expectations that predate the typed errors still match.
func (e *Error) Error() string {
	if e.Desc == "" {
		return fmt.Sprintf("cuda: %s failed: %s", e.Op, e.Code.Name())
	}
	return fmt.Sprintf("cuda: %s failed: %s (%s)", e.Op, e.Desc, e.Code.Name())
}

// Unwrap exposes the status code, which is what errors.Is compares against.
func (e *Error) Unwrap() error { return e.Code }

// NVRTCResult is an NVRTC status code (nvrtcResult). Like Result it is its own
// sentinel, and its names are spelled out here for the same reason.
type NVRTCResult int32

// The nvrtcResult enumeration, transcribed. NVRTCSuccess is the zero value
// because the C enumeration makes it one, not by Go convention.
const (
	NVRTCSuccess                              NVRTCResult = 0
	ErrNVRTCOutOfMemory                       NVRTCResult = 1
	ErrNVRTCProgramCreationFailure            NVRTCResult = 2
	ErrNVRTCInvalidInput                      NVRTCResult = 3
	ErrNVRTCInvalidProgram                    NVRTCResult = 4
	ErrNVRTCInvalidOption                     NVRTCResult = 5
	ErrNVRTCCompilation                       NVRTCResult = 6
	ErrNVRTCBuiltinOperationFailure           NVRTCResult = 7
	ErrNVRTCNoNameExpressionsAfterCompilation NVRTCResult = 8
	ErrNVRTCNoLoweredNamesBeforeCompilation   NVRTCResult = 9
	ErrNVRTCNameExpressionNotValid            NVRTCResult = 10
	ErrNVRTCInternalError                     NVRTCResult = 11
	ErrNVRTCTimeFileWriteFailed               NVRTCResult = 12
	ErrNVRTCNoPCHCreateAttempted              NVRTCResult = 13
	ErrNVRTCPCHCreateHeapExhausted            NVRTCResult = 14
	ErrNVRTCPCHCreate                         NVRTCResult = 15
	ErrNVRTCCancelled                         NVRTCResult = 16
)

var nvrtcNames = map[NVRTCResult]string{
	NVRTCSuccess:                              "NVRTC_SUCCESS",
	ErrNVRTCOutOfMemory:                       "NVRTC_ERROR_OUT_OF_MEMORY",
	ErrNVRTCProgramCreationFailure:            "NVRTC_ERROR_PROGRAM_CREATION_FAILURE",
	ErrNVRTCInvalidInput:                      "NVRTC_ERROR_INVALID_INPUT",
	ErrNVRTCInvalidProgram:                    "NVRTC_ERROR_INVALID_PROGRAM",
	ErrNVRTCInvalidOption:                     "NVRTC_ERROR_INVALID_OPTION",
	ErrNVRTCCompilation:                       "NVRTC_ERROR_COMPILATION",
	ErrNVRTCBuiltinOperationFailure:           "NVRTC_ERROR_BUILTIN_OPERATION_FAILURE",
	ErrNVRTCNoNameExpressionsAfterCompilation: "NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION",
	ErrNVRTCNoLoweredNamesBeforeCompilation:   "NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION",
	ErrNVRTCNameExpressionNotValid:            "NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID",
	ErrNVRTCInternalError:                     "NVRTC_ERROR_INTERNAL_ERROR",
	ErrNVRTCTimeFileWriteFailed:               "NVRTC_ERROR_TIME_FILE_WRITE_FAILED",
	ErrNVRTCNoPCHCreateAttempted:              "NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED",
	ErrNVRTCPCHCreateHeapExhausted:            "NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED",
	ErrNVRTCPCHCreate:                         "NVRTC_ERROR_PCH_CREATE",
	ErrNVRTCCancelled:                         "NVRTC_ERROR_CANCELLED",
}

// Name returns NVRTC's own spelling of the code, e.g.
// "NVRTC_ERROR_COMPILATION". An unnamed code keeps its number.
func (r NVRTCResult) Name() string {
	if name, ok := nvrtcNames[r]; ok {
		return name
	}
	return "nvrtcResult(" + strconv.Itoa(int(r)) + ")"
}

// Error makes a bare code usable as a sentinel; see Result.Error.
func (r NVRTCResult) Error() string { return "nvrtc: " + r.Name() }

// NVRTCError is a failed NVRTC call that is not the compilation itself.
// A kernel NVRTC refused is reported as a *CompileError instead, because there
// the log matters more than the code.
type NVRTCError struct {
	Op   string // the NVRTC entry point, e.g. "nvrtcCreateProgram"
	Code NVRTCResult
	Desc string // nvrtcGetErrorString's text, captured at the call site
}

func (e *NVRTCError) Error() string {
	if e.Desc == "" {
		return fmt.Sprintf("nvrtc: %s failed: %s", e.Op, e.Code.Name())
	}
	return fmt.Sprintf("nvrtc: %s failed: %s", e.Op, e.Desc)
}

// Unwrap exposes the status code, which is what errors.Is compares against.
func (e *NVRTCError) Unwrap() error { return e.Code }

// PTX is the result of a successful NVRTC compilation. The log travels with
// the PTX because NVRTC reports warnings there and nowhere else, so dropping
// it would silently hide everything a generated kernel got away with.
type PTX struct {
	Bytes []byte
	Log   string // NVRTC's log; non-empty means warnings
	Arch  string
}

// A CompileOption is one setting handed to NVRTC alongside --gpu-architecture.
//
// Options are functional rather than a struct so that Compile's signature
// survives the next one: the arch is required and everything else is a choice,
// which is the shape simt.BuildOption already has. The type lives here, in the
// untagged half, so that driver_cuda.go and stub.go declare the same Compile
// and cuda/surface_test.go can compare them.
type CompileOption func(*compileOptions)

// compileOptions is the accumulated set. It is unexported because the only
// thing a caller can usefully do with it is build one, and NVRTC's option
// strings are an implementation detail of this package rather than an API.
type compileOptions struct {
	fastMath bool
}

// WithFastMath compiles with --use_fast_math.
//
// That single flag sets four: --ftz=true, --prec-div=false, --prec-sqrt=false
// and --fmad=true. NUMERICS.md records what each one does to the emitted PTX
// and what a test may still assert about a kernel compiled this way. It is
// never inferred -- simt reaches it only from a kernel's //simtgo:fastmath
// directive, which is also recorded in the generated source so that the two
// compilations hash differently.
func WithFastMath() CompileOption { return func(o *compileOptions) { o.fastMath = true } }

// nvrtcOptions renders the options as the argument strings NVRTC expects,
// arch first. It is shared by both build tags so that a build without the tag
// still type-checks the same option set, and so the order is decided in one
// place -- the committed PTX is reproducible only if the command line is.
func nvrtcOptions(arch string, opts []CompileOption) []string {
	var o compileOptions
	for _, opt := range opts {
		opt(&o)
	}
	args := []string{"--gpu-architecture=" + arch}
	if o.fastMath {
		args = append(args, "--use_fast_math")
	}
	return args
}

// CompileError is a kernel NVRTC refused.
type CompileError struct {
	Name, Arch string
	Code       NVRTCResult
	Log        string // the whole compiler log -- the only thing that explains the failure
}

// Error keeps the code-then-log shape the package has always printed: the log
// is the message, and anything reading these failures out of a build log looks
// for it right after the NVRTC code.
func (e *CompileError) Error() string {
	return fmt.Sprintf("nvrtc: nvrtcCompileProgram failed: %s\n%s", e.Code.Name(), e.Log)
}

// Unwrap exposes the status code, which is what errors.Is compares against.
func (e *CompileError) Unwrap() error { return e.Code }

// ContextPoisonedError reports a context that a device-side fault has made
// permanently unusable, and is returned by every later call on it in place of
// the driver error that would otherwise arrive with no explanation.
//
// It unwraps to the original Result, so errors.Is(err, cuda.ErrLaunchFailed)
// keeps working on the tenth call as well as the first.
type ContextPoisonedError struct {
	// Op is the driver call that first failed, which is where the fault
	// surfaced rather than where it happened: a kernel fault is asynchronous
	// and is usually reported by cuCtxSynchronize.
	Op string
	// Code is what that call returned.
	Code Result
}

func (e *ContextPoisonedError) Error() string {
	return fmt.Sprintf("cuda: this context is unusable: %s failed with %s, "+
		"and a fault of that kind is sticky -- the driver fails every later call in it. "+
		"Measured here, retaining a fresh primary context fails too, so this process "+
		"cannot use CUDA again and has to exit", e.Op, e.Code.Name())
}

func (e *ContextPoisonedError) Unwrap() error { return e.Code }

// LengthError reports a copy whose two ends are different sizes, or a
// negative allocation.
//
// It lives here rather than being a plain fmt.Errorf because the copies it
// guards are the ones a caller reaches for when a kernel produced the wrong
// number of elements, and "want 1024, got 1023" is the sentence that finds
// that bug. The driver would not have caught it: a short copy into a longer
// buffer is a perfectly legal cuMemcpy.
type LengthError struct {
	// Op is the operation that refused.
	Op string
	// Want is the length the operation required, Got the one it was given.
	Want, Got int
}

func (e *LengthError) Error() string {
	return fmt.Sprintf("cuda: %s: length mismatch: want %d elements, got %d", e.Op, e.Want, e.Got)
}

// BusyStreamError reports an operation refused because a cancelled Wait left
// work outstanding on the stream.
type BusyStreamError struct {
	// Op is the operation that refused.
	Op string
}

func (e *BusyStreamError) Error() string {
	return fmt.Sprintf("cuda: %s: a cancelled Wait left work on this stream and the device is still running it; "+
		"Sync to let it finish, or CloseAbandoned to say the outstanding work is accepted", e.Op)
}

// NilError reports a required argument that was nil.
//
// The alternative is the panic Go would give anyway, one frame further in and
// naming a field rather than a parameter. These are the package's own handle
// types -- a *Stream nobody created, a *HostSlice whose allocation failed and
// whose error was dropped -- and an error saying which argument it was gets
// the caller back to their own line.
type NilError struct {
	// Op is the operation that refused, and Arg the parameter that was nil.
	Op, Arg string
}

func (e *NilError) Error() string {
	return fmt.Sprintf("cuda: %s: %s must not be nil", e.Op, e.Arg)
}
