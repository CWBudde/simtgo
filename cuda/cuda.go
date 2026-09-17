// Package cuda is a small, hand-rolled binding for the CUDA driver API and
// NVRTC, covering exactly the surface this proof of concept needs: allocate
// device memory, JIT-compile CUDA C at runtime, and launch a kernel.
//
// All cgo code sits behind the "cuda" build tag so that the rest of the module
// (notably the Go-to-CUDA transpiler in package simt) builds and tests on
// machines without a CUDA toolchain. Without the tag, every entry point here
// fails with ErrNoCUDA.
//
// # On context.Context
//
// Nothing here takes one, deliberately. Every call in this package is
// synchronous and uncancellable: cuCtxSynchronize, cuMemcpyHtoD and
// cuLaunchKernel cannot be interrupted once issued, so a deadline passed in
// could only be ignored, and accepting one would advertise a guarantee that
// does not exist.
//
// It becomes meaningful once streams and events exist, because cuStreamQuery
// can genuinely be polled against a cancelled context. When it arrives the
// convention is a context.Context first, named ctx, and the CUDA context
// second, named dev -- which is why the parameters are already spelled that
// way.
package cuda

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// ErrNoCUDA is returned by every operation when the module was built without
// the "cuda" build tag.
var ErrNoCUDA = errors.New("cuda: built without the 'cuda' build tag (rebuild with -tags cuda)")

// ErrContextClosed is returned when a context is asked to load something after
// it has been closed. It lives here rather than beside the driver bindings so
// that callers can test for it in either build.
var ErrContextClosed = errors.New("cuda: context is closed")

// ParseArch splits a virtual architecture such as "compute_75" into its
// compute capability.
//
// The minor version is the last digit and the major version is everything
// before it, which is the only reading that survives compute capability 10:
// "compute_100" is 10.0, not 1.0 with a stray digit, and a fixed two-digit
// split silently gets every Blackwell-class device wrong.
func ParseArch(s string) (major, minor int, err error) {
	const prefix = "compute_"
	digits, ok := strings.CutPrefix(s, prefix)
	if !ok {
		return 0, 0, fmt.Errorf("cuda: %q is not a virtual architecture: expected a %q prefix", s, prefix)
	}
	// Arch-conditional targets ("compute_90a", "compute_100f") are rejected
	// rather than parsed: their PTX is tied to that exact architecture and is
	// not forward compatible, so a caller that asked for one and got a plain
	// capability back would be told its kernel runs on hardware it does not.
	if n := len(digits); n > 1 {
		if suffix := digits[n-1]; suffix == 'a' || suffix == 'f' {
			if isDigits(digits[:n-1]) {
				return 0, 0, fmt.Errorf("cuda: %q is an arch-conditional target and its PTX is not forward compatible; use %q instead", s, prefix+digits[:n-1])
			}
		}
	}
	if !isDigits(digits) {
		return 0, 0, fmt.Errorf("cuda: %q is not a virtual architecture: %q is not a compute capability", s, digits)
	}
	if len(digits) < 2 {
		return 0, 0, fmt.Errorf("cuda: %q is not a virtual architecture: a compute capability needs a major and a minor digit, e.g. %q", s, "compute_75")
	}
	major, err = strconv.Atoi(digits[:len(digits)-1])
	if err != nil {
		return 0, 0, fmt.Errorf("cuda: %q is not a virtual architecture: %w", s, err)
	}
	return major, int(digits[len(digits)-1] - '0'), nil
}

func isDigits(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return true
}

// DevPtr is a device address (CUdeviceptr).
type DevPtr uint64

// Dim3 is a CUDA grid or block dimension.
type Dim3 struct{ X, Y, Z uint32 }

// D1 returns a one-dimensional Dim3.
func D1(x int) Dim3 { return Dim3{X: uint32(x), Y: 1, Z: 1} }

// Arg is one kernel parameter, held as its raw little-endian bytes. Launch
// copies these into C memory, so an Arg never exposes Go memory to the driver.
type Arg struct{ b []byte }

// ArgDev passes a device pointer.
func ArgDev(p DevPtr) Arg {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], uint64(p))
	return Arg{b: b[:]}
}

// ArgI32 passes a 32-bit integer.
func ArgI32(v int32) Arg {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], uint32(v))
	return Arg{b: b[:]}
}

// ArgF32 passes a 32-bit float.
func ArgF32(v float32) Arg {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
	return Arg{b: b[:]}
}

// Argser is implemented by values that expand to one or more kernel
// parameters. A device slice expands to two: a pointer and a length.
type Argser interface {
	KernelArgs() []Arg
}

// BuildArgs converts a call written in terms of Go values into the flat
// parameter list the generated kernel expects, so a launch can be written to
// mirror the Go kernel's own signature.
func BuildArgs(vals ...any) ([]Arg, error) {
	var out []Arg
	for i, v := range vals {
		switch v := v.(type) {
		case Argser:
			out = append(out, v.KernelArgs()...)
		case float32:
			out = append(out, ArgF32(v))
		case int32:
			out = append(out, ArgI32(v))
		case int:
			out = append(out, ArgI32(int32(v)))
		case Arg:
			out = append(out, v)
		default:
			return nil, fmt.Errorf("cuda: argument %d has unsupported type %T", i, v)
		}
	}
	return out, nil
}
