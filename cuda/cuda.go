// Package cuda is a small, hand-rolled binding for the CUDA driver API and
// NVRTC, covering exactly the surface this proof of concept needs: allocate
// device memory, JIT-compile CUDA C at runtime, and launch a kernel.
//
// All cgo code sits behind the "cuda" build tag so that the rest of the module
// (notably the Go-to-CUDA transpiler in package simt) builds and tests on
// machines without a CUDA toolchain. Without the tag, every entry point here
// fails with ErrNoCUDA.
package cuda

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

// ErrNoCUDA is returned by every operation when the module was built without
// the "cuda" build tag.
var ErrNoCUDA = errors.New("cuda: built without the 'cuda' build tag (rebuild with -tags cuda)")

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
