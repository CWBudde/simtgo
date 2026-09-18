// Package cuda is a small, hand-rolled binding for the CUDA driver API and
// NVRTC, covering exactly the surface this module needs: allocate device
// memory, JIT-compile CUDA C at runtime, and launch a kernel.
//
// There is no cgo. libcuda and libnvrtc are dlopen'd at run time through
// purego, so this package builds with CGO_ENABLED=0 and with no toolkit
// installed. The "cuda" build tag still selects the implementation: the driver
// lives behind it, and without the tag every entry point here fails with
// ErrNoCUDA, so the rest of the module -- notably the Go-to-CUDA transpiler in
// package simt -- builds and tests on machines with no CUDA at all.
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
	"unsafe"
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
	for i := range len(s) {
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

// D1 returns a one-dimensional Dim3. It, D2 and D3 leave the axes above their
// own rank at one -- which is what CUDA treats an unused axis as, not zero.
func D1(x int) Dim3 { return Dim3{X: uint32(x), Y: 1, Z: 1} }

// D2 returns a two-dimensional Dim3.
func D2(x, y int) Dim3 { return Dim3{X: uint32(x), Y: uint32(y), Z: 1} }

// D3 returns a three-dimensional Dim3.
func D3(x, y, z int) Dim3 { return Dim3{X: uint32(x), Y: uint32(y), Z: uint32(z)} }

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

// ArgU32 passes a 32-bit unsigned integer.
func ArgU32(v uint32) Arg {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], v)
	return Arg{b: b[:]}
}

// ArgI64 passes a 64-bit integer, which the kernel receives as a long long.
func ArgI64(v int64) Arg {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], uint64(v))
	return Arg{b: b[:]}
}

// ArgU64 passes a 64-bit unsigned integer.
func ArgU64(v uint64) Arg {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], v)
	return Arg{b: b[:]}
}

// ArgF64 passes a double, which only a kernel carrying //gocuda:float64 can
// declare a parameter for.
func ArgF64(v float64) Arg {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], math.Float64bits(v))
	return Arg{b: b[:]}
}

// ArgBool passes a bool. CUDA's bool is one byte, which the generated C
// asserts, so this is one byte too.
func ArgBool(v bool) Arg {
	var b [1]byte
	if v {
		b[0] = 1
	}
	return Arg{b: b[:]}
}

// ArgOf passes any value whose Go layout is what the kernel's generated C
// declares -- in practice a struct of scalars.
//
// It is explicit rather than something BuildArgs infers, because deciding
// whether a struct is one the device can take means checking every field
// against the supported set, and that rule lives in internal/lower. A second
// copy of it here would be a second opinion about the subset, and the two would
// eventually differ. The caller asserting the layout is safe because the
// generated C carries a static_assert on the type's size and alignment, so a
// disagreement is a compile error rather than wrong numbers.
func ArgOf[T any](v T) Arg {
	b := make([]byte, unsafe.Sizeof(v))
	copy(b, unsafe.Slice((*byte)(unsafe.Pointer(&v)), len(b)))
	return Arg{b: b}
}

// Argser is implemented by values that expand to one or more kernel
// parameters. A device slice expands to two: a pointer and a length.
type Argser interface {
	KernelArgs() []Arg
}

// Ranger is implemented by a kernel argument that occupies a stretch of device
// memory: its base address and its size in bytes.
//
// It exists so that a caller can ask whether two arguments are the same
// memory. The generated kernels declare every pointer __restrict__, which
// promises they are not, and this is what lets simt check that promise against
// the buffers a launch actually binds instead of leaving it as undefined
// behaviour. An argument that cannot answer -- a raw Arg, say -- is simply not
// checked.
type Ranger interface {
	DeviceRange() (base DevPtr, bytes int)
}

// BuildArgs converts a call written in terms of Go values into the flat
// parameter list the generated kernel expects, so a launch can be written to
// mirror the Go kernel's own signature.
//
// A struct parameter is passed as cuda.ArgOf(v) rather than as itself: see
// ArgOf for why that is deliberate.
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
			// The one narrowing the subset documents: a kernel's int parameter
			// is C's 32-bit int, and an index is bounded by the grid. It is
			// only safe here, for a value passed on its own -- []int is refused
			// by the transpiler, because there the narrowing is a stride.
			out = append(out, ArgI32(int32(v)))
		case uint32:
			out = append(out, ArgU32(v))
		case int64:
			out = append(out, ArgI64(v))
		case uint64:
			out = append(out, ArgU64(v))
		case float64:
			out = append(out, ArgF64(v))
		case bool:
			out = append(out, ArgBool(v))
		case Arg:
			out = append(out, v)
		default:
			return nil, fmt.Errorf("cuda: argument %d has unsupported type %T", i, v)
		}
	}
	return out, nil
}
