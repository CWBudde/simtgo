//go:build cuda

package cuda

import "unsafe"

// Slice is a typed device buffer. Generics give the host side of CUDA
// something the C API cannot: allocation, upload and download that remember
// their element type.
type Slice[T any] struct {
	ptr DevPtr
	n   int
	ctx *Context
}

func sizeOf[T any]() int {
	var zero T
	return int(unsafe.Sizeof(zero))
}

// NewSlice allocates an uninitialised device buffer of n elements.
func NewSlice[T any](c *Context, n int) (*Slice[T], error) {
	p, err := c.Alloc(n * sizeOf[T]())
	if err != nil {
		return nil, err
	}
	return &Slice[T]{ptr: p, n: n, ctx: c}, nil
}

// Upload allocates a device buffer and copies xs into it.
func Upload[T any](c *Context, xs []T) (*Slice[T], error) {
	s, err := NewSlice[T](c, len(xs))
	if err != nil {
		return nil, err
	}
	if err := s.CopyFrom(xs); err != nil {
		s.Free()
		return nil, err
	}
	return s, nil
}

// Download copies the buffer back to a fresh host slice.
func (s *Slice[T]) Download() ([]T, error) {
	out := make([]T, s.n)
	if err := s.CopyTo(out); err != nil {
		return nil, err
	}
	return out, nil
}

// CopyFrom fills this buffer from a host slice of the same length.
//
// It is what Upload does without the allocation, and the reason to have it
// separately is reuse: a program that uploads new data into one buffer every
// iteration should not allocate and free device memory every iteration. It is
// also the synchronous counterpart of UploadAsync, and takes a plain []T
// because a synchronous copy finishes before it returns -- pinning the
// source for the duration of the call is enough, which is exactly what the
// asynchronous version cannot say.
//
// The lengths must match. A shorter source would be a legal cuMemcpy leaving
// the tail of the buffer holding whatever was there before.
func (s *Slice[T]) CopyFrom(xs []T) error {
	if len(xs) != s.n {
		return &LengthError{Op: "CopyFrom", Want: s.n, Got: len(xs)}
	}
	if s.n == 0 {
		return nil
	}
	return s.ctx.copyHtoD(s.ptr, unsafe.Pointer(&xs[0]), s.n*sizeOf[T]())
}

// CopyTo reads this buffer into a host slice of the same length. It is
// Download without the allocation; see CopyFrom.
func (s *Slice[T]) CopyTo(xs []T) error {
	if len(xs) != s.n {
		return &LengthError{Op: "CopyTo", Want: s.n, Got: len(xs)}
	}
	if s.n == 0 {
		return nil
	}
	return s.ctx.copyDtoH(unsafe.Pointer(&xs[0]), s.ptr, s.n*sizeOf[T]())
}

// Len reports the number of elements.
func (s *Slice[T]) Len() int { return s.n }

// Arg passes the buffer's device pointer to a kernel.
func (s *Slice[T]) Arg() Arg { return ArgDev(s.ptr) }

// KernelArgs expands the slice into the pointer and length pair that the
// transpiler emits for a Go slice parameter.
func (s *Slice[T]) KernelArgs() []Arg { return []Arg{ArgDev(s.ptr), ArgI32(int32(s.n))} }

// DeviceRange reports the memory the buffer occupies, which is what tells two
// kernel arguments apart -- or finds them to be the same allocation. See
// Ranger.
func (s *Slice[T]) DeviceRange() (DevPtr, int) { return s.ptr, s.n * sizeOf[T]() }

// Free releases the buffer.
func (s *Slice[T]) Free() {
	if s.ptr != 0 {
		_ = s.ctx.Free(s.ptr)
		s.ptr = 0
	}
}
