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
	if len(xs) > 0 {
		if err := c.copyHtoD(s.ptr, unsafe.Pointer(&xs[0]), len(xs)*sizeOf[T]()); err != nil {
			s.Free()
			return nil, err
		}
	}
	return s, nil
}

// Download copies the buffer back to a fresh host slice.
func (s *Slice[T]) Download() ([]T, error) {
	out := make([]T, s.n)
	if s.n == 0 {
		return out, nil
	}
	if err := s.ctx.copyDtoH(unsafe.Pointer(&out[0]), s.ptr, s.n*sizeOf[T]()); err != nil {
		return nil, err
	}
	return out, nil
}

// Len reports the number of elements.
func (s *Slice[T]) Len() int { return s.n }

// Arg passes the buffer's device pointer to a kernel.
func (s *Slice[T]) Arg() Arg { return ArgDev(s.ptr) }

// KernelArgs expands the slice into the pointer and length pair that the
// transpiler emits for a Go slice parameter.
func (s *Slice[T]) KernelArgs() []Arg { return []Arg{ArgDev(s.ptr), ArgI32(int32(s.n))} }

// Free releases the buffer.
func (s *Slice[T]) Free() {
	if s.ptr != 0 {
		_ = s.ctx.Free(s.ptr)
		s.ptr = 0
	}
}
