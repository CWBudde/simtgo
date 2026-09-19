//go:build cuda

package cuda

import "unsafe"

// Page-locked host memory.
//
// Ordinary Go memory is pageable: the kernel may move it or swap it out, so
// the driver cannot hand its physical address to the copy engine. Every
// cuMemcpyHtoD out of a Go slice therefore stages through a page-locked
// bounce buffer the driver keeps for the purpose, which costs a second copy
// and caps the transfer well below what the link can do.
//
// Page-locked memory is the buffer, owned by the caller instead of borrowed
// per copy. Two things follow, and the second is the one that matters here:
//
//  1. A synchronous copy out of it is faster, because the staging copy is
//     gone. Less than the textbook factor of two, as it turns out -- about 5%
//     on the machine in docs/toolchain.md, where the driver evidently
//     pipelines the staging copy well. Not, on its own, a reason to use this.
//  2. An *asynchronous* copy is possible at all. cuMemcpyHtoDAsync out of
//     pageable memory is documented as falling back to a synchronous
//     transfer, so a program built on Go slices can ask for overlap and
//     silently never get it.
//
// There is a third reason, specific to Go and not to CUDA. An async copy
// outlives the call that issued it, so the host buffer has to stay put until
// the stream drains -- and runtime.Pinner, which is how this package shows Go
// memory to the driver everywhere else, pins only for the duration of a call.
// Memory from cuMemHostAlloc is not Go memory and is not the collector's
// business at all, which is why UploadAsync and DownloadAsync take a
// *HostSlice and not a []T. See
// docs/decisions.md#an-asynchronous-copy-does-not-take-a-go-slice.

// hostAllocDefault is CU_MEMHOSTALLOC_DEFAULT: page-locked and nothing else.
//
// The three flags not used are PORTABLE (visible to every context in the
// process, which this package has no cross-context story for), DEVICEMAP
// (maps the allocation into the device address space, a different feature
// with a different failure mode) and WRITECOMBINED (faster over the bus,
// dramatically slower for the CPU to read back, so it is right only for
// write-once staging buffers and wrong as a default).
const hostAllocDefault = 0

// HostSlice is a typed host buffer the driver has page-locked.
//
// It is the host-side counterpart to Slice: same element type, same
// Len/Free shape, and the memory is reached through Slice(), which hands back
// an ordinary []T. Everything a Go slice can do to those elements works.
//
// The memory does not come from the Go heap, so nothing about it is collected
// and it must be released with Free -- a dropped HostSlice is a leak of
// page-locked memory, which is scarcer than ordinary memory because the
// kernel cannot reclaim it under pressure.
type HostSlice[T any] struct {
	p   unsafe.Pointer
	n   int
	ctx *Context
}

// NewHostSlice allocates n elements of page-locked host memory.
//
// The allocation belongs to the context: page-locking is a property the
// driver records against the current context, so this goes through call like
// every other driver entry point here.
func NewHostSlice[T any](c *Context, n int) (*HostSlice[T], error) {
	if n < 0 {
		return nil, &LengthError{Op: "NewHostSlice", Want: 0, Got: n}
	}
	h := &HostSlice[T]{n: n, ctx: c}
	if n == 0 {
		// cuMemHostAlloc rejects a zero-byte request, and a caller sizing a
		// buffer from data that turned out to be empty should not have to
		// special-case it. An empty HostSlice holds no pointer, and Slice()
		// gives back an empty slice.
		return h, nil
	}
	if err := c.call("cuMemHostAlloc", func() Result {
		return cuMemHostAlloc(&h.p, uint64(n*sizeOf[T]()), hostAllocDefault)
	}); err != nil {
		return nil, err
	}
	return h, nil
}

// Slice returns the buffer as an ordinary Go slice.
//
// The elements live in driver memory, not on the Go heap. That has one
// consequence a caller has to know: the slice is only valid until Free, and
// nothing in the language will stop them using it afterwards -- there is no
// collector holding the allocation alive because a reference survives. Keep
// the *HostSlice for as long as the []T is in use.
//
// Appending to it reallocates onto the Go heap, as it would for any slice
// over its capacity, and the result is ordinary pageable memory that the
// async copies will refuse. That is a quiet way to lose the page-locking, so
// treat the slice as fixed-length.
func (h *HostSlice[T]) Slice() []T {
	if h.p == nil {
		return nil
	}
	return unsafe.Slice((*T)(h.p), h.n)
}

// Len reports the number of elements.
func (h *HostSlice[T]) Len() int { return h.n }

// Free releases the page-locked memory. It is idempotent, so a deferred Free
// beside an explicit one is not an error.
func (h *HostSlice[T]) Free() {
	if h.p == nil {
		return
	}
	p := h.p
	h.p = nil
	_ = h.ctx.call("cuMemFreeHost", func() Result { return cuMemFreeHost(p) })
}
