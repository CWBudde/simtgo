//go:build cuda

package cuda_test

import (
	"errors"
	"math"
	"runtime"
	"testing"

	"github.com/CWBudde/simtgo/cuda"
)

// device opens a context for one test or benchmark and closes it afterwards.
//
// Every test in this package was opening its own by hand; these are the first
// that need one from a benchmark as well, where the b.Fatalf/Close pair is
// the same three lines a fourth time.
func device(tb testing.TB) *cuda.Context {
	tb.Helper()
	requireDevice(tb)
	dev, err := cuda.NewContext(0)
	if err != nil {
		tb.Fatalf("NewContext: %v", err)
	}
	tb.Cleanup(func() { _ = dev.Close() })
	return dev
}

func TestHostSliceRoundTrip(t *testing.T) {
	dev := device(t)
	const n = 4096

	src, err := cuda.NewHostSlice[float32](dev, n)
	if err != nil {
		t.Fatalf("NewHostSlice: %v", err)
	}
	defer src.Free()
	if got := src.Len(); got != n {
		t.Errorf("Len = %d, want %d", got, n)
	}
	xs := src.Slice()
	if len(xs) != n {
		t.Fatalf("Slice has %d elements, want %d", len(xs), n)
	}
	for i := range xs {
		xs[i] = float32(i) * 0.5
	}

	d, err := cuda.NewSlice[float32](dev, n)
	if err != nil {
		t.Fatalf("NewSlice: %v", err)
	}
	defer d.Free()
	if err := d.CopyFrom(xs); err != nil {
		t.Fatalf("CopyFrom: %v", err)
	}

	// Back into a second page-locked buffer, so both directions are exercised
	// against driver memory rather than only the upload.
	dst, err := cuda.NewHostSlice[float32](dev, n)
	if err != nil {
		t.Fatalf("NewHostSlice: %v", err)
	}
	defer dst.Free()
	if err := d.CopyTo(dst.Slice()); err != nil {
		t.Fatalf("CopyTo: %v", err)
	}
	for i, got := range dst.Slice() {
		if want := float32(i) * 0.5; got != want {
			t.Fatalf("element %d = %v, want %v", i, got, want)
		}
	}
}

// TestHostSliceIsNotGoMemory is the property the whole type exists for, and
// the one a reader is most likely to doubt.
//
// A []T over driver memory has to survive a garbage collection that moves
// nothing it knows about, and keep the same address: if the slice were
// somehow Go memory, a stack-allocated copy or a moving collector would show
// up here. It also pins down that Slice() returns the same window every time
// rather than a fresh copy, which is what makes writing through it work.
func TestHostSliceIsNotGoMemory(t *testing.T) {
	dev := device(t)
	h, err := cuda.NewHostSlice[float32](dev, 256)
	if err != nil {
		t.Fatalf("NewHostSlice: %v", err)
	}
	defer h.Free()

	h.Slice()[7] = 42
	before := &h.Slice()[0]
	for range 3 {
		runtime.GC()
	}
	if after := &h.Slice()[0]; after != before {
		t.Errorf("the buffer moved across a GC: %p then %p", before, after)
	}
	if got := h.Slice()[7]; got != 42 {
		t.Errorf("the write did not survive: element 7 = %v, want 42", got)
	}
}

func TestHostSliceEmpty(t *testing.T) {
	dev := device(t)

	// cuMemHostAlloc rejects a zero-byte request, so an empty buffer holds no
	// allocation at all. It still has to behave like one.
	h, err := cuda.NewHostSlice[float32](dev, 0)
	if err != nil {
		t.Fatalf("NewHostSlice(0): %v", err)
	}
	if got := h.Len(); got != 0 {
		t.Errorf("Len = %d, want 0", got)
	}
	if got := len(h.Slice()); got != 0 {
		t.Errorf("Slice has %d elements, want 0", got)
	}
	h.Free()
	h.Free() // idempotent, like Slice.Free

	if _, err := cuda.NewHostSlice[float32](dev, -1); err == nil {
		t.Error("NewHostSlice(-1) succeeded")
	}
}

func TestHostSliceFreeIsIdempotent(t *testing.T) {
	dev := device(t)
	h, err := cuda.NewHostSlice[byte](dev, 64)
	if err != nil {
		t.Fatalf("NewHostSlice: %v", err)
	}
	h.Free()
	h.Free()
	if got := h.Slice(); got != nil {
		t.Errorf("Slice after Free = %v, want nil", got)
	}
}

// TestCopyRefusesTheWrongLength covers what the driver would have accepted.
//
// cuMemcpy takes a byte count, so a short source into a longer buffer is a
// perfectly legal copy that leaves the tail holding whatever was there
// before. That is the shape of a bug worth a typed error.
func TestCopyRefusesTheWrongLength(t *testing.T) {
	dev := device(t)
	d, err := cuda.NewSlice[float32](dev, 16)
	if err != nil {
		t.Fatalf("NewSlice: %v", err)
	}
	defer d.Free()

	for _, tc := range []struct {
		name string
		call func() error
	}{
		{"CopyFrom short", func() error { return d.CopyFrom(make([]float32, 8)) }},
		{"CopyFrom long", func() error { return d.CopyFrom(make([]float32, 32)) }},
		{"CopyTo short", func() error { return d.CopyTo(make([]float32, 8)) }},
		{"CopyTo long", func() error { return d.CopyTo(make([]float32, 32)) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var e *cuda.LengthError
			if err := tc.call(); !errors.As(err, &e) {
				t.Fatalf("got %v, want a *cuda.LengthError", err)
			}
			if e.Want != 16 {
				t.Errorf("Want = %d, want 16", e.Want)
			}
		})
	}
}

// BenchmarkCopy is what says whether page-locking is worth the API.
//
// The two cases differ in exactly one thing: where the host buffer came from.
// Both copy the same number of bytes into the same device allocation with the
// same synchronous driver call, so the ratio is the cost of the staging copy
// the driver does for pageable memory and nothing else.
//
// 32 MiB, because the transfer has to dominate the fixed per-call cost for
// the ratio to mean anything. The numbers this produced are in
// docs/toolchain.md, with the machine named.
func BenchmarkCopy(b *testing.B) {
	dev := device(b)
	const n = 8 << 20 // 32 MiB of float32

	d, err := cuda.NewSlice[float32](dev, n)
	if err != nil {
		b.Fatalf("NewSlice: %v", err)
	}
	defer d.Free()

	pinned, err := cuda.NewHostSlice[float32](dev, n)
	if err != nil {
		b.Fatalf("NewHostSlice: %v", err)
	}
	defer pinned.Free()
	pageable := make([]float32, n)

	for _, tc := range []struct {
		name string
		host []float32
	}{
		{"pageable", pageable},
		{"pinned", pinned.Slice()},
	} {
		b.Run("HtoD/"+tc.name, func(b *testing.B) {
			b.SetBytes(int64(n) * 4)
			for b.Loop() {
				if err := d.CopyFrom(tc.host); err != nil {
					b.Fatalf("CopyFrom: %v", err)
				}
			}
		})
		b.Run("DtoH/"+tc.name, func(b *testing.B) {
			b.SetBytes(int64(n) * 4)
			for b.Loop() {
				if err := d.CopyTo(tc.host); err != nil {
					b.Fatalf("CopyTo: %v", err)
				}
			}
		})
	}
}

// TestNewHostSliceRefusesPointerBearingTypes is the review finding that
// prompted the check: page-locked memory is not scanned by the collector, so
// a []T over it that holds the only reference to a Go object keeps nothing
// alive.
//
// It runs here with a device as well as in the untagged test of checkElem,
// because what matters to a caller is that NewHostSlice refuses -- and that
// it refuses before it allocates anything, which is why the error comes back
// rather than a live HostSlice beside it.
func TestNewHostSliceRefusesPointerBearingTypes(t *testing.T) {
	dev := device(t)

	var e *cuda.ElementTypeError
	h, err := cuda.NewHostSlice[*float32](dev, 16)
	if !errors.As(err, &e) {
		t.Fatalf("NewHostSlice[*float32] = %v, want an *cuda.ElementTypeError", err)
	}
	if h != nil {
		t.Error("a refused allocation came back with a HostSlice beside its error")
	}
	if _, err := cuda.NewHostSlice[string](dev, 16); !errors.As(err, &e) {
		t.Errorf("NewHostSlice[string] = %v, want an *cuda.ElementTypeError", err)
	}

	// The pointer-free case still works, which is the half that would be easy
	// to break while adding the refusal.
	ok, err := cuda.NewHostSlice[float32](dev, 16)
	if err != nil {
		t.Fatalf("NewHostSlice[float32]: %v", err)
	}
	ok.Free()
}

// TestNewHostSliceRefusesALengthThatWouldWrap covers the second review
// finding. The byte count is computed in a Go int before it reaches the
// driver as a size_t, and a product that wrapped to a small positive number
// would allocate far too little and then hand back a Slice() claiming every
// element that was asked for.
func TestNewHostSliceRefusesALengthThatWouldWrap(t *testing.T) {
	dev := device(t)
	if _, err := cuda.NewHostSlice[float32](dev, math.MaxInt/4+1); err == nil {
		t.Error("accepted a length whose byte size overflows an int")
	}
	if _, err := cuda.NewHostSlice[float32](dev, -1); err == nil {
		t.Error("accepted a negative length")
	}
}
