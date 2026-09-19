package simt

import (
	"errors"
	"math"
	"testing"

	"github.com/CWBudde/simtgo/cuda"
)

// The launch contract that can be checked without a device: how many arguments
// a kernel takes, and how big a dynamically sized tile may be asked for.
//
// Both refusals happen before anything is marshalled or launched, which is why
// a Kernel with no cuda.Function behind it is enough to exercise them -- and
// why they are worth having at all: past this point the driver reads the
// parameter array against the signature the PTX declares and says nothing.

// dynProbeParams is the signature of a kernel with a dynamically sized tile:
// two slices, and no parameter for the tile's length. The length is generated,
// so it is LaunchSharedDim that appends it and Params that never mentions it.
func dynProbeParams() []Param {
	return []Param{
		{Name: "bins", Slice: true},
		{Name: "x", Slice: true, ReadOnly: true},
	}
}

func TestCheckArgs(t *testing.T) {
	plain := &Kernel{Name: "K", Params: []Param{
		{Name: "c", Slice: true},
		{Name: "a", Slice: true, ReadOnly: true},
		{Name: "k"},
	}}
	// The same signature, plus a dynamically sized tile. A launch of this one
	// reaches launch with one argument more than Params describes.
	dyn := &Kernel{Name: "D", DynSharedWidth: 4, Params: dynProbeParams()}

	cases := []struct {
		name      string
		kernel    *Kernel
		args      int // arguments reaching launch
		want, got int // what the error must report, -1 for "no error"
	}{
		{name: "an exact call", kernel: plain, args: 3, want: -1},
		{name: "one argument too few", kernel: plain, args: 2, want: 3, got: 2},
		{name: "one argument too many", kernel: plain, args: 4, want: 3, got: 4},
		{name: "no arguments at all", kernel: plain, args: 0, want: 3, got: 0},
		// Three, because LaunchSharedDim appended the generated length to the
		// caller's two. The count the error would report is the caller's.
		{name: "an exact call with a dynamic tile", kernel: dyn, args: 3, want: -1},
		{name: "one too few with a dynamic tile", kernel: dyn, args: 2, want: 2, got: 1},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.kernel.checkArgs(make([]any, tc.args))
			if tc.want < 0 {
				if err != nil {
					t.Fatalf("launch refused: %v", err)
				}
				return
			}
			var bad *ArgCountError
			if !errors.As(err, &bad) {
				t.Fatalf("got %v, want an ArgCountError", err)
			}
			if bad.Want != tc.want || bad.Got != tc.got {
				t.Errorf("ArgCountError reports want %d got %d, want want %d got %d",
					bad.Want, bad.Got, tc.want, tc.got)
			}
		})
	}
}

// TestCheckAliasingWithDynamicTile pins the reason checkAliasing walks the
// parameters rather than the arguments.
//
// A launch of a kernel with a dynamically sized tile carries the generated
// length as a trailing argument that Params has no entry for. Comparing the
// two lengths for equality -- which is what the check used to do before
// skipping -- made every such launch a mismatch, and so turned the aliasing
// check off for exactly the kernels that stage a shared tile.
func TestCheckAliasingWithDynamicTile(t *testing.T) {
	k := &Kernel{Name: "D", DynSharedWidth: 4, Params: dynProbeParams()}
	const size = 4096
	buf := fakeBuffer{base: 0x1000, bytes: size}

	// bins and x bound to the same memory, with the kernel writing through
	// bins, and the tile's length last, as LaunchSharedDim appends it.
	err := k.checkAliasing([]any{buf, buf, int32(192)})
	var bad *AliasError
	if !errors.As(err, &bad) {
		t.Fatalf("got %v, want an AliasError", err)
	}
	if bad.Write != "bins" || bad.Other != "x" {
		t.Errorf("AliasError names %s and %s, want bins and x", bad.Write, bad.Other)
	}

	// And the same launch with distinct buffers is still accepted, so the
	// trailing length is not itself being mistaken for a range.
	other := fakeBuffer{base: 0x1000 + size, bytes: size}
	if err := k.checkAliasing([]any{buf, other, int32(192)}); err != nil {
		t.Fatalf("launch refused: %v", err)
	}
}

// TestLaunchSharedElementCount covers the sizes LaunchSharedDim refuses before
// it multiplies.
//
// The product is computed in a host int, so on a 64-bit host it does not wrap
// anywhere near the point where n stops fitting the int32 length the kernel
// reads. A tile asked for in those counts would be allocated at one size and
// read at another, and maxShared == 0 -- no device was asked -- leaves the
// shared-memory limit with nothing to say about it either.
func TestLaunchSharedElementCount(t *testing.T) {
	k := &Kernel{Name: "D", DynSharedWidth: 4, Params: dynProbeParams()}
	var count *ArgCountError
	for _, n := range []int{-1, math.MaxInt32, math.MaxInt32/4 + 1} {
		// A well formed call otherwise, so that the size is the only thing
		// left to refuse it.
		err := k.LaunchSharedDim(cuda.D1(1), cuda.D1(32), n, nil, nil)
		if err == nil {
			t.Errorf("a shared tile of %d elements was accepted", n)
		}
		if errors.As(err, &count) {
			t.Errorf("a shared tile of %d elements was refused for its arguments: %v", n, err)
		}
	}
	// The largest count that does fit has to survive, so that it is the
	// arithmetic being refused above and not simply everything large. Asked
	// for with one argument too few, which is the next check along: reaching
	// it at all is what says the size was accepted, and it answers without a
	// device, which a real launch would not.
	n := math.MaxInt32 / 4
	if err := k.LaunchSharedDim(cuda.D1(1), cuda.D1(32), n, nil); !errors.As(err, &count) {
		t.Errorf("a shared tile of %d elements gave %v, want the launch to reach its argument check", n, err)
	}
}

// TestTrapErrorOnlyWrapsADeviceFault pins which failures a bounds-checked
// launch is allowed to call a trap.
//
// The wrapper used to catch every error the driver returned, and the two
// cases below are why that was wrong. A configuration failure means the
// kernel never ran, so calling it a fault sends the reader hunting an index
// that was never read. A ContextPoisonedError means an *earlier* kernel
// faulted and this one never got past bind, so naming this kernel accuses one
// that did not execute -- and replaces the one sentence in the whole library
// that already says what happened.
//
// No device: diagnose is pure, and handing it the errors the driver would
// have produced is what makes the untagged build able to check this at all.
func TestTrapErrorOnlyWrapsADeviceFault(t *testing.T) {
	k := &Kernel{Name: "Overrun", BoundsChecks: true}

	for _, tc := range []struct {
		name string
		err  error
		trap bool
	}{
		{"an unspecified launch failure, which is what __trap surfaces as",
			&cuda.Error{Op: "cuCtxSynchronize", Code: cuda.ErrLaunchFailed}, true},
		{"an illegal address, which an overrun the checks missed lands on",
			&cuda.Error{Op: "cuCtxSynchronize", Code: cuda.ErrIllegalAddress}, true},
		{"out of resources, where the launch was refused and nothing ran",
			&cuda.Error{Op: "cuLaunchKernel", Code: cuda.ErrLaunchOutOfResources}, false},
		{"an invalid value, likewise",
			&cuda.Error{Op: "cuLaunchKernel", Code: cuda.ErrInvalidValue}, false},
		{"a context already poisoned by some earlier kernel",
			&cuda.ContextPoisonedError{Op: "cuCtxSynchronize", Code: cuda.ErrLaunchFailed}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := k.diagnose(tc.err)
			var trap *TrapError
			if errors.As(got, &trap) != tc.trap {
				t.Fatalf("diagnose(%v) = %v; wrapped as a TrapError = %v, want %v", tc.err, got, !tc.trap, tc.trap)
			}
			if !errors.Is(got, tc.err) {
				t.Errorf("diagnose(%v) = %v, which no longer unwraps to what the driver said", tc.err, got)
			}
		})
	}

	// And nothing is wrapped at all in a release build, whatever the code.
	plain := &Kernel{Name: "Overrun"}
	var trap *TrapError
	if got := plain.diagnose(&cuda.Error{Op: "cuCtxSynchronize", Code: cuda.ErrLaunchFailed}); errors.As(got, &trap) {
		t.Errorf("a kernel built without bounds checks reported %v, and it has no checks to blame", got)
	}
}
