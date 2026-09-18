package simt

import (
	"testing"

	"github.com/CWBudde/gocuda/cuda"
)

// The launch-time half of __restrict__, tested without a device.
//
// It is an internal test because it calls checkAliasing directly: a Kernel
// built by Build needs a GPU, and the rule being checked is arithmetic on
// device addresses that has nothing to do with one. The parity test exercises
// the same rule through a real launch, and cannot run where there is no
// hardware -- which is exactly why this one exists.

// fakeBuffer is a device range that was never allocated. cuda.Ranger is the
// whole of what the check asks an argument for.
type fakeBuffer struct {
	base  cuda.DevPtr
	bytes int
}

func (b fakeBuffer) DeviceRange() (cuda.DevPtr, int) { return b.base, b.bytes }

func TestCheckAliasing(t *testing.T) {
	// A kernel shaped like VecAdd: one output and two inputs.
	k := &Kernel{Name: "K", Params: []Param{
		{Name: "c", Slice: true},
		{Name: "a", Slice: true, ReadOnly: true},
		{Name: "b", Slice: true, ReadOnly: true},
		{Name: "k"},
	}}
	const size = 4096
	first := fakeBuffer{base: 0x1000, bytes: size}
	second := fakeBuffer{base: 0x1000 + size, bytes: size}

	cases := []struct {
		name        string
		args        []any
		write, with string // empty when the launch must be accepted
	}{{
		name: "distinct buffers",
		args: []any{first, second, fakeBuffer{base: 0x9000, bytes: size}, float32(1)},
	}, {
		name:  "the output aliases an input",
		args:  []any{first, first, second, float32(1)},
		write: "c", with: "a",
	}, {
		// The order of the two does not follow the arguments: the message has
		// to name the written one first, whichever side of the pair it is.
		name:  "an input aliases the output",
		args:  []any{second, first, second, float32(1)},
		write: "c", with: "b",
	}, {
		// What __restrict__ forbids is reaching a modified object through
		// another pointer, so two readers of one buffer are fine.
		name: "two read-only parameters share a buffer",
		args: []any{first, second, second, float32(1)},
	}, {
		// Ranges rather than identities: cuda.Slice hands out whole
		// allocations today, and a subrange would still have to be caught.
		name:  "a partial overlap",
		args:  []any{fakeBuffer{base: 0x1000 + size/2, bytes: size}, first, second, float32(1)},
		write: "c", with: "a",
	}, {
		// An empty buffer occupies nothing, so it cannot overlap anything.
		name: "an empty buffer at the same address",
		args: []any{first, fakeBuffer{base: 0x1000, bytes: 0}, second, float32(1)},
	}, {
		// A raw cuda.Arg carries a pointer and no length. There is nothing to
		// compare, and refusing it would be a guess.
		name: "an argument that cannot report its range",
		args: []any{first, cuda.ArgDev(0x1000), second, float32(1)},
	}, {
		// A call with the wrong number of arguments is a different error, and
		// the driver gives it. Pairing them off here would be a guess.
		name: "too few arguments",
		args: []any{first, first},
	}}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := k.checkAliasing(tc.args)
			if tc.write == "" {
				if err != nil {
					t.Fatalf("launch refused: %v", err)
				}
				return
			}
			bad, ok := err.(*AliasError)
			if !ok {
				t.Fatalf("got %v, want an AliasError", err)
			}
			if bad.Write != tc.write || bad.Other != tc.with {
				t.Errorf("AliasError names %s and %s, want %s and %s", bad.Write, bad.Other, tc.write, tc.with)
			}
		})
	}
}
