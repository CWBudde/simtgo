package tile

import (
	"errors"
	"strings"
	"testing"

	"github.com/CWBudde/gocuda/cuda"
)

// Both halves of the __restrict__ promise the tile track makes, tested without
// a device.
//
// It is an internal test because it calls checkAliasing directly, and untagged
// because neither half needs hardware: what the generated C declares is a
// string, and the check is arithmetic on device addresses. The same reasoning
// simt/alias_test.go gives for itself.

// fakeBuffer is a device range that was never allocated. cuda.Ranger is the
// whole of what the check asks an argument for.
type fakeBuffer struct {
	base  cuda.DevPtr
	bytes int
}

func (b fakeBuffer) DeviceRange() (cuda.DevPtr, int) { return b.base, b.bytes }

func TestGeneratedKernelQualifiesItsPointers(t *testing.T) {
	g := New(nil)
	out := Add(g.Input(make([]float32, 8)), Scale(g.Input(make([]float32, 8)), 2))
	src, err := out.Source()
	if err != nil {
		t.Fatalf("Source: %v", err)
	}

	want := "extern \"C\" __global__ void fused(float* __restrict__ out, int out_len, " +
		"const float* __restrict__ p0, int p0_len, const float* __restrict__ p1, int p1_len)"
	if !strings.Contains(src, want) {
		t.Errorf("signature is not qualified\ngot:\n%s\nwant a line:\n%s", src, want)
	}
	// An input is read and never written, so it is const as well as
	// restrict; out is the only thing the kernel stores through.
	if strings.Contains(src, "const float* __restrict__ out") {
		t.Error("out is declared const, but the kernel writes through it")
	}
}

func TestCheckAliasing(t *testing.T) {
	const size = 4096
	out := fakeBuffer{base: 0x1000, bytes: size}
	other := fakeBuffer{base: 0x1000 + size, bytes: size}

	cases := []struct {
		name  string
		args  []any
		input string // empty when the launch must be accepted
	}{{
		name: "distinct buffers",
		args: []any{out, other, fakeBuffer{base: 0x9000, bytes: size}},
	}, {
		name:  "an input is the output",
		args:  []any{out, other, out},
		input: "p1",
	}, {
		// What restrict forbids is reaching a modified object through another
		// pointer, so two readers of one buffer are fine. MaterializeStepwise
		// produces exactly this whenever one node feeds both operands.
		name: "two inputs share a buffer",
		args: []any{out, other, other},
	}, {
		name:  "an input overlaps the output only partly",
		args:  []any{out, fakeBuffer{base: 0x1000 + size/2, bytes: size}},
		input: "p0",
	}, {
		// A buffer of no bytes occupies no address, so it overlaps nothing.
		name: "a zero-length input at the output's address",
		args: []any{out, fakeBuffer{base: 0x1000, bytes: 0}},
	}, {
		// Nothing to compare: an argument that cannot report a range is
		// skipped rather than guessed at.
		name: "an argument that is not a device range",
		args: []any{out, int32(7)},
	}, {
		name: "no arguments at all",
		args: nil,
	}}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := checkAliasing(c.args)
			if c.input == "" {
				if err != nil {
					t.Fatalf("accepted launch refused: %v", err)
				}
				return
			}
			var ae *AliasError
			if !errors.As(err, &ae) {
				t.Fatalf("got %v, want an *AliasError naming %s", err, c.input)
			}
			if ae.Input != c.input {
				t.Errorf("names input %s, want %s", ae.Input, c.input)
			}
		})
	}
}
