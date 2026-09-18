package cuda

import (
	"slices"
	"testing"
)

// TestNVRTCOptions pins the command line NVRTC is given.
//
// It lives in the untagged half, beside library_test.go and for the same
// reason: the option list is a policy decision, and a policy that can only be
// checked on a machine with CUDA installed is a policy nobody checks. It is
// also the order that matters rather than the set -- the committed PTX is
// reproducible only if the command line is -- so the assertions are on slices
// and not on membership.
func TestNVRTCOptions(t *testing.T) {
	for _, tc := range []struct {
		name string
		opts []CompileOption
		want []string
	}{{
		name: "arch alone is the default",
		want: []string{"--gpu-architecture=compute_75"},
	}, {
		name: "fast math is added after the arch",
		opts: []CompileOption{WithFastMath()},
		want: []string{"--gpu-architecture=compute_75", "--use_fast_math"},
	}, {
		// Asking twice is asking once. A caller assembling options from a
		// slice should not be able to change the command line by doing so.
		name: "fast math twice is still one option",
		opts: []CompileOption{WithFastMath(), WithFastMath()},
		want: []string{"--gpu-architecture=compute_75", "--use_fast_math"},
	}} {
		t.Run(tc.name, func(t *testing.T) {
			got := nvrtcOptions("compute_75", tc.opts)
			if !slices.Equal(got, tc.want) {
				t.Errorf("nvrtcOptions = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestNVRTCOptionsFastMathIsNeverImplicit is the assertion that matters for
// NUMERICS.md: nothing turns the flag on by itself. Every numerical property
// of a kernel that did not ask is an NVRTC default, and this is what keeps
// that sentence true.
func TestNVRTCOptionsFastMathIsNeverImplicit(t *testing.T) {
	for _, arch := range []string{"compute_50", "compute_75", "compute_90"} {
		if got := nvrtcOptions(arch, nil); len(got) != 1 {
			t.Errorf("nvrtcOptions(%q, nil) = %q, want the arch alone", arch, got)
		}
	}
}
