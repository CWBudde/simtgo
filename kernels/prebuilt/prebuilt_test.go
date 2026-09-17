package prebuilt_test

import (
	"testing"

	gocuda "github.com/CWBudde/gocuda"
	"github.com/CWBudde/gocuda/kernels/prebuilt"
	"github.com/CWBudde/gocuda/simt"
)

// TestPrebuiltIsCurrent is the half of the build gate that the generated
// symbols cannot cover.
//
// Removing a symbol catches a kernel that was regenerated and turned out not
// to lower. It cannot catch a kernel that was edited and never regenerated:
// the old symbol is still there, the build is green, and the mismatch only
// shows up at run time when the stale prebuilt is not found and the kernel is
// compiled from source instead. This test closes that, needs no GPU, and runs
// under a plain "go test ./...".
func TestPrebuiltIsCurrent(t *testing.T) {
	if err := simt.VerifyPrebuilt(gocuda.Kernels(), prebuilt.Names()...); err != nil {
		t.Errorf("%v", err)
	}
}

// TestGateListsEveryKernel keeps gate.go, which is hand-written, from falling
// behind the kernels beside it: a kernel missing from Gate is not gated at
// all, so it would go back to failing in main().
func TestGateListsEveryKernel(t *testing.T) {
	gated := map[string]bool{}
	for _, n := range prebuilt.Names() {
		gated[n] = true
	}
	for _, name := range []string{"VecAdd", "Magnitude", "Scale", "FIR", "Classify", "Softclip", "Transpose"} {
		if !gated[name] {
			t.Errorf("kernel %s is not listed in gate.go, so nothing makes it fail the build", name)
		}
	}
}
