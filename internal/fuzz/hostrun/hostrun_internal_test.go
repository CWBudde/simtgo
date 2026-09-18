package hostrun

import (
	"errors"
	"testing"
	"testing/fstest"
	"time"

	"github.com/CWBudde/gocuda/gpu"
	"github.com/CWBudde/gocuda/simt"
)

// TestNonTerminatingKernelTimesOut pins the bound rather than the hang.
//
// The fuzzer produced a program whose generated C did not terminate, and
// because nothing bounded the child it took a whole fuzz worker with it: no
// diagnosis, no failing input, and a run that spent its budget waiting. A
// kernel that does not finish is a finding, so this is what reporting it looks
// like.
//
// This file is in package hostrun, rather than beside the other tests in
// package hostrun_test, only so that it can reach runTimeout -- which is the
// same reason gpu/warp_internal_test.go exists.
//
// The loop is written in the Go subset rather than as C by hand, because a
// hand-written loop would be testing the shim and not the thing that actually
// happened. It steps an even counter towards an odd bound it can never equal,
// which the emitter lowers faithfully: the loop runs for ever on both
// backends, so the test does not depend on a mistranslation to produce the
// case it is about.
//
// The counter is unsigned deliberately. The obvious spelling -- a signed
// counter walking off its own range -- terminates after about four billion
// iterations, which the first version of this test discovered by passing in a
// second and a half, and a signed one that did not terminate would be relying
// on overflow, which C leaves undefined. Unsigned wrapping is defined, so this
// loop is infinite because of what it computes and not because of what the
// compiler is entitled to assume.
func TestNonTerminatingKernelTimesOut(t *testing.T) {
	if err := Available(); err != nil {
		t.Skipf("no host C++ compiler: %v", err)
	}
	saved := runTimeout
	runTimeout = 2 * time.Second
	t.Cleanup(func() { runTimeout = saved })

	const src = "package kernels\n\nimport \"github.com/CWBudde/gocuda/gpu\"\n\n" +
		"func Spin(ctx gpu.Ctx, y []uint32, n uint32) {\n" +
		"\tfor i := uint32(0); i != n; i += 2 {\n\t\ty[0] = i\n\t}\n}\n"
	u, err := simt.Transpile(fstest.MapFS{"k.go": &fstest.MapFile{Data: []byte(src)}}, "Spin")
	if err != nil {
		t.Fatalf("Transpile: %v", err)
	}

	start := time.Now()
	err = Run(u, gpu.D1(1), gpu.D1(1), make([]uint32, 1), uint32(1))
	elapsed := time.Since(start)

	var te *TimeoutError
	if !errors.As(err, &te) {
		t.Fatalf("got %v after %s, want a *TimeoutError", err, elapsed)
	}
	// The bound is what ends it, not the compile. A call far longer than the
	// bound would mean the child outlived the kill, which is the failure this
	// whole mechanism exists to prevent.
	if elapsed > 30*time.Second {
		t.Errorf("the bound was 2s but the call took %s", elapsed)
	}
}
