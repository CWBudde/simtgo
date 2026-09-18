//go:build cuda

package tile_test

import (
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/CWBudde/gocuda/cuda"
	"github.com/CWBudde/gocuda/internal/tolerance"
	"github.com/CWBudde/gocuda/tile"
)

func device(t *testing.T) *cuda.Context {
	t.Helper()
	if !cuda.Available() {
		t.Skip("no CUDA device available")
	}
	ctx, err := cuda.NewContext(0)
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	t.Cleanup(func() { ctx.Close() })
	return ctx
}

func signal(n int, seed uint64) []float32 {
	r := rand.New(rand.NewPCG(seed, 99))
	xs := make([]float32, n)
	for i := range xs {
		xs[i] = float32(r.NormFloat64())
	}
	return xs
}

// The float32 comparisons here are tolerance.AssertClose, which is the same
// rule the SIMT parity tests apply and is stated in NUMERICS.md. This file
// used to carry a second copy of it under the same name, differing from the
// other one in signature but not in what it bounded -- one rule spelled twice
// is one rule waiting to become two.

// TestPipeline checks the fused kernel against a plain Go implementation of
// the same pipeline, and against the op-at-a-time path.
func TestPipeline(t *testing.T) {
	ctx := device(t)
	const n = 1 << 14
	re, im := signal(n, 1), signal(n, 2)
	h := make([]float32, 17)
	for i := range h {
		h[i] = 1 / float32(len(h))
	}

	g := tile.New(ctx)
	filtered := tile.FIR(tile.Scale(g.Input(re), 0.5), h)
	out := tile.Hypot(filtered, g.Input(im))

	src, err := out.Source()
	if err != nil {
		t.Fatalf("Source: %v", err)
	}
	if strings.Count(src, "__global__") != 1 {
		t.Fatalf("expected exactly one kernel:\n%s", src)
	}

	got, err := out.Materialize()
	if err != nil {
		t.Fatalf("Materialize: %v", err)
	}

	want := make([]float32, n)
	for i := range want {
		var acc float32
		for k := range h {
			if i-k >= 0 {
				acc += h[k] * 0.5 * re[i-k]
			}
		}
		want[i] = float32(math.Hypot(float64(acc), float64(im[i])))
	}
	tolerance.AssertClose(t, "fused vs reference", got, want, 1e-5)

	stepwise, launches, err := out.MaterializeStepwise()
	if err != nil {
		t.Fatalf("MaterializeStepwise: %v", err)
	}
	if launches != 3 {
		t.Errorf("stepwise used %d launches, want 3 (scale, fir, hypot)", launches)
	}
	tolerance.AssertClose(t, "stepwise vs fused", stepwise, got, 1e-6)
}

// TestShapeMismatch checks the runtime shape check that stands in for the
// static guarantees Rust's const generics provide.
func TestShapeMismatch(t *testing.T) {
	ctx := device(t)
	g := tile.New(ctx)
	out := tile.Add(g.Input(make([]float32, 8)), g.Input(make([]float32, 9)))
	if _, err := out.Materialize(); err == nil || !strings.Contains(err.Error(), "does not match") {
		t.Fatalf("got %v, want a shape error", err)
	}
}

// TestChainedWindowRejected pins the documented limitation.
func TestChainedWindowRejected(t *testing.T) {
	ctx := device(t)
	g := tile.New(ctx)
	h := []float32{1, 2, 3}
	out := tile.FIR(tile.FIR(g.Input(signal(1024, 3)), h), h)
	if _, err := out.Materialize(); err == nil || !strings.Contains(err.Error(), "chaining windowed") {
		t.Fatalf("got %v, want a chaining error", err)
	}
}
