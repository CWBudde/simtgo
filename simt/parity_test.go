//go:build cuda

package simt_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/CWBudde/gocuda"
	"github.com/CWBudde/gocuda/cuda"
	"github.com/CWBudde/gocuda/gpu"
	"github.com/CWBudde/gocuda/kernels"
	"github.com/CWBudde/gocuda/simt"
)

// These tests are the point of the SIMT track: the same Go function is run by
// the CPU emulator and, after transpilation, by the GPU. Any divergence is a
// transpiler bug.

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

func randomSignal(n int) []float32 {
	r := rand.New(rand.NewPCG(1, 2))
	xs := make([]float32, n)
	for i := range xs {
		xs[i] = float32(r.NormFloat64())
	}
	return xs
}

func assertClose(t *testing.T, got, want []float32, tol float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length %d, want %d", len(got), len(want))
	}
	for i := range got {
		d := float32(math.Abs(float64(got[i] - want[i])))
		scale := max(float32(math.Abs(float64(want[i]))), 1)
		if d/scale > tol {
			t.Fatalf("element %d: gpu %v, cpu %v (delta %v)", i, got[i], want[i], d)
		}
	}
}

func TestVecAddParity(t *testing.T) {
	ctx := device(t)
	const n, block = 1 << 16, 256
	a, b := randomSignal(n), randomSignal(n)

	want := make([]float32, n)
	gpu.RunCPU((n+block-1)/block, block, func(c gpu.Ctx) { kernels.VecAdd(c, want, a, b) })

	k, err := simt.Build(ctx, gocuda.Kernels(), "VecAdd")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	dc, _ := cuda.NewSlice[float32](ctx, n)
	da, _ := cuda.Upload(ctx, a)
	db, _ := cuda.Upload(ctx, b)
	defer dc.Free()
	defer da.Free()
	defer db.Free()

	if err := k.LaunchN(n, block, dc, da, db); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	got, err := dc.Download()
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	assertClose(t, got, want, 1e-6)
}

func TestScaleParity(t *testing.T) {
	ctx := device(t)
	const n, block = 1 << 14, 128
	x := randomSignal(n)
	const factor float32 = 0.75

	want := make([]float32, n)
	gpu.RunCPU((n+block-1)/block, block, func(c gpu.Ctx) { kernels.Scale(c, want, x, factor) })

	k, err := simt.Build(ctx, gocuda.Kernels(), "Scale")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	dy, _ := cuda.NewSlice[float32](ctx, n)
	dx, _ := cuda.Upload(ctx, x)
	defer dy.Free()
	defer dx.Free()

	if err := k.LaunchN(n, block, dy, dx, factor); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	got, _ := dy.Download()
	assertClose(t, got, want, 1e-6)
}

func TestMagnitudeParity(t *testing.T) {
	ctx := device(t)
	const n, block = 1 << 14, 256
	re, im := randomSignal(n), randomSignal(n)

	want := make([]float32, n)
	gpu.RunCPU((n+block-1)/block, block, func(c gpu.Ctx) { kernels.Magnitude(c, want, re, im) })

	k, err := simt.Build(ctx, gocuda.Kernels(), "Magnitude")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	dm, _ := cuda.NewSlice[float32](ctx, n)
	dre, _ := cuda.Upload(ctx, re)
	dim, _ := cuda.Upload(ctx, im)
	defer dm.Free()
	defer dre.Free()
	defer dim.Free()

	if err := k.LaunchN(n, block, dm, dre, dim); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	got, _ := dm.Download()
	assertClose(t, got, want, 1e-6)
}

// TestFIRParity is the interesting one: shared memory, a halo, a barrier and
// a tap loop all have to line up between the CPU emulator and the device.
func TestFIRParity(t *testing.T) {
	ctx := device(t)
	const n = 1 << 15
	const block = kernels.FIRBlock
	x := randomSignal(n)
	h := make([]float32, 33)
	for i := range h {
		h[i] = float32(1) / float32(len(h))
	}

	want := make([]float32, n)
	gpu.RunCPU((n+block-1)/block, block, func(c gpu.Ctx) { kernels.FIR(c, want, x, h) })

	// An independent reference, so a shared misunderstanding between the two
	// backends cannot pass as agreement.
	ref := make([]float32, n)
	for i := range ref {
		var acc float32
		for k := range h {
			if i-k >= 0 {
				acc += h[k] * x[i-k]
			}
		}
		ref[i] = acc
	}
	assertClose(t, want, ref, 1e-5)

	k, err := simt.Build(ctx, gocuda.Kernels(), "FIR")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	dy, _ := cuda.NewSlice[float32](ctx, n)
	dx, _ := cuda.Upload(ctx, x)
	dh, _ := cuda.Upload(ctx, h)
	defer dy.Free()
	defer dx.Free()
	defer dh.Free()

	if err := k.LaunchN(n, block, dy, dx, dh); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	got, _ := dy.Download()
	assertClose(t, got, want, 1e-5)
}

// TestClassifyParity covers the statement forms the transpiler gained
// together: a switch over a constant set of bands, a labelled break out of a
// nested search, and a range loop that binds the value.
func TestClassifyParity(t *testing.T) {
	ctx := device(t)
	const n, block = 1 << 14, 128
	x := randomSignal(n)
	edges := []float32{0.1, 0.25, 0.5, 0.8, 1.2, 1.8, 2.5}

	want := make([]float32, n)
	gpu.RunCPU((n+block-1)/block, block, func(c gpu.Ctx) { kernels.Classify(c, want, x, edges) })

	// An independent reference: the band is the index of the first edge the
	// sample does not exceed, written as a flat scan rather than the kernel's
	// grouped one.
	ref := make([]float32, n)
	var bias float32
	for _, e := range edges {
		bias += e
	}
	for i, v := range x {
		v = float32(math.Abs(float64(v)))
		band := len(edges)
		for j, e := range edges {
			if v <= e {
				band = j
				break
			}
		}
		gain := float32(1)
		switch band {
		case 0:
			gain = 0.25
		case 1, 2:
			gain = 0.5
		case 3:
			gain = 0.75
		}
		ref[i] = gain*v + bias*0.001
	}
	assertClose(t, want, ref, 1e-6)

	k, err := simt.Build(ctx, gocuda.Kernels(), "Classify")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	dout, _ := cuda.NewSlice[float32](ctx, n)
	dx, _ := cuda.Upload(ctx, x)
	de, _ := cuda.Upload(ctx, edges)
	defer dout.Free()
	defer dx.Free()
	defer de.Free()

	if err := k.LaunchN(n, block, dout, dx, de); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	got, err := dout.Download()
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	assertClose(t, got, want, 1e-6)
}
