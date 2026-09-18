//go:build cuda

package simt_test

import (
	"errors"
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

// TestSoftclipParity covers a kernel that calls another Go function. The CPU
// side needs nothing for this: a device function is ordinary Go, so the
// emulator runs the very same code the device compiles.
func TestSoftclipParity(t *testing.T) {
	ctx := device(t)
	const n, block = 1 << 14, 256
	const threshold float32 = 0.5
	x := randomSignal(n)

	want := make([]float32, n)
	gpu.RunCPU((n+block-1)/block, block, func(c gpu.Ctx) { kernels.Softclip(c, want, x, threshold) })

	// An independent reference, written as one expression rather than the
	// kernel's branches.
	ref := make([]float32, n)
	for i, v := range x {
		a := math.Abs(float64(v))
		s := a
		if a > float64(threshold) {
			over := a - float64(threshold)
			s = float64(threshold) + over/(1+over)
		}
		ref[i] = float32(math.Copysign(s, float64(v)))
	}
	assertClose(t, want, ref, 1e-6)

	k, err := simt.Build(ctx, gocuda.Kernels(), "Softclip")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	dy, _ := cuda.NewSlice[float32](ctx, n)
	dx, _ := cuda.Upload(ctx, x)
	defer dy.Free()
	defer dx.Free()

	if err := k.LaunchN(n, block, dy, dx, threshold); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	got, err := dy.Download()
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	assertClose(t, got, want, 1e-6)
}

// TestTransposeParity is the two-dimensional launch end to end: a 2-D grid of
// 2-D blocks, staged through shared memory, emulated by RunCPUDim and
// launched by LaunchDim.
func TestTransposeParity(t *testing.T) {
	ctx := device(t)
	// Deliberately not multiples of the tile, so the ragged edges of the grid
	// are covered on both axes.
	const w, h = 133, 71
	const tile = kernels.TransposeTile
	in := randomSignal(w * h)

	grid := gpu.D2((w+tile-1)/tile, (h+tile-1)/tile)
	block := gpu.D2(tile, tile)

	want := make([]float32, w*h)
	gpu.RunCPUDim(grid, block, func(c gpu.Ctx) { kernels.Transpose(c, want, in, w, h) })

	// An independent reference: the transpose written the obvious way.
	ref := make([]float32, w*h)
	for y := range h {
		for x := range w {
			ref[x*h+y] = in[y*w+x]
		}
	}
	assertClose(t, want, ref, 0)

	k, err := simt.Build(ctx, gocuda.Kernels(), "Transpose")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	dout, _ := cuda.NewSlice[float32](ctx, w*h)
	din, _ := cuda.Upload(ctx, in)
	defer dout.Free()
	defer din.Free()

	if err := k.LaunchDim(cuda.D2(grid.X, grid.Y), cuda.D2(tile, tile), dout, din, int32(w), int32(h)); err != nil {
		t.Fatalf("LaunchDim: %v", err)
	}
	got, err := dout.Download()
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	assertClose(t, got, want, 0)
}

// TestLaunchDimCountsTheWholeBlock pins the launch contract against a 2-D
// block: AssumeBlockDim counts threads, not the x extent, so a 16x16 block
// satisfies a kernel that asked for 256 and a 16x8 one does not.
func TestLaunchDimCountsTheWholeBlock(t *testing.T) {
	ctx := device(t)
	k, err := simt.Build(ctx, gocuda.Kernels(), "Transpose")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	dout, _ := cuda.NewSlice[float32](ctx, 16)
	din, _ := cuda.Upload(ctx, make([]float32, 16))
	defer dout.Free()
	defer din.Free()

	err = k.LaunchDim(cuda.D1(1), cuda.D2(16, 8), dout, din, int32(4), int32(4))
	var bad *simt.BlockSizeError
	if !errors.As(err, &bad) {
		t.Fatalf("launching a 128-thread block gave %v, want a BlockSizeError", err)
	}
	if bad.Want != 256 || bad.Got != 128 {
		t.Errorf("BlockSizeError says want %d got %d, expected 256 and 128", bad.Want, bad.Got)
	}
}
