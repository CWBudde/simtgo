//go:build cuda

package simt_test

import (
	"errors"
	"math"
	"math/rand/v2"
	"testing"
	"testing/fstest"

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

// assertEqual is assertClose's counterpart for types that have no tolerance to
// speak of. An integer or a bool that disagrees is a bug, not a rounding.
func assertEqual[T comparable](t *testing.T, name string, got, want []T) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s element %d: gpu %v, cpu %v", name, i, got[i], want[i])
		}
	}
}

// TestQuantizeParity covers the integer half of the widened type map: an int32
// output, an int64 accumulator, a []bool, and a uint32 whose arithmetic wraps.
//
// Everything is compared exactly. Nothing here is floating point once the
// rounding has happened, so a tolerance would only hide a disagreement.
func TestQuantizeParity(t *testing.T) {
	ctx := device(t)
	const n, block = 1 << 14, 256
	const step, seed = 0.05, 0x9e3779b9

	x := randomSignal(n)
	code := make([]int32, n)
	energy := make([]int64, n)
	clipped := make([]bool, n)
	gpu.RunCPU((n+block-1)/block, block, func(c gpu.Ctx) {
		kernels.Quantize(c, code, energy, clipped, x, step, seed)
	})

	// An independent reference for the parts that do not depend on the dither:
	// energy is the square of the code whatever the rounding did, and clipped
	// is exactly the codes at the rails.
	for i := range code {
		if want := int64(code[i]) * int64(code[i]); energy[i] != want {
			t.Fatalf("cpu energy %d: %d, want %d", i, energy[i], want)
		}
		if want := code[i] == 127 || code[i] == -128; clipped[i] != want {
			t.Fatalf("cpu clipped %d: %v, want %v", i, clipped[i], want)
		}
	}

	k, err := simt.Build(ctx, gocuda.Kernels(), "Quantize")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	dcode, _ := cuda.NewSlice[int32](ctx, n)
	denergy, _ := cuda.NewSlice[int64](ctx, n)
	dclipped, _ := cuda.NewSlice[bool](ctx, n)
	dx, _ := cuda.Upload(ctx, x)
	defer dcode.Free()
	defer denergy.Free()
	defer dclipped.Free()
	defer dx.Free()

	if err := k.LaunchN(n, block, dcode, denergy, dclipped, dx, float32(step), uint32(seed)); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	gotCode, err := dcode.Download()
	if err != nil {
		t.Fatalf("Download code: %v", err)
	}
	gotEnergy, err := denergy.Download()
	if err != nil {
		t.Fatalf("Download energy: %v", err)
	}
	gotClipped, err := dclipped.Download()
	if err != nil {
		t.Fatalf("Download clipped: %v", err)
	}
	assertEqual(t, "code", gotCode, code)
	assertEqual(t, "energy", gotEnergy, energy)
	assertEqual(t, "clipped", gotClipped, clipped)
}

// TestBandGainParity covers the struct half: a []Band read as a slice of
// structs, a Shape passed by value, a fixed-size array local, and float64
// accumulation under //gocuda:float64.
//
// The Shape by value is also what the fixed 8-byte parameter slot could not
// carry: at 16 bytes it used to overwrite nothing, because nothing that wide
// existed, and would have overwritten the next parameter the moment one did.
func TestBandGainParity(t *testing.T) {
	ctx := device(t)
	const n, block = 1 << 14, 256

	x := randomSignal(n)
	bands := []kernels.Band{
		{Upper: -0.5, Gain: 0.25},
		{Upper: 0, Gain: 0.5},
		{Upper: 0.5, Gain: 1},
		{Upper: 1, Gain: 2},
	}
	cfg := kernels.Shape{Floor: 0.125, Count: 3, Bias: 0.001}

	want := make([]float32, n)
	gpu.RunCPU((n+block-1)/block, block, func(c gpu.Ctx) {
		kernels.BandGain(c, want, x, bands, cfg)
	})

	// An independent reference, written the obvious way rather than the way the
	// kernel is written: a linear scan for the band and a plain windowed mean.
	ref := make([]float32, n)
	for i := range ref {
		gain := cfg.Floor
		for _, b := range bands {
			if x[i] <= b.Upper {
				gain = b.Gain
				break
			}
		}
		sum := 0.0
		for j := range kernels.BandGainTaps {
			if k := i + j - kernels.BandGainTaps/2; k >= 0 && k < n {
				sum += float64(x[k])
			}
		}
		acc := sum / float64(kernels.BandGainTaps) * float64(gain)
		ref[i] = float32(acc+cfg.Bias) * float32(cfg.Count)
	}
	assertClose(t, want, ref, 1e-6)

	k, err := simt.Build(ctx, gocuda.Kernels(), "BandGain")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	dy, _ := cuda.NewSlice[float32](ctx, n)
	dx, _ := cuda.Upload(ctx, x)
	dbands, _ := cuda.Upload(ctx, bands)
	defer dy.Free()
	defer dx.Free()
	defer dbands.Free()

	if err := k.LaunchN(n, block, dy, dx, dbands, cuda.ArgOf(cfg)); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	got, err := dy.Download()
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	assertClose(t, got, want, 1e-6)
}

// TestStructLayoutRoundTrip pins the field offsets that the generated
// static_asserts cannot.
//
// NVRTC compiles a string with no include path, so it has no offsetof and the
// assertions can only reach sizeof and alignof -- which catch a padding
// disagreement but not a reordering. This puts a distinct value in each field
// and has the device read them back one field per output slot, so an offset
// that disagrees returns another field's value rather than something merely
// close. Bias is the one that matters most: it sits after four bytes of
// padding, which is exactly where two languages would part company.
//
// The probe kernel is built from an inline source rather than committed to
// kernels/, because it is a test of the layout rule and not a kernel anyone
// would launch -- and a committed one would cost a gate entry, a golden and a
// regeneration to say the same thing.
func TestStructLayoutRoundTrip(t *testing.T) {
	ctx := device(t)

	const probe = `package kernels

import "github.com/CWBudde/gocuda/gpu"

type Band struct {
	Upper float32
	Gain  float32
}

type Shape struct {
	Floor float32
	Count int32
	Bias  float64
}

//gocuda:float64
func StructProbe(ctx gpu.Ctx, out []float32, bands []Band, cfg Shape) {
	if ctx.GlobalID() != 0 {
		return
	}
	out[0] = bands[0].Upper
	out[1] = bands[0].Gain
	out[2] = bands[1].Upper
	out[3] = bands[1].Gain
	out[4] = cfg.Floor
	out[5] = float32(cfg.Count)
	out[6] = float32(cfg.Bias)
}
`
	src := fstest.MapFS{"probe.go": &fstest.MapFile{Data: []byte(probe)}}
	k, err := simt.Build(ctx, src, "StructProbe")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	// Values a shifted offset cannot reproduce by accident: no two fields share
	// one, and they differ by orders of magnitude rather than by a little.
	bands := []kernels.Band{{Upper: 1, Gain: 1000}, {Upper: 2, Gain: 2000}}
	cfg := kernels.Shape{Floor: 7, Count: 11, Bias: 13}

	dout, _ := cuda.NewSlice[float32](ctx, 7)
	dbands, _ := cuda.Upload(ctx, bands)
	defer dout.Free()
	defer dbands.Free()

	if err := k.LaunchN(1, 32, dout, dbands, cuda.ArgOf(cfg)); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	got, err := dout.Download()
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	assertEqual(t, "fields", got, []float32{1, 1000, 2, 2000, 7, 11, 13})
}

// TestAtomicHistogramParity is the device half of the atomics vocabulary.
//
// The reference is an ordinary sequential loop rather than the same kernel
// under RunCPU. That is deliberate and it is stronger: a loop agrees with the
// device only if the device's adds really were atomic, whereas the emulator
// and the device could in principle both lose updates in ways that happened to
// cancel. Integers make the comparison exact regardless of the order the
// threads arrived in, which is what lets this assert equality at all.
//
// The probe kernel is inline rather than committed to kernels/, for the reason
// TestStructLayoutRoundTrip gives: it tests a rule, not a kernel anyone would
// launch, and committing one would cost a gate entry, a golden and a
// regeneration to say the same thing.
func TestAtomicHistogramParity(t *testing.T) {
	ctx := device(t)

	const probe = `package kernels

import "github.com/CWBudde/gocuda/gpu"

func AtomicProbe(ctx gpu.Ctx, bins []int32, x []int32) {
	i := ctx.GlobalID()
	if i < len(x) {
		gpu.AtomicAddI32(bins, int(x[i])%len(bins), 1)
	}
}
`
	src := fstest.MapFS{"probe.go": &fstest.MapFile{Data: []byte(probe)}}
	k, err := simt.Build(ctx, src, "AtomicProbe")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	const n, bins = 1 << 14, 16
	x := make([]int32, n)
	for i := range x {
		// Deliberately uneven, so a bin that is never written is visible as a
		// zero rather than hidden by every bin holding the same count.
		x[i] = int32(i*i%97 + i%7)
	}
	want := make([]int32, bins)
	for _, v := range x {
		want[int(v)%bins]++
	}

	dbins, _ := cuda.NewSlice[int32](ctx, bins)
	dx, _ := cuda.Upload(ctx, x)
	defer dbins.Free()
	defer dx.Free()

	if err := k.LaunchN(n, 256, dbins, dx); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	got, err := dbins.Download()
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	assertEqual(t, "bins", got, want)
}

// TestAtomicCASParity pins the primitive the others cannot stand in for: every
// thread in the grid races to claim one flag, and exactly one may win.
//
// A compare-and-swap that was not atomic would let two threads read the same
// zero and both believe they won, which the winner count catches; one that
// returned the compared value rather than the held one would make every thread
// believe it won.
func TestAtomicCASParity(t *testing.T) {
	ctx := device(t)

	const probe = `package kernels

import "github.com/CWBudde/gocuda/gpu"

func CASProbe(ctx gpu.Ctx, out []int32) {
	i := ctx.GlobalID()
	if gpu.AtomicCASI32(out, 0, 0, int32(i)+1) == 0 {
		gpu.AtomicAddI32(out, 1, 1)
	}
}
`
	src := fstest.MapFS{"probe.go": &fstest.MapFile{Data: []byte(probe)}}
	k, err := simt.Build(ctx, src, "CASProbe")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	const n = 1 << 12
	dout, _ := cuda.NewSlice[int32](ctx, 2)
	defer dout.Free()

	if err := k.LaunchN(n, 256, dout); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	got, err := dout.Download()
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	if got[1] != 1 {
		t.Errorf("%d threads believed they won the flag, want exactly 1", got[1])
	}
	if got[0] < 1 || got[0] > n {
		t.Errorf("the flag holds %d, which is no thread's identity", got[0])
	}
}

// TestAtomicSharedTileParity reaches the case whose C is indistinguishable
// from the global one: &s[i] on a __shared__ array is a generic pointer, and
// it is the hardware that resolves it back to a shared atomic.
//
// Every thread adds 1.0, so each partial sum is a small integer and exact in
// float32 whatever order the adds happened in. That is what lets the result be
// compared exactly; a kernel adding real data could not be, because float
// addition is not associative and the device's order is not the CPU's.
func TestAtomicSharedTileParity(t *testing.T) {
	ctx := device(t)

	const probe = `package kernels

import "github.com/CWBudde/gocuda/gpu"

func TileProbe(ctx gpu.Ctx, out []float32) {
	s := ctx.SharedF32(1)
	if ctx.ThreadIdx() == 0 {
		s[0] = 0
	}
	ctx.SyncThreads()
	gpu.AtomicAddF32(s, 0, 1)
	ctx.SyncThreads()
	if ctx.ThreadIdx() == 0 {
		gpu.AtomicAddF32(out, 0, s[0])
	}
}
`
	src := fstest.MapFS{"probe.go": &fstest.MapFile{Data: []byte(probe)}}
	k, err := simt.Build(ctx, src, "TileProbe")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	const blocks, block = 32, 128
	dout, _ := cuda.NewSlice[float32](ctx, 1)
	defer dout.Free()

	if err := k.LaunchDim(cuda.Dim3{X: blocks, Y: 1, Z: 1}, cuda.Dim3{X: block, Y: 1, Z: 1}, dout); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	got, err := dout.Download()
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	assertEqual(t, "total", got, []float32{blocks * block})
}
