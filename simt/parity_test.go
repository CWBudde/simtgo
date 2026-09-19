//go:build cuda

package simt_test

import (
	"errors"
	"math"
	"math/rand/v2"
	"os"
	"testing"
	"testing/fstest"

	"github.com/CWBudde/simtgo"
	"github.com/CWBudde/simtgo/cuda"
	"github.com/CWBudde/simtgo/gpu"
	"github.com/CWBudde/simtgo/internal/tolerance"
	"github.com/CWBudde/simtgo/kernels"
	"github.com/CWBudde/simtgo/simt"
)

// These tests are the point of the SIMT track: the same Go function is run by
// the CPU emulator and, after transpilation, by the GPU. Any divergence is a
// transpiler bug.

func device(t *testing.T) *cuda.Context {
	t.Helper()
	if !cuda.Available() {
		// A sanitizer run that launches nothing is green and means nothing,
		// so a job that has promised a device says so and fails instead of
		// skipping. See .github/workflows/sanitizer.yml.
		if os.Getenv("SIMTGO_REQUIRE_DEVICE") != "" {
			t.Fatal("SIMTGO_REQUIRE_DEVICE is set, but no CUDA device is available")
		}
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

// The comparisons below are tolerance.AssertClose and tolerance.AssertEqual
// rather than helpers of this file's own. The rule they apply is stated in
// NUMERICS.md and implemented once, because two of the three places in this
// repository that compare a float32 result are in other packages and had
// drifted into other rules.

func TestVecAddParity(t *testing.T) {
	ctx := device(t)
	const n, block = 1 << 16, 256
	a, b := randomSignal(n), randomSignal(n)

	want := make([]float32, n)
	gpu.RunCPU((n+block-1)/block, block, func(c gpu.Ctx) { kernels.VecAdd(c, want, a, b) })

	k, err := simt.Build(ctx, simtgo.Kernels(), "VecAdd")
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
	tolerance.AssertClose(t, "gpu vs cpu", got, want, 1e-6)
}

func TestScaleParity(t *testing.T) {
	ctx := device(t)
	const n, block = 1 << 14, 128
	x := randomSignal(n)
	const factor float32 = 0.75

	want := make([]float32, n)
	gpu.RunCPU((n+block-1)/block, block, func(c gpu.Ctx) { kernels.Scale(c, want, x, factor) })

	k, err := simt.Build(ctx, simtgo.Kernels(), "Scale")
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
	tolerance.AssertClose(t, "gpu vs cpu", got, want, 1e-6)
}

func TestMagnitudeParity(t *testing.T) {
	ctx := device(t)
	const n, block = 1 << 14, 256
	re, im := randomSignal(n), randomSignal(n)

	want := make([]float32, n)
	gpu.RunCPU((n+block-1)/block, block, func(c gpu.Ctx) { kernels.Magnitude(c, want, re, im) })

	k, err := simt.Build(ctx, simtgo.Kernels(), "Magnitude")
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
	tolerance.AssertClose(t, "gpu vs cpu", got, want, 1e-6)
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
	tolerance.AssertClose(t, "cpu vs reference", want, ref, 1e-5)

	k, err := simt.Build(ctx, simtgo.Kernels(), "FIR")
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
	tolerance.AssertClose(t, "gpu vs cpu", got, want, 1e-5)
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
	tolerance.AssertClose(t, "cpu vs reference", want, ref, 1e-6)

	k, err := simt.Build(ctx, simtgo.Kernels(), "Classify")
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
	tolerance.AssertClose(t, "gpu vs cpu", got, want, 1e-6)
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
	tolerance.AssertClose(t, "cpu vs reference", want, ref, 1e-6)

	k, err := simt.Build(ctx, simtgo.Kernels(), "Softclip")
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
	tolerance.AssertClose(t, "gpu vs cpu", got, want, 1e-6)
}

// TestTransposeParity is the two-dimensional launch end to end: a 2-D grid of
// 2-D blocks, staged through shared memory, emulated by RunCPUDim and
// launched by LaunchDim.
//
// The comparisons are exact, and the rule in NUMERICS.md is why: a transpose
// moves values and does arithmetic on none of them, so every float32 that
// comes back is one that went in. It used to be spelled as a tolerance of
// zero, which asserts the same thing while reading as though there were
// something here to round.
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
	tolerance.AssertEqual(t, "cpu vs reference", want, ref)

	k, err := simt.Build(ctx, simtgo.Kernels(), "Transpose")
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
	tolerance.AssertEqual(t, "gpu vs cpu", got, want)
}

// TestLaunchDimCountsTheWholeBlock pins the launch contract against a 2-D
// block: AssumeBlockDim counts threads, not the x extent, so a 16x16 block
// satisfies a kernel that asked for 256 and a 16x8 one does not.
func TestLaunchDimCountsTheWholeBlock(t *testing.T) {
	ctx := device(t)
	k, err := simt.Build(ctx, simtgo.Kernels(), "Transpose")
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

	k, err := simt.Build(ctx, simtgo.Kernels(), "Quantize")
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
	tolerance.AssertEqual(t, "code", gotCode, code)
	tolerance.AssertEqual(t, "energy", gotEnergy, energy)
	tolerance.AssertEqual(t, "clipped", gotClipped, clipped)
}

// TestBandGainParity covers the struct half: a []Band read as a slice of
// structs, a Shape passed by value, a fixed-size array local, and float64
// accumulation under //simtgo:float64.
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
	tolerance.AssertClose(t, "cpu vs reference", want, ref, 1e-6)

	k, err := simt.Build(ctx, simtgo.Kernels(), "BandGain")
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
	tolerance.AssertClose(t, "gpu vs cpu", got, want, 1e-6)
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

import "github.com/CWBudde/simtgo/gpu"

type Band struct {
	Upper float32
	Gain  float32
}

type Shape struct {
	Floor float32
	Count int32
	Bias  float64
}

//simtgo:float64
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
	tolerance.AssertEqual(t, "fields", got, []float32{1, 1000, 2, 2000, 7, 11, 13})
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

import "github.com/CWBudde/simtgo/gpu"

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

	// Uploaded zeros rather than NewSlice: the kernel only ever adds to these
	// bins, so an uninitialised allocation would be counted as part of the
	// histogram. NewSlice does not clear what it hands back, and on a device
	// that has been used the difference is not academic.
	dbins, _ := cuda.Upload(ctx, make([]int32, bins))
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
	tolerance.AssertEqual(t, "bins", got, want)
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

import "github.com/CWBudde/simtgo/gpu"

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
	// Zeros for the same reason, and here it decides the test rather than
	// skewing it: the flag has to start at 0 for any thread to win the CAS,
	// and the winner count has to start at 0 to mean anything.
	dout, _ := cuda.Upload(ctx, make([]int32, 2))
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

import "github.com/CWBudde/simtgo/gpu"

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
	// Zeros again: every block adds its partial sum into out[0].
	dout, _ := cuda.Upload(ctx, make([]float32, 1))
	defer dout.Free()

	if err := k.LaunchDim(cuda.Dim3{X: blocks, Y: 1, Z: 1}, cuda.Dim3{X: block, Y: 1, Z: 1}, dout); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	got, err := dout.Download()
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	tolerance.AssertEqual(t, "total", got, []float32{blocks * block})
}

// TestGrayParity is the narrow storage types end to end: []uint8 in, []uint8
// out, every arithmetic step at int32.
//
// The reference is written out here rather than taken from the emulator run,
// and it is exact. Integer weights are what make that possible: a float32
// pipeline would need a tolerance, and a tolerance is where a one-bit
// disagreement in how the two languages truncated a byte would hide. The whole
// point of this kernel is that no byte is ever an operand, so nothing here may
// be approximately right.
func TestGrayParity(t *testing.T) {
	ctx := device(t)
	const n, block = 1 << 14, 256

	rgb := make([]uint8, 3*n)
	for i := range rgb {
		// 37 is coprime with 256, so every channel takes every value; the
		// second term keeps the three channels of a pixel from moving in step.
		rgb[i] = uint8((i*37 + i/251) % 256)
	}
	// The two pixels the weights have to land on exactly. White is where a sum
	// that did not round, or weights that did not add up to 1<<GrayShift, comes
	// back as 254; black is where anything that carried a stray term shows.
	for c := range 3 {
		rgb[c], rgb[3+c] = 255, 0
	}

	want := make([]uint8, n)
	gpu.RunCPU((n+block-1)/block, block, func(c gpu.Ctx) { kernels.Gray(c, want, rgb) })

	ref := make([]uint8, n)
	for i := range ref {
		r, g, b := int(rgb[3*i]), int(rgb[3*i+1]), int(rgb[3*i+2])
		v := (kernels.GrayWeightR*r + kernels.GrayWeightG*g + kernels.GrayWeightB*b + 1<<(kernels.GrayShift-1)) >> kernels.GrayShift
		ref[i] = uint8(v)
	}
	tolerance.AssertEqual(t, "cpu luma", want, ref)

	k, err := simt.Build(ctx, simtgo.Kernels(), "Gray")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	dout, _ := cuda.NewSlice[uint8](ctx, n)
	drgb, _ := cuda.Upload(ctx, rgb)
	defer dout.Free()
	defer drgb.Free()

	if err := k.LaunchN(n, block, dout, drgb); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	got, err := dout.Download()
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	tolerance.AssertEqual(t, "luma", got, want)
}

// TestNarrowStorageRoundTrip pins the stride rather than the arithmetic.
//
// Gray covers a []uint8 computing something. What it cannot cover is the
// other three widths, and the question they raise is the one []int leaves
// behind: cuda.Upload copies Go's bytes, so if the device read an element at a
// different width every value after the first would be wrong. Each buffer here
// holds a pattern whose elements differ in every byte, and the kernel copies
// element i of each into its own output slot, so a stride that disagreed comes
// back as another element's value rather than as something merely close.
//
// The probe kernel is inline rather than committed to kernels/, for the reason
// TestStructLayoutRoundTrip gives: it tests a rule, not a kernel anyone would
// launch.
func TestNarrowStorageRoundTrip(t *testing.T) {
	ctx := device(t)

	const probe = `package kernels

import "github.com/CWBudde/simtgo/gpu"

func NarrowProbe(ctx gpu.Ctx, out []int32, a []int8, b []uint8, c []int16, d []uint16) {
	i := ctx.GlobalID()
	if i >= len(a) {
		return
	}
	out[4*i+0] = int32(a[i])
	out[4*i+1] = int32(b[i])
	out[4*i+2] = int32(c[i])
	out[4*i+3] = int32(d[i])
}
`
	src := fstest.MapFS{"probe.go": &fstest.MapFile{Data: []byte(probe)}}
	k, err := simt.Build(ctx, src, "NarrowProbe")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	const n = 64
	a := make([]int8, n)
	b := make([]uint8, n)
	c := make([]int16, n)
	d := make([]uint16, n)
	want := make([]int32, 4*n)
	for i := range a {
		// Both signs and both rails, so a signed element read as unsigned --
		// or a "char" whose signedness the compiler chose for itself -- is a
		// wrong number and not a coincidence.
		a[i] = int8(i - 100)
		b[i] = uint8(200 - i)
		c[i] = int16(i*517 - 20000)
		d[i] = uint16(60000 - i*601)
		want[4*i+0] = int32(a[i])
		want[4*i+1] = int32(b[i])
		want[4*i+2] = int32(c[i])
		want[4*i+3] = int32(d[i])
	}

	dout, _ := cuda.NewSlice[int32](ctx, 4*n)
	da, _ := cuda.Upload(ctx, a)
	db, _ := cuda.Upload(ctx, b)
	dc, _ := cuda.Upload(ctx, c)
	dd, _ := cuda.Upload(ctx, d)
	defer dout.Free()
	defer da.Free()
	defer db.Free()
	defer dc.Free()
	defer dd.Free()

	if err := k.LaunchN(n, 32, dout, da, db, dc, dd); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	got, err := dout.Download()
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	tolerance.AssertEqual(t, "elements", got, want)
}

// arrayFieldTaps mirrors the struct the probe below declares, so that
// cuda.Upload copies Go's layout of exactly that type.
//
// The field order is chosen to leave a hole in both places one can be. N is one
// byte and W wants four, so Go pads three; W ends at 16 and Gain wants eight,
// which lands with nothing between them; Tag is one byte at 24 and the struct
// is eight-aligned, so seven bytes of slack close it at 32. The emitter has to
// declare both of those holes, and the trailing one is the interesting half --
// a field placed wrongly earlier could otherwise hide inside it and leave
// sizeof unchanged.
type arrayFieldTaps struct {
	N    uint8
	W    [3]float32
	Gain float64
	Tag  uint8
}

// TestArrayFieldLayoutRoundTrip extends the struct round trip to the field
// shape that was refused until the offsets could be pinned.
//
// An array member is where a size assertion says least: sizeof counts the whole
// array, so a struct that put W one slot earlier or later would come out
// exactly the same size. Reading each element back through the device is what
// says where it actually is. The values differ by orders of magnitude, and the
// second element is read from a second struct, so a slipped offset or a wrong
// stride returns another field's number rather than a nearby one.
func TestArrayFieldLayoutRoundTrip(t *testing.T) {
	ctx := device(t)

	const probe = `package kernels

import "github.com/CWBudde/simtgo/gpu"

type Taps struct {
	N    uint8
	W    [3]float32
	Gain float64
	Tag  uint8
}

//simtgo:float64
func ArrayFieldProbe(ctx gpu.Ctx, out []float32, ts []Taps) {
	if ctx.GlobalID() != 0 {
		return
	}
	out[0] = float32(int32(ts[0].N))
	out[1] = ts[0].W[0]
	out[2] = ts[0].W[1]
	out[3] = ts[0].W[2]
	out[4] = float32(ts[0].Gain)
	out[5] = float32(int32(ts[0].Tag))
	out[6] = float32(int32(ts[1].N))
	out[7] = ts[1].W[2]
	out[8] = float32(ts[1].Gain)
	out[9] = float32(int32(ts[1].Tag))
}
`
	src := fstest.MapFS{"probe.go": &fstest.MapFile{Data: []byte(probe)}}
	k, err := simt.Build(ctx, src, "ArrayFieldProbe")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	ts := []arrayFieldTaps{
		{N: 3, W: [3]float32{1, 10, 100}, Gain: 1000, Tag: 5},
		{N: 7, W: [3]float32{2, 20, 200}, Gain: 2000, Tag: 9},
	}

	dout, _ := cuda.NewSlice[float32](ctx, 10)
	dts, _ := cuda.Upload(ctx, ts)
	defer dout.Free()
	defer dts.Free()

	if err := k.LaunchN(1, 32, dout, dts); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	got, err := dout.Download()
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	tolerance.AssertEqual(t, "fields", got, []float32{3, 1, 10, 100, 1000, 5, 7, 200, 2000, 9})
}

// TestAliasedLaunchIsRefused is the other half of __restrict__.
//
// The generated C promises that a kernel's pointer parameters do not overlap,
// and the Go source cannot make that promise: VecAdd(ctx, c, a, b) is three
// parameters and one buffer may fill all three. So the promise is checked
// where the buffers are known, and a launch that breaks it is refused with an
// error instead of producing whatever that architecture's scheduler made of
// the reordering it was told it could do.
//
// The accepted half is here too, and it is the half that keeps the check
// honest: two parameters the kernel only reads may share memory, because
// __restrict__ only forbids reaching a *modified* object through another
// pointer. A check that refused every repeat would forbid `dot(x, x)`.
func TestAliasedLaunchIsRefused(t *testing.T) {
	ctx := device(t)
	k, err := simt.Build(ctx, simtgo.Kernels(), "VecAdd")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	const n, block = 1024, 256
	d1, _ := cuda.Upload(ctx, randomSignal(n))
	d2, _ := cuda.Upload(ctx, randomSignal(n))
	defer d1.Free()
	defer d2.Free()

	// c and a are the same buffer, and VecAdd writes c.
	err = k.LaunchN(n, block, d1, d1, d2)
	var bad *simt.AliasError
	if !errors.As(err, &bad) {
		t.Fatalf("launching with c and a aliased gave %v, want an AliasError", err)
	}
	if bad.Write != "c" || bad.Other != "a" {
		t.Errorf("AliasError names %s and %s, want c and a", bad.Write, bad.Other)
	}

	// The two read-only parameters may be the same buffer: nothing is
	// modified through either, so there is nothing for restrict to forbid.
	if err := k.LaunchN(n, block, d1, d2, d2); err != nil {
		t.Errorf("two read-only parameters sharing a buffer were refused: %v", err)
	}
}

// TestParallelAssignmentParity runs the two-phase assignment on the device.
//
// It is a probe rather than a committed kernel for the reason the struct
// round-trip is: it tests a lowering rule, not a kernel anyone would launch.
// The expectations are written out rather than taken from RunCPU, because both
// cases exist to catch a device doing something Go does not -- and the second
// one would still be wrong if the emulator and the emitter made the same
// mistake.
//
// The reversal is the readable half. The lifted index is the half that would
// fail: Go evaluates the subscript in `i, y[i] = 2, 7` against the old i, so
// the 7 lands in y[0], while a C lowering that assigned in order would put it
// in y[2] -- code that compiles and computes something else.
func TestParallelAssignmentParity(t *testing.T) {
	ctx := device(t)

	const probe = "package kernels\n\n" +
		"import \"github.com/CWBudde/simtgo/gpu\"\n\n" +
		"func ReverseProbe(ctx gpu.Ctx, y []float32) {\n" +
		"\tif ctx.GlobalID() != 0 {\n\t\treturn\n\t}\n" +
		"\ti := 0\n\tj := len(y) - 1\n" +
		"\tfor i < j {\n\t\ty[i], y[j] = y[j], y[i]\n\t\ti, j = i+1, j-1\n\t}\n}\n\n" +
		"func IndexProbe(ctx gpu.Ctx, y []float32) {\n" +
		"\tif ctx.GlobalID() != 0 {\n\t\treturn\n\t}\n" +
		"\ti := 0\n\ti, y[i] = 2, 7\n\ty[1] = float32(i)\n}\n"

	src := fstest.MapFS{"probe.go": &fstest.MapFile{Data: []byte(probe)}}

	rev, err := simt.Build(ctx, src, "ReverseProbe")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	dy, _ := cuda.Upload(ctx, []float32{1, 2, 3, 4, 5, 6, 7})
	defer dy.Free()
	if err := rev.Launch(1, 32, dy); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	got, err := dy.Download()
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	tolerance.AssertEqual(t, "reversed", got, []float32{7, 6, 5, 4, 3, 2, 1})

	idx, err := simt.Build(ctx, src, "IndexProbe")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	dz, _ := cuda.Upload(ctx, []float32{0, 0, 0})
	defer dz.Free()
	if err := idx.Launch(1, 32, dz); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	got, err = dz.Download()
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	tolerance.AssertEqual(t, "lifted index", got, []float32{7, 2, 0})
}

// TestMathHelperParity runs five of the seven float32 helpers that no kernel
// in kernels/ reaches, so that the name table in internal/lower is checked by
// something that runs rather than only by a golden file. Fmin and Fmax are the
// other two and have a test of their own below.
//
// The reference is binary64 and is not independent of the emulator: gpu.Sqrt
// and the rest are float32(math.F(float64(x))) already, so the emulator half
// of this comparison is a tautology. It is asserted anyway, and exactly,
// because that identity is the contract -- a helper quietly reimplemented as a
// float32 approximation would still pass a tolerance against itself. What the
// binary64 reference is genuinely for is the device half: rounded once to
// float32 it is within half an ulp of the exact value, so the distance
// measured below is very nearly the device library's own error.
//
// That is also why this cannot be tightened from here. NVIDIA documents sqrtf
// as correctly rounded under --prec-sqrt=true, logf within 1 ulp and sinf,
// cosf and expf within 2 (CUDA C++ Programming Guide, "Mathematical
// Functions"), which with the reference's own half ulp bounds the difference
// at about 1.5e-7 relative. The bound used here is nearly two orders looser,
// because nobody has run it: the figures are NVIDIA's for the library and this
// repository has measured PTX and not SASS. NUMERICS.md records that, and
// tightening it is work for the first person with a device.
//
// The probe kernel is inline rather than committed to kernels/, for the reason
// TestStructLayoutRoundTrip gives: it tests a rule, not a kernel anyone would
// launch.
func TestMathHelperParity(t *testing.T) {
	ctx := device(t)

	const probe = `package kernels

import "github.com/CWBudde/simtgo/gpu"

func MathProbe(ctx gpu.Ctx, out, a, b []float32) {
	i := ctx.GlobalID()
	n := len(a)
	if i >= n {
		return
	}
	x := a[i]
	y := b[i]
	out[i] = gpu.Sqrt(x)
	out[n+i] = gpu.Log(x)
	out[2*n+i] = gpu.Exp(y)
	out[3*n+i] = gpu.Sin(y)
	out[4*n+i] = gpu.Cos(y)
}
`
	src := fstest.MapFS{"probe.go": &fstest.MapFile{Data: []byte(probe)}}
	k, err := simt.Build(ctx, src, "MathProbe")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	// a is strictly positive and spans e**-8 to e**8, because Sqrt and Log
	// have nothing to say below zero; b is the linear span, which is where Sin
	// and Cos change sign repeatedly and where Exp reaches four digits.
	const n, block, parts = 1 << 12, 256, 5
	a := make([]float32, n)
	b := make([]float32, n)
	for i := range a {
		u := float64(i)/float64(n)*16 - 8
		a[i] = float32(math.Exp(u))
		b[i] = float32(u)
	}

	ref := make([]float32, parts*n)
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		ref[i] = float32(math.Sqrt(x))
		ref[n+i] = float32(math.Log(x))
		ref[2*n+i] = float32(math.Exp(y))
		ref[3*n+i] = float32(math.Sin(y))
		ref[4*n+i] = float32(math.Cos(y))
	}

	want := make([]float32, parts*n)
	gpu.RunCPU((n+block-1)/block, block, func(c gpu.Ctx) { mathProbe(c, want, a, b) })
	tolerance.AssertEqual(t, "cpu vs binary64", want, ref)

	dout, _ := cuda.NewSlice[float32](ctx, parts*n)
	da, _ := cuda.Upload(ctx, a)
	db, _ := cuda.Upload(ctx, b)
	defer dout.Free()
	defer da.Free()
	defer db.Free()

	if err := k.LaunchN(n, block, dout, da, db); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	got, err := dout.Download()
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	for at, name := range []string{"Sqrt", "Log", "Exp", "Sin", "Cos"} {
		lo, hi := at*n, (at+1)*n
		tolerance.AssertClose(t, "gpu "+name+" vs binary64", got[lo:hi], ref[lo:hi], 1e-5)
	}
}

// mathProbe is the Go the probe above transpiles from, so that the emulator
// runs the same function the device does. It cannot be the probe source
// itself: that source exists to be parsed, and Go has no way to run a string.
//
// The copy cannot drift silently, which is why it is tolerable. Both halves
// are compared against the same binary64 reference rather than against each
// other, so a change to either one alone fails rather than agreeing with
// itself.
func mathProbe(ctx gpu.Ctx, out, a, b []float32) {
	i := ctx.GlobalID()
	n := len(a)
	if i >= n {
		return
	}
	x := a[i]
	y := b[i]
	out[i] = gpu.Sqrt(x)
	out[n+i] = gpu.Log(x)
	out[2*n+i] = gpu.Exp(y)
	out[3*n+i] = gpu.Sin(y)
	out[4*n+i] = gpu.Cos(y)
}

// TestFminFmaxParity is the device half of the pair that was wrong until
// recently, and the cases that made it wrong are the cases it runs.
//
// The emulator side is pinned by TestFminFmaxFollowTheBuiltinTheyClaimToBe in
// package gpu, so this asks the other question: does fminf on the device do
// what gpu.Fmin was corrected to do? Every expectation here is written out
// from the rule rather than taken from gpu.Fmin, which would be the function
// under test standing in for its own reference.
//
// Every case is compared exactly, and NUMERICS.md says why it may be: fminf
// returns one of its operands, so there is nothing here that could have been
// rounded. NaN is the exception the tolerance rule cannot express at all --
// no bound relates a NaN to anything -- so the two NaN cases ask IsNaN, and
// the signed zeros ask for the sign bit, since == cannot tell them apart.
//
// The signed zeros are also the one case that could legitimately fail. NVIDIA
// documents the NaN behaviour of fminf and fmaxf and says nothing about which
// zero comes back, so the expectation below is gpu.Fmin's rule held against
// the device rather than the device's documented behaviour. If it fails, it is
// gpu.Fmin that should be revisited and NUMERICS.md that should record what
// the device actually did.
func TestFminFmaxParity(t *testing.T) {
	ctx := device(t)

	const probe = `package kernels

import "github.com/CWBudde/simtgo/gpu"

func MinMaxProbe(ctx gpu.Ctx, out, a, b []float32) {
	i := ctx.GlobalID()
	n := len(a)
	if i >= n {
		return
	}
	out[i] = gpu.Fmin(a[i], b[i])
	out[n+i] = gpu.Fmax(a[i], b[i])
}
`
	src := fstest.MapFS{"probe.go": &fstest.MapFile{Data: []byte(probe)}}
	k, err := simt.Build(ctx, src, "MinMaxProbe")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	nan := float32(math.NaN())
	inf := float32(math.Inf(1))
	negZero := float32(math.Copysign(0, -1))

	// The cases whose answer is an ordinary number, so == decides them.
	numeric := []struct {
		name           string
		x, y, min, max float32
	}{
		{"two positives", 2, 3, 2, 3},
		{"two positives, reversed", 3, 2, 2, 3},
		{"across zero", -1, 1, -1, 1},
		{"a NaN on the right", 1, nan, 1, 1},
		{"a NaN on the left", nan, 1, 1, 1},
		{"an infinity", inf, 1, 1, inf},
		{"a negative infinity", -inf, 1, -inf, 1},
	}
	// The cases == cannot decide, in the order they are indexed below.
	special := []struct{ x, y float32 }{
		{nan, nan},
		{negZero, 0},
		{0, negZero},
	}

	n := len(numeric) + len(special)
	a := make([]float32, n)
	b := make([]float32, n)
	for i, c := range numeric {
		a[i], b[i] = c.x, c.y
	}
	for i, c := range special {
		a[len(numeric)+i], b[len(numeric)+i] = c.x, c.y
	}

	dout, _ := cuda.NewSlice[float32](ctx, 2*n)
	da, _ := cuda.Upload(ctx, a)
	db, _ := cuda.Upload(ctx, b)
	defer dout.Free()
	defer da.Free()
	defer db.Free()

	if err := k.LaunchN(n, 32, dout, da, db); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	got, err := dout.Download()
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	mins, maxs := got[:n], got[n:]

	for i, c := range numeric {
		if mins[i] != c.min {
			t.Errorf("fminf(%v, %v) = %v, want %v (%s)", c.x, c.y, mins[i], c.min, c.name)
		}
		if maxs[i] != c.max {
			t.Errorf("fmaxf(%v, %v) = %v, want %v (%s)", c.x, c.y, maxs[i], c.max, c.name)
		}
	}

	// Both operands NaN is the one case with nothing to return but a NaN, and
	// which NaN is not something this repository has an opinion about.
	bothNaN := len(numeric)
	if !math.IsNaN(float64(mins[bothNaN])) {
		t.Errorf("fminf(NaN, NaN) = %v, want a NaN", mins[bothNaN])
	}
	if !math.IsNaN(float64(maxs[bothNaN])) {
		t.Errorf("fmaxf(NaN, NaN) = %v, want a NaN", maxs[bothNaN])
	}

	// -0 and +0 in both orders. fminf is expected to prefer the negative zero
	// and fmaxf the positive one, whichever way round they arrived.
	for i, c := range special[1:] {
		at := len(numeric) + 1 + i
		if !math.Signbit(float64(mins[at])) {
			t.Errorf("fminf(%v, %v) returned +0, want -0", c.x, c.y)
		}
		if math.Signbit(float64(maxs[at])) {
			t.Errorf("fmaxf(%v, %v) returned -0, want +0", c.x, c.y)
		}
	}
}

// TestMagnitudeFastParity is the fast-math kernel, and the one parity test in
// this file whose tolerance is not 1e-6.
//
// The band is 1e-5 rather than 1e-6 because --use_fast_math replaces
// sqrt.rn.f32 with sqrt.approx.f32 and div.rn.f32 with div.approx.f32, which
// CUDA documents to roughly 2 ULP each. The measured worst error on this
// device is one ULP -- the test logs it -- so the band is headroom over a
// documented bound, not a number the measurement demanded. NUMERICS.md records
// both figures and says which of them a test may rely on.
//
// The reference exists for the usual reason and earns its keep twice here: it
// is the only way to tell an approximate instruction apart from a wrong one.
func TestMagnitudeFastParity(t *testing.T) {
	ctx := device(t)
	const n, block = 1 << 14, 256
	const scale float32 = 3
	re, im := randomSignal(n), randomSignal(n)

	want := make([]float32, n)
	gpu.RunCPU((n+block-1)/block, block, func(c gpu.Ctx) { kernels.MagnitudeFast(c, want, re, im, scale) })

	// An independent reference, in float64 throughout so that the comparison
	// measures what the device did rather than what a float32 host did too.
	ref := make([]float32, n)
	for i := range ref {
		r, m := float64(re[i]), float64(im[i])
		ref[i] = float32(math.Sqrt(r*r+m*m) / float64(scale))
	}
	tolerance.AssertClose(t, "cpu vs reference", want, ref, 1e-6)

	k, err := simt.Build(ctx, simtgo.Kernels(), "MagnitudeFast")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	dm, _ := cuda.NewSlice[float32](ctx, n)
	dre, _ := cuda.Upload(ctx, re)
	dim, _ := cuda.Upload(ctx, im)
	defer dm.Free()
	defer dre.Free()
	defer dim.Free()

	if err := k.LaunchN(n, block, dm, dre, dim, scale); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	got, _ := dm.Download()
	tolerance.AssertClose(t, "gpu vs cpu", got, want, 1e-5)
	tolerance.AssertClose(t, "gpu vs reference", got, ref, 1e-5)

	// The measurement the tolerance above rests on. It is reported rather than
	// asserted at a tighter bound on purpose: the number is a property of this
	// driver on this architecture, and pinning it would turn a NUMERICS.md
	// finding into a test that fails on somebody else's card.
	var worst float64
	for i := range got {
		if d := math.Abs(float64(got[i]) - float64(ref[i])); d/math.Max(math.Abs(float64(ref[i])), 1) > worst {
			worst = d / math.Max(math.Abs(float64(ref[i])), 1)
		}
	}
	t.Logf("worst relative error against the float64 reference: %g", worst)
}

// TestFastMathChangesTheAnswer is the negative control for the directive, and
// the measurement NUMERICS.md's fast-math section rests on.
//
// Everything else about fast math can be verified without a device: the marker
// line, the hash split, the option string. None of that proves the flag
// reached the hardware. This builds one source twice, differing only in the
// directive, and compares what the device produced -- so a regression that
// quietly dropped the option would show up as two identical outputs.
//
// It asserts the *shape* of the difference rather than a number: that some
// outputs differ at all, and that none differs by more than the band the
// parity test above uses. The exact count is a property of this driver, this
// architecture and this input, so it is logged and not pinned.
func TestFastMathChangesTheAnswer(t *testing.T) {
	ctx := device(t)
	const n, block = 1 << 14, 256
	const d float32 = 3
	a, b := randomSignal(n), randomSignal(n)

	body := "func K(ctx gpu.Ctx, y, a, b []float32, d float32) {\n" +
		"\ti := ctx.GlobalID()\n" +
		"\tif i < len(y) {\n" +
		"\t\tr, m := a[i], b[i]\n" +
		"\t\ty[i] = gpu.Sqrt(r*r+m*m) / d\n" +
		"\t}\n}"

	run := func(fast bool) []float32 {
		src := "package kernels\n\nimport \"github.com/CWBudde/simtgo/gpu\"\n\n"
		if fast {
			src += "//simtgo:fastmath\n"
		}
		src += body + "\n"

		k, err := simt.Build(ctx, fstest.MapFS{"k.go": &fstest.MapFile{Data: []byte(src)}}, "K")
		if err != nil {
			t.Fatalf("Build(fastmath=%v): %v", fast, err)
		}
		dy, _ := cuda.NewSlice[float32](ctx, n)
		da, _ := cuda.Upload(ctx, a)
		db, _ := cuda.Upload(ctx, b)
		defer dy.Free()
		defer da.Free()
		defer db.Free()
		if err := k.LaunchN(n, block, dy, da, db, d); err != nil {
			t.Fatalf("Launch(fastmath=%v): %v", fast, err)
		}
		out, _ := dy.Download()
		return out
	}

	plain, fast := run(false), run(true)

	differing, worst := 0, 0.0
	for i := range plain {
		if plain[i] == fast[i] {
			continue
		}
		differing++
		if e := math.Abs(float64(plain[i])-float64(fast[i])) / math.Max(math.Abs(float64(plain[i])), 1); e > worst {
			worst = e
		}
	}
	t.Logf("%d of %d outputs differ (%.1f%%), worst relative gap %g", differing, n, 100*float64(differing)/float64(n), worst)

	if differing == 0 {
		t.Error("fast math changed nothing on the device; the option is not reaching NVRTC")
	}
	if worst > 1e-5 {
		t.Errorf("fast math moved a result by %g, beyond the 1e-5 the parity tests assert", worst)
	}
}

// TestBoundsChecksAgreeInRange is the claim WithBoundsChecks rests on: a
// correct kernel computes the same thing with the checks on.
//
// It is worth a device test rather than a golden. The untagged tests pin what
// is emitted and the NVRTC leg pins that it compiles, but neither asks whether
// the long long round trip through the helper changes an index -- and the
// subset allows four integer types as an index, all of which arrive at the
// helper by promotion. An off-by-one in the promotion would pass every
// untagged test and every compile.
//
// The build is also its own negative control for the prebuilt path: a debug
// build must miss the registry and go through NVRTC, and Kernel.Prebuilt is
// where that shows.
func TestBoundsChecksAgreeInRange(t *testing.T) {
	ctx := device(t)

	const probe = `package kernels

import "github.com/CWBudde/simtgo/gpu"

func BoundsProbe(ctx gpu.Ctx, y, x []float32) {
	tile := ctx.SharedF32(64)
	t := ctx.ThreadIdx()
	tile[t] = x[ctx.GlobalID()]
	ctx.SyncThreads()
	i := ctx.GlobalID()
	if i < len(y) {
		y[i] = tile[t] * 2
	}
}
`
	src := fstest.MapFS{"probe.go": &fstest.MapFile{Data: []byte(probe)}}

	const n = 64
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(i) * 0.5
	}
	want := make([]float32, n)
	for i, v := range x {
		want[i] = v * 2
	}

	run := func(t *testing.T, opts ...simt.BuildOption) ([]float32, *simt.Kernel) {
		t.Helper()
		k, err := simt.Build(ctx, src, "BoundsProbe", opts...)
		if err != nil {
			t.Fatalf("Build: %v", err)
		}
		dy, _ := cuda.NewSlice[float32](ctx, n)
		dx, _ := cuda.Upload(ctx, x)
		defer dy.Free()
		defer dx.Free()
		if err := k.LaunchN(n, 64, dy, dx); err != nil {
			t.Fatalf("Launch: %v", err)
		}
		got, err := dy.Download()
		if err != nil {
			t.Fatalf("Download: %v", err)
		}
		return got, k
	}

	release, rk := run(t)
	debug, dk := run(t, simt.WithBoundsChecks())

	tolerance.AssertEqual(t, "release vs reference", release, want)
	tolerance.AssertEqual(t, "debug vs reference", debug, want)
	tolerance.AssertEqual(t, "debug vs release", debug, release)

	if rk.Source == dk.Source {
		t.Error("the two builds generated the same C, so this compared one kernel with itself")
	}
	if dk.Prebuilt {
		t.Error("a bounds-checked build loaded a prebuilt artifact")
	}
}
