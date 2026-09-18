//go:build cuda

package simt_test

import (
	"errors"
	"testing"
	"testing/fstest"

	"github.com/CWBudde/gocuda"
	"github.com/CWBudde/gocuda/cuda"
	"github.com/CWBudde/gocuda/internal/tolerance"
	"github.com/CWBudde/gocuda/simt"
)

// The device half of the shared-memory work. Every reference here is an
// ordinary sequential Go loop rather than the same kernel under RunCPU, for
// the reason TestAtomicHistogramParity gives: a loop agrees with the device
// only if the device really did what the kernel says, whereas the emulator and
// the device could in principle share a misunderstanding. Integers keep the
// comparisons exact whatever order the threads arrived in.
//
// The probe kernels are inline rather than committed, for the reason
// TestStructLayoutRoundTrip gives: they test a rule, not a kernel anyone would
// launch. Histogram is the exception, and is committed, because privatising a
// histogram in shared memory is the thing itself rather than a rule about it.

// TestSharedProbesCompile puts the probe kernels above through NVRTC.
//
// Everything else in this file needs a device, and most machines that build
// this module have none, so without this the probes would be source nothing
// ever looked at -- and a probe that stopped lowering would look exactly like
// a test that was skipped. A toolkit is enough to answer that much.
func TestSharedProbesCompile(t *testing.T) {
	if _, _, err := cuda.NVRTCVersion(); errors.Is(err, cuda.ErrNoCUDA) {
		t.Skipf("no CUDA toolkit available: %v", err)
	}
	for _, tc := range []struct{ name, entry, src string }{
		{"DynProbe", "DynProbe", dynProbe},
		{"TypedProbe", "TypedProbe", typedProbe},
	} {
		t.Run(tc.name, func(t *testing.T) {
			src := fstest.MapFS{"probe.go": &fstest.MapFile{Data: []byte(tc.src)}}
			u, err := simt.Transpile(src, tc.entry)
			if err != nil {
				t.Fatalf("Transpile: %v", err)
			}
			if _, err := cuda.Compile(u.Source, tc.entry+".cu", "compute_75"); err != nil {
				t.Fatalf("NVRTC refused the generated CUDA: %v\n%s", err, u.Source)
			}
		})
	}
}

// histogram is the independent reference: an ordinary sequential count,
// written the way anybody would write it without a GPU.
func histogram(x []int32, bins int) []int32 {
	out := make([]int32, bins)
	for _, v := range x {
		if v >= 0 && int(v) < bins {
			out[v]++
		}
	}
	return out
}

// TestHistogramParity is the committed kernel: an int32 tile, atomics into it,
// a barrier, and one fold into global memory per block.
//
// It proves the two halves together. A tile that was not really per block would
// double-count across the grid, and an add that was not really atomic would
// lose updates inside a block; either shows up as a bin the loop above does not
// agree with.
func TestHistogramParity(t *testing.T) {
	ctx := device(t)
	k, err := simt.Build(ctx, gocuda.Kernels(), "Histogram")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if want := 4 * 256; k.SharedBytes != want {
		t.Errorf("SharedBytes = %d, want %d for a 256-element int32 tile", k.SharedBytes, want)
	}

	const n, bins = 1 << 15, 256
	x := make([]int32, n)
	for i := range x {
		// Deliberately uneven, and with values outside the range as well, so
		// that a bin nobody writes stays visibly zero and the kernel's choice
		// to drop out-of-range samples is exercised rather than assumed.
		x[i] = int32(i*i%311) - 8
	}

	// Uploaded zeros rather than NewSlice: the kernel only ever adds to these
	// bins, so an uninitialised allocation would be counted as part of the
	// histogram.
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
	tolerance.AssertEqual(t, "bins", got, histogram(x, bins))
}

// dynProbe privatises a histogram in shared memory, with the tile sized at
// the launch rather than in the source.
const dynProbe = `package kernels

import "github.com/CWBudde/gocuda/gpu"

func DynProbe(ctx gpu.Ctx, bins, x []int32) {
	tile := ctx.SharedDynI32()
	for k := ctx.ThreadIdx(); k < len(tile); k += ctx.BlockDim() {
		tile[k] = 0
	}
	ctx.SyncThreads()
	i := ctx.GlobalID()
	if i < len(x) {
		v := int(x[i])
		if v >= 0 && v < len(tile) {
			gpu.AtomicAddI32(tile, v, 1)
		}
	}
	ctx.SyncThreads()
	for k := ctx.ThreadIdx(); k < len(tile); k += ctx.BlockDim() {
		c := tile[k]
		if c != 0 && k < len(bins) {
			gpu.AtomicAddI32(bins, k, c)
		}
	}
}
`

// TestDynamicSharedTileParity is the launch-sized tile on hardware.
//
// Two things could go wrong invisibly and neither would be a compile error.
// The launch could pass too few bytes, in which case writes past the end of
// the block land in nobody's memory and bins go missing; or the generated
// length parameter could disagree with the bytes that were allocated, in
// which case the loops above walk off the tile. Both show up as a histogram
// that is not the loop's, which is why the reference is a loop.
func TestDynamicSharedTileParity(t *testing.T) {
	ctx := device(t)
	src := fstest.MapFS{"probe.go": &fstest.MapFile{Data: []byte(dynProbe)}}
	k, err := simt.Build(ctx, src, "DynProbe")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if k.DynSharedWidth != 4 {
		t.Fatalf("DynSharedWidth = %d, want 4 for an int32 tile", k.DynSharedWidth)
	}
	if k.SharedBytes != 0 {
		t.Errorf("SharedBytes = %d, want 0: the tile is not statically declared", k.SharedBytes)
	}

	const n, bins = 1 << 15, 192
	x := make([]int32, n)
	for i := range x {
		x[i] = int32(i*i%311) - 8
	}
	dbins, _ := cuda.Upload(ctx, make([]int32, bins))
	dx, _ := cuda.Upload(ctx, x)
	defer dbins.Free()
	defer dx.Free()

	if err := k.LaunchShared((n+255)/256, 256, bins, dbins, dx); err != nil {
		t.Fatalf("LaunchShared: %v", err)
	}
	got, err := dbins.Download()
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	tolerance.AssertEqual(t, "bins", got, histogram(x, bins))
}

// TestLaunchRefusesTheWrongSharedSpelling pins the two typed errors, which
// exist because both mistakes are otherwise silent: a kernel with a dynamic
// tile launched through Launch gets no bytes for it, and one without launched
// through LaunchShared is handed a size and an argument its signature has no
// parameter for.
func TestLaunchRefusesTheWrongSharedSpelling(t *testing.T) {
	ctx := device(t)
	src := fstest.MapFS{"probe.go": &fstest.MapFile{Data: []byte(dynProbe)}}
	dyn, err := simt.Build(ctx, src, "DynProbe")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	plain, err := simt.Build(ctx, gocuda.Kernels(), "VecAdd")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	dbins, _ := cuda.Upload(ctx, make([]int32, 8))
	dx, _ := cuda.Upload(ctx, make([]int32, 8))
	defer dbins.Free()
	defer dx.Free()

	var bad *simt.DynamicSharedError
	if err := dyn.Launch(1, 32, dbins, dx); !errors.As(err, &bad) {
		t.Fatalf("Launch of a kernel with a dynamic tile gave %v, want a DynamicSharedError", err)
	} else if !bad.Dynamic {
		t.Errorf("DynamicSharedError says the kernel has no dynamic tile, but it has one")
	}
	if err := plain.LaunchShared(1, 32, 8, dbins, dx); !errors.As(err, &bad) {
		t.Fatalf("LaunchShared of a kernel without a dynamic tile gave %v, want a DynamicSharedError", err)
	} else if bad.Dynamic {
		t.Errorf("DynamicSharedError says the kernel has a dynamic tile, but it has none")
	}

	// A tile larger than the device offers is refused here rather than by
	// cuLaunchKernel, which would answer with a generic code and neither
	// number. This is the first moment the total exists at all.
	limit := ctx.MaxSharedMemPerBlock()
	if limit <= 0 {
		t.Skip("the device did not report a shared memory limit")
	}
	var big *simt.SharedMemoryError
	if err := dyn.LaunchShared(1, 32, limit, dbins, dx); !errors.As(err, &big) {
		t.Fatalf("LaunchShared of %d int32 elements gave %v, want a SharedMemoryError", limit, err)
	}
}

// typedProbe stages through a tile per element type and through a device
// function that has one of its own, which is the construct A3 made legal.
//
// The ragged tail is guarded rather than returned from, and the guard covers
// the store rather than the call, and neither is a style preference. An early
// return before a barrier is a thread that never arrives at it, which is
// undefined on the device and, until the block barrier learned to be left, an
// unkillable hang under RunCPU. This probe had that shape and survived only
// because the test below launches a geometry in which the guard is never true.
// Turning the return into a guard moved the problem rather than removing it:
// staged() barriers inside, so a call to it under the guard is the same thread
// missing the same rendezvous, one level down -- which is what the
// barrier-divergence check now refuses and what made it visible. Hoisting the
// call keeps every thread on the same path through both barriers, which is
// what the kernels in kernels/ do too.
const typedProbe = `package kernels

import "github.com/CWBudde/gocuda/gpu"

//gocuda:device
func staged(ctx gpu.Ctx, v int64) int64 {
	scratch := ctx.SharedI64(64)
	scratch[ctx.ThreadIdx()%64] = v * 2
	ctx.SyncThreads()
	return scratch[ctx.ThreadIdx()%64]
}

func TypedProbe(ctx gpu.Ctx, out []int64, x []int32) {
	wide := ctx.SharedI64(64)
	narrow := ctx.SharedU32(64)
	t := ctx.ThreadIdx() % 64
	i := ctx.GlobalID()
	if i < len(out) {
		wide[t] = int64(x[i])
		narrow[t] = uint32(x[i])
	}
	ctx.SyncThreads()
	s := staged(ctx, wide[t])
	if i < len(out) {
		out[i] = s + int64(narrow[t])
	}
}
`

// TestTypedSharedTileParity is the element types on hardware, and the tile
// inside a __device__ function with them.
//
// The widths are the point: an int64 tile read back as 32 bits, or a device
// function whose tile aliased the kernel's, would return numbers this loop
// does not produce. The values are chosen so that the two contributions cannot
// be confused -- 2*v and v are different for every v but zero.
func TestTypedSharedTileParity(t *testing.T) {
	ctx := device(t)
	src := fstest.MapFS{"probe.go": &fstest.MapFile{Data: []byte(typedProbe)}}
	k, err := simt.Build(ctx, src, "TypedProbe")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	// 64 int64 for the kernel's tile, 64 uint32 for its second, and 64 more
	// int64 for the device function's -- counted once, although two calls
	// reach it.
	if want := 64*8 + 64*4 + 64*8; k.SharedBytes != want {
		t.Errorf("SharedBytes = %d, want %d", k.SharedBytes, want)
	}

	const n, block = 1 << 12, 64
	x := make([]int32, n)
	for i := range x {
		// Large enough that the high word of the int64 matters: a tile read
		// back at 32 bits would truncate it.
		x[i] = int32(i)*7 + 1000000
	}
	want := make([]int64, n)
	for i, v := range x {
		want[i] = int64(v)*2 + int64(uint32(v))
	}

	dout, _ := cuda.NewSlice[int64](ctx, n)
	dx, _ := cuda.Upload(ctx, x)
	defer dout.Free()
	defer dx.Free()

	if err := k.LaunchN(n, block, dout, dx); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	got, err := dout.Download()
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	tolerance.AssertEqual(t, "out", got, want)
}
