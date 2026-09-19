//go:build cuda

package simt_test

import (
	"testing"
	"testing/fstest"

	"github.com/CWBudde/simtgo"
	"github.com/CWBudde/simtgo/cuda"
	"github.com/CWBudde/simtgo/gpu"
	"github.com/CWBudde/simtgo/internal/tolerance"
	"github.com/CWBudde/simtgo/kernels"
	"github.com/CWBudde/simtgo/simt"
)

// The warp primitives are the one part of the vocabulary where the CPU
// emulator is not a model of the device but a stand-in for it: a device warp
// is in lockstep and the emulator's is a rendezvous between goroutines. Both
// halves are therefore compared against a sequential Go reference written from
// the definition rather than against each other, because agreeing with an
// emulation of yourself is not evidence.

// TestWarpReduceSumParity is the committed kernel.
//
// The reference sums each warp's stretch of the input in a plain loop, which
// is not the kernel's order: the kernel halves the distance five times, and
// that is a different association of the same float additions. Small integers
// are what let this assert equality anyway -- every partial sum is exact in a
// float32 regardless of the order -- which is the same move the atomic
// histogram makes and for the same reason.
func TestWarpReduceSumParity(t *testing.T) {
	ctx := device(t)
	const block = kernels.WarpReduceBlock
	// Deliberately not a multiple of the block size: the threads past the end
	// of the input still have to reach every exchange, contributing zero.
	const n = 37*block + 53
	grid := (n + block - 1) / block
	warps := grid * block / gpu.WarpSize

	x := make([]float32, n)
	for i := range x {
		x[i] = float32(i%17 - 8)
	}

	// The independent reference: warp w covers global threads
	// [32w, 32w+32), and a thread past the end of x contributes nothing.
	want := make([]float32, warps)
	for w := range warps {
		var sum float32
		for k := w * gpu.WarpSize; k < (w+1)*gpu.WarpSize; k++ {
			if k < n {
				sum += x[k]
			}
		}
		want[w] = sum
	}

	cpu := make([]float32, warps)
	gpu.RunCPU(grid, block, func(c gpu.Ctx) { kernels.WarpReduceSum(c, cpu, x) })
	tolerance.AssertEqual(t, "cpu sums", cpu, want)

	k, err := simt.Build(ctx, simtgo.Kernels(), "WarpReduceSum")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	dout, _ := cuda.NewSlice[float32](ctx, warps)
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
	tolerance.AssertEqual(t, "sums", got, want)
}

// TestWarpVocabularyParity is the rest of the vocabulary, as a probe kernel
// rather than a committed one: it tests the rules, not a kernel anybody would
// launch, and committing one would cost a gate entry, a golden and a
// regeneration to say the same thing (see TestStructLayoutRoundTrip).
//
// The launch is a whole number of warps on purpose. In a short warp the full
// participation mask the emitter writes names lanes that do not exist, which
// CUDA leaves undefined, and a parity test over undefined behaviour would be
// asserting that two implementations happen to agree today.
func TestWarpVocabularyParity(t *testing.T) {
	ctx := device(t)

	const probe = `package kernels

import "github.com/CWBudde/simtgo/gpu"

func WarpProbe(ctx gpu.Ctx, bcast, up []float32, mask, vote []int32, x []float32) {
	i := ctx.GlobalID()
	v := x[i]
	bcast[i] = ctx.ShuffleF32(v, 0)
	up[i] = ctx.ShuffleUpF32(v, 1)
	mask[i] = int32(ctx.Ballot(v > 0))
	n := int32(0)
	if ctx.Any(v > 2) {
		n += 1
	}
	if ctx.All(v > -2) {
		n += 2
	}
	if ctx.ActiveMask() == 0xffffffff {
		n += 4
	}
	vote[i] = n
}
`
	src := fstest.MapFS{"probe.go": &fstest.MapFile{Data: []byte(probe)}}

	const grid, block = 5, 64
	const n = grid * block
	x := randomSignal(n)

	// The sequential reference, written from the definition of a warp: 32
	// consecutive threads of a block, and here a block is two whole warps.
	bcast := make([]float32, n)
	up := make([]float32, n)
	mask := make([]int32, n)
	vote := make([]int32, n)
	for w := range n / gpu.WarpSize {
		first := w * gpu.WarpSize
		var m uint32
		hot, bounded := false, true
		for lane := range gpu.WarpSize {
			v := x[first+lane]
			if v > 0 {
				m |= 1 << uint(lane)
			}
			hot = hot || v > 2
			bounded = bounded && v > -2
		}
		for lane := range gpu.WarpSize {
			i := first + lane
			bcast[i] = x[first]
			up[i] = x[i]
			if lane > 0 {
				up[i] = x[i-1]
			}
			mask[i] = int32(m)
			vote[i] = 4 // every lane of a full warp is active
			if hot {
				vote[i]++
			}
			if bounded {
				vote[i] += 2
			}
		}
	}

	k, err := simt.Build(ctx, src, "WarpProbe")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	dbcast, _ := cuda.NewSlice[float32](ctx, n)
	dup, _ := cuda.NewSlice[float32](ctx, n)
	dmask, _ := cuda.NewSlice[int32](ctx, n)
	dvote, _ := cuda.NewSlice[int32](ctx, n)
	dx, _ := cuda.Upload(ctx, x)
	defer dbcast.Free()
	defer dup.Free()
	defer dmask.Free()
	defer dvote.Free()
	defer dx.Free()

	if err := k.LaunchN(n, block, dbcast, dup, dmask, dvote, dx); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	gb, _ := dbcast.Download()
	gu, _ := dup.Download()
	gm, _ := dmask.Download()
	gv, _ := dvote.Download()
	// Exact: every one of these moves a value or a bit, and none of them does
	// any arithmetic that could round.
	tolerance.AssertEqual(t, "broadcast", gb, bcast)
	tolerance.AssertEqual(t, "shuffle up", gu, up)
	tolerance.AssertEqual(t, "ballot", gm, mask)
	tolerance.AssertEqual(t, "votes", gv, vote)
}
