package gpu_test

import (
	"testing"
	"time"

	"github.com/CWBudde/gocuda/gpu"
)

// The warp primitives are a rendezvous between goroutines, so these tests ask
// two separate questions of each one: does it exchange the right values, and
// does it terminate. The second is the reason the machinery exists at all --
// the device has no rendezvous, its lanes are already in lockstep -- and a
// hang is the one failure mode a test suite cannot report.

// TestShuffleBroadcasts is the simplest exchange there is: every lane reads
// lane 0. It fails unless the values really crossed between goroutines.
func TestShuffleBroadcasts(t *testing.T) {
	const grid, block = 4, 64
	got := make([]float32, grid*block)
	gpu.RunCPU(grid, block, func(ctx gpu.Ctx) {
		i := ctx.GlobalID()
		got[i] = ctx.ShuffleF32(float32(i), 0)
	})
	for i, v := range got {
		// Lane 0 of the warp i belongs to, in global numbering.
		want := float32(i - i%gpu.WarpSize)
		if v != want {
			t.Fatalf("thread %d read %v from lane 0, want %v", i, v, want)
		}
	}
}

// TestShuffleWrapsTheSourceLane pins the one case where a source lane outside
// the warp is defined rather than merely definite: __shfl_sync takes it modulo
// the warp size.
func TestShuffleWrapsTheSourceLane(t *testing.T) {
	const block = 32
	got := make([]int32, block)
	gpu.RunCPU(1, block, func(ctx gpu.Ctx) {
		got[ctx.ThreadIdx()] = ctx.ShuffleI32(int32(ctx.LaneID()), gpu.WarpSize+3)
	})
	for i, v := range got {
		if v != 3 {
			t.Fatalf("lane %d read lane %d, want lane 3: srcLane is taken modulo the warp size", i, v)
		}
	}
}

// TestShuffleDownOffTheEndKeepsOwnValue is the property the reduction loop
// depends on: the top delta lanes have no neighbour above them and get their
// own value back, so the loop needs no special case for them.
func TestShuffleDownOffTheEndKeepsOwnValue(t *testing.T) {
	const block = 32
	const delta = 8
	got := make([]float32, block)
	gpu.RunCPU(1, block, func(ctx gpu.Ctx) {
		got[ctx.ThreadIdx()] = ctx.ShuffleDownF32(float32(ctx.LaneID()), delta)
	})
	for lane, v := range got {
		want := float32(lane + delta)
		if lane+delta >= gpu.WarpSize {
			want = float32(lane)
		}
		if v != want {
			t.Fatalf("lane %d read %v, want %v", lane, v, want)
		}
	}
	// The mirror image, for the lanes at the bottom.
	gpu.RunCPU(1, block, func(ctx gpu.Ctx) {
		got[ctx.ThreadIdx()] = ctx.ShuffleUpF32(float32(ctx.LaneID()), delta)
	})
	for lane, v := range got {
		want := float32(lane - delta)
		if lane-delta < 0 {
			want = float32(lane)
		}
		if v != want {
			t.Fatalf("lane %d read %v from below, want %v", lane, v, want)
		}
	}
}

// TestButterflyReductionSumsTheWarp exercises the exchange every lane takes
// part in, both ways round: the xor butterfly leaves the total in every lane,
// the halving shuffle-down leaves it in lane 0.
func TestButterflyReductionSumsTheWarp(t *testing.T) {
	const grid, block = 2, 64
	xor := make([]float32, grid*block)
	down := make([]float32, grid*block)
	gpu.RunCPU(grid, block, func(ctx gpu.Ctx) {
		i := ctx.GlobalID()
		a := float32(i)
		for m := 1; m < gpu.WarpSize; m *= 2 {
			a += ctx.ShuffleXorF32(a, m)
		}
		xor[i] = a

		b := float32(i)
		for off := gpu.WarpSize / 2; off > 0; off /= 2 {
			b += ctx.ShuffleDownF32(b, off)
		}
		down[i] = b
	})
	for i := range xor {
		first := i - i%gpu.WarpSize
		var want float32
		for k := first; k < first+gpu.WarpSize; k++ {
			want += float32(k)
		}
		if xor[i] != want {
			t.Fatalf("thread %d: butterfly sum %v, want %v", i, xor[i], want)
		}
		if i%gpu.WarpSize == 0 && down[i] != want {
			t.Fatalf("thread %d: halving sum %v, want %v", i, down[i], want)
		}
	}
}

// TestVotes covers the three that read a predicate rather than a value.
func TestVotes(t *testing.T) {
	const block = 32
	ballot := make([]uint32, block)
	any := make([]bool, block)
	all := make([]bool, block)
	active := make([]uint32, block)
	gpu.RunCPU(1, block, func(ctx gpu.Ctx) {
		lane := ctx.LaneID()
		ballot[lane] = ctx.Ballot(lane%2 == 0)
		any[lane] = ctx.Any(lane == 7)
		all[lane] = ctx.All(lane < gpu.WarpSize)
		active[lane] = ctx.ActiveMask()
	})
	for lane := range block {
		if ballot[lane] != 0x55555555 {
			t.Fatalf("lane %d: ballot %#x, want %#x", lane, ballot[lane], uint32(0x55555555))
		}
		if !any[lane] {
			t.Fatalf("lane %d: Any over a predicate one lane passes was false", lane)
		}
		if !all[lane] {
			t.Fatalf("lane %d: All over a predicate every lane passes was false", lane)
		}
		if active[lane] != 0xffffffff {
			t.Fatalf("lane %d: ActiveMask %#x, want every lane of a full warp", lane, active[lane])
		}
	}

	// All is a vote, not a broadcast: one dissenting lane turns it false
	// everywhere.
	gpu.RunCPU(1, block, func(ctx gpu.Ctx) {
		all[ctx.LaneID()] = ctx.All(ctx.LaneID() != 13)
	})
	for lane := range block {
		if all[lane] {
			t.Fatalf("lane %d: All was true although lane 13 voted against", lane)
		}
	}
}

// TestTailWarp is the block that is not a multiple of the warp size: 48
// threads are a warp of 32 and a warp of 16.
//
// What the short warp does here is the emulator's answer and not the device's
// -- see gpu/warp.go -- which is exactly why it is pinned: a change in it is a
// change in what the two backends disagree about.
func TestTailWarp(t *testing.T) {
	const block = 48
	active := make([]uint32, block)
	read := make([]float32, block)
	gpu.RunCPU(1, block, func(ctx gpu.Ctx) {
		i := ctx.ThreadIdx()
		active[i] = ctx.ActiveMask()
		// Lane 20 exists in the first warp and not in the second.
		read[i] = ctx.ShuffleF32(float32(i), 20)
	})
	for i := range block {
		wantMask := uint32(0xffffffff)
		wantRead := float32(i - i%gpu.WarpSize + 20)
		if i >= gpu.WarpSize {
			wantMask = 0xffff     // sixteen lanes, and no more
			wantRead = float32(i) // lane 20 is not there: the caller's own value
		}
		if active[i] != wantMask {
			t.Fatalf("thread %d: ActiveMask %#x, want %#x", i, active[i], wantMask)
		}
		if read[i] != wantRead {
			t.Fatalf("thread %d: read %v from lane 20, want %v", i, read[i], wantRead)
		}
	}
}

// TestBlockSmallerThanAWarp is the other end of the same question. Eight
// threads are one warp of eight, and every vote is over those eight.
func TestBlockSmallerThanAWarp(t *testing.T) {
	const block = 8
	active := make([]uint32, block)
	sum := make([]float32, block)
	gpu.RunCPU(1, block, func(ctx gpu.Ctx) {
		lane := ctx.LaneID()
		active[lane] = ctx.ActiveMask()
		v := float32(lane)
		for m := 1; m < block; m *= 2 {
			v += ctx.ShuffleXorF32(v, m)
		}
		sum[lane] = v
	})
	for lane := range block {
		if active[lane] != 0xff {
			t.Fatalf("lane %d: ActiveMask %#x, want eight lanes", lane, active[lane])
		}
		if sum[lane] != 28 { // 0+1+...+7
			t.Fatalf("lane %d: butterfly sum %v, want 28", lane, sum[lane])
		}
	}
}

// TestLaneIDFollowsTheFlatIndex pins the definition against the obvious wrong
// one. A 16x4 block is 64 threads and two warps, and threadIdx.x alone would
// make every row lane 0 to 15 of warp 0.
func TestLaneIDFollowsTheFlatIndex(t *testing.T) {
	block := gpu.D2(16, 4)
	got := make([]int, 64)
	gpu.RunCPUDim(gpu.D1(1), block, func(ctx gpu.Ctx) {
		flat := ctx.ThreadIdx() + 16*ctx.ThreadIdxY()
		got[flat] = ctx.LaneID()
	})
	for flat, lane := range got {
		if lane != flat%gpu.WarpSize {
			t.Fatalf("flat thread %d is lane %d, want %d", flat, lane, flat%gpu.WarpSize)
		}
	}
}

// TestWarpOutsideALaunch calls the vocabulary on a zero-value Ctx, which is
// what a kernel body called as a plain Go function gets. A warp of one is the
// answer, because the alternative is that a kernel cannot be run as ordinary
// Go -- and running it as ordinary Go is how its reference implementation is
// usually written.
func TestWarpOutsideALaunch(t *testing.T) {
	var ctx gpu.Ctx
	if got := ctx.LaneID(); got != 0 {
		t.Errorf("LaneID = %d, want 0", got)
	}
	if got := ctx.ShuffleF32(2.5, 7); got != 2.5 {
		t.Errorf("ShuffleF32 = %v, want the caller's own value", got)
	}
	if got := ctx.ShuffleDownI32(9, 1); got != 9 {
		t.Errorf("ShuffleDownI32 = %v, want the caller's own value", got)
	}
	if got := ctx.Ballot(true); got != 1 {
		t.Errorf("Ballot(true) = %#x, want just this lane", got)
	}
	if got := ctx.Ballot(false); got != 0 {
		t.Errorf("Ballot(false) = %#x, want nothing", got)
	}
	if !ctx.Any(true) || ctx.Any(false) || !ctx.All(true) || ctx.All(false) {
		t.Error("Any and All of one thread must be that thread's predicate")
	}
	if got := ctx.ActiveMask(); got != 1 {
		t.Errorf("ActiveMask = %#x, want just this lane", got)
	}
	ctx.SyncWarp() // must not block, and must not panic
}

// TestDivergenceThatExitsDoesNotHang is the ordinary shape of divergence, and
// the one CUDA itself permits: half the warp leaves the kernel without
// reaching the rendezvous the other half is waiting in.
//
// A plain barrier would wait for those threads for ever. The release condition
// counts threads that have left as having arrived, so the waiters get through
// -- and this test returning at all is the assertion. What they see is the
// mask of who was actually there, which is what makes it observable.
func TestDivergenceThatExitsDoesNotHang(t *testing.T) {
	const block = 32
	mask := make([]uint32, block)
	done := make(chan struct{})
	go func() {
		defer close(done)
		gpu.RunCPU(1, block, func(ctx gpu.Ctx) {
			lane := ctx.LaneID()
			if lane >= 16 {
				return // leaves without taking part
			}
			mask[lane] = ctx.ActiveMask()
		})
	}()
	select {
	case <-done:
	case <-time.After(30 * time.Second):
		t.Fatal("a warp whose other half returned early never released")
	}
	for lane := range 16 {
		if mask[lane] != 0xffff {
			t.Fatalf("lane %d saw %#x taking part, want the sixteen that stayed", lane, mask[lane])
		}
	}
}

// TestWarpPanicDoesNotStrandTheWarp is the warp's version of
// TestAtomicPanicDoesNotStrandTheLock, and its assertion is the same one:
// returning at all.
//
// A thread that panics never reaches the rendezvous its siblings are waiting
// in. Without the abandon in RunCPU's per-thread recover they would wait for
// it for ever, and the deadlock would bury the panic that caused it -- the
// test binary would hang with no diagnosis rather than fail with one.
func TestWarpPanicDoesNotStrandTheWarp(t *testing.T) {
	x := make([]float32, 4)
	msg := mustPanic(t, func() {
		gpu.RunCPU(1, 32, func(ctx gpu.Ctx) {
			lane := ctx.LaneID()
			if lane == 0 {
				x[999] = 1 // panics, on this thread's goroutine
			}
			ctx.ShuffleF32(float32(lane), 0)
		})
	})
	wantContains(t, msg, "index out of range")

	// And the machinery is still usable afterwards: nothing global was left
	// holding a lock or a generation.
	done := make(chan struct{})
	go func() {
		defer close(done)
		gpu.RunCPU(2, 32, func(ctx gpu.Ctx) { ctx.SyncWarp() })
	}()
	select {
	case <-done:
	case <-time.After(30 * time.Second):
		t.Fatal("a later launch using warp primitives blocked after an earlier kernel panicked in one")
	}
}

// TestLanesMustMeetInTheSameCall covers the other way threads can disagree
// about a rendezvous: they all arrive, but not at the same primitive.
//
// Lanes are matched by the order they arrive in, exactly as SharedF32 calls
// are, so a warp half of which is voting while the other half is
// synchronising has exchanged nothing meaningful. It is diagnosed rather than
// answered, for the same reason a SharedF32 size mismatch is.
func TestLanesMustMeetInTheSameCall(t *testing.T) {
	msg := mustPanic(t, func() {
		gpu.RunCPU(1, 32, func(ctx gpu.Ctx) {
			if ctx.LaneID() < 16 {
				ctx.SyncWarp()
				return
			}
			ctx.Ballot(true)
		})
	})
	wantContains(t, msg, "met in different warp-level calls", "SyncWarp", "Ballot")
}
