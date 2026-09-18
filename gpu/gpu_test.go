package gpu_test

import (
	"fmt"
	"math"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/CWBudde/gocuda/gpu"
)

// mustPanic runs fn and returns the message it panicked with, failing the test
// if it did not panic. The emulator deliberately raises every diagnosis on the
// goroutine that called RunCPU precisely so that this is possible: a panic on
// a thread goroutine could not be recovered here and would take the test
// binary down with it.
func mustPanic(t *testing.T, fn func()) string {
	t.Helper()
	var msg string
	var panicked bool
	func() {
		defer func() {
			r := recover()
			if r == nil {
				return
			}
			panicked = true
			s, ok := r.(string)
			if !ok {
				t.Fatalf("panicked with %T (%v), want a string diagnosis", r, r)
			}
			msg = s
		}()
		fn()
	}()
	if !panicked {
		t.Fatal("did not panic, want a diagnosis")
	}
	if !strings.HasPrefix(msg, "gpu: ") {
		t.Errorf("panic message %q is not prefixed with \"gpu: \"", msg)
	}
	return msg
}

// wantContains asserts that the diagnosis names the things a kernel author
// needs in order to find the offending line.
func wantContains(t *testing.T, msg string, parts ...string) {
	t.Helper()
	for _, p := range parts {
		if !strings.Contains(msg, p) {
			t.Errorf("panic message does not mention %q:\n%s", p, msg)
		}
	}
}

// TestSharedAndBarrier reverses each block through shared memory, which only
// produces the right answer if SharedF32 is genuinely shared per block and
// SyncThreads is a genuine barrier.
func TestSharedAndBarrier(t *testing.T) {
	const grid, block = 8, 32
	n := grid * block
	in := make([]float32, n)
	out := make([]float32, n)
	for i := range in {
		in[i] = float32(i)
	}

	gpu.RunCPU(grid, block, func(ctx gpu.Ctx) {
		s := ctx.SharedF32(block)
		i := ctx.GlobalID()
		s[ctx.ThreadIdx()] = in[i]
		ctx.SyncThreads()
		out[i] = s[ctx.BlockDim()-1-ctx.ThreadIdx()]
	})

	for b := range grid {
		for th := range block {
			i := b*block + th
			want := float32(b*block + (block - 1 - th))
			if out[i] != want {
				t.Fatalf("out[%d] = %v, want %v", i, out[i], want)
			}
		}
	}
}

// TestGridCoordinates checks that every thread of the grid runs exactly once
// with a distinct global id.
func TestGridCoordinates(t *testing.T) {
	const grid, block = 17, 13
	seen := make([]int32, grid*block)
	gpu.RunCPU(grid, block, func(ctx gpu.Ctx) {
		if ctx.GridDim() != grid || ctx.BlockDim() != block {
			t.Errorf("dims = %d/%d, want %d/%d", ctx.GridDim(), ctx.BlockDim(), grid, block)
		}
		seen[ctx.GlobalID()]++
	})
	for i, c := range seen {
		if c != 1 {
			t.Fatalf("thread %d ran %d times, want 1", i, c)
		}
	}
}

// TestTwoSharedBuffers checks that successive SharedF32 calls hand out
// distinct per-block buffers, matched across threads by call order.
func TestTwoSharedBuffers(t *testing.T) {
	const grid, block = 4, 16
	out := make([]float32, grid*block)
	gpu.RunCPU(grid, block, func(ctx gpu.Ctx) {
		a := ctx.SharedF32(block)
		b := ctx.SharedF32(block)
		a[ctx.ThreadIdx()] = 1
		b[ctx.ThreadIdx()] = 2
		ctx.SyncThreads()
		out[ctx.GlobalID()] = a[ctx.ThreadIdx()] + b[ctx.ThreadIdx()]
	})
	for i, v := range out {
		if v != 3 {
			t.Fatalf("out[%d] = %v, want 3", i, v)
		}
	}
}

// TestSharedF32SizeMismatch proves that threads disagreeing about the size of
// the same shared buffer are diagnosed instead of silently sharing whatever
// the first thread to arrive happened to allocate.
func TestSharedF32SizeMismatch(t *testing.T) {
	msg := mustPanic(t, func() {
		gpu.RunCPU(1, 8, func(ctx gpu.Ctx) {
			if ctx.ThreadIdx()%2 == 0 {
				_ = ctx.SharedF32(8)
			} else {
				_ = ctx.SharedF32(5)
			}
		})
	})
	wantContains(t, msg, "SharedF32 size mismatch", "call #0", "block 0")
	// Whichever thread got there first, both requested sizes and the id of the
	// thread that disagreed have to be in the message.
	if !strings.Contains(msg, "asked for 8") || !strings.Contains(msg, "asked for 5") {
		t.Errorf("panic message does not name both sizes:\n%s", msg)
	}
	// Which thread is the one to notice depends on who won the race for the
	// first allocation, so accept any of the block's threads by id.
	named := false
	for id := range 8 {
		if strings.Contains(msg, fmt.Sprintf("thread %d asked for", id)) {
			named = true
		}
	}
	if !named {
		t.Errorf("panic message does not name the offending thread:\n%s", msg)
	}
}

// TestDivergentSharedF32 proves that threads reaching a different number of
// SharedF32 calls are diagnosed. Only even thread ids take the second call, so
// from that call on the odd threads would be handed the buffer of somebody
// else's call index.
func TestDivergentSharedF32(t *testing.T) {
	msg := mustPanic(t, func() {
		gpu.RunCPU(1, 8, func(ctx gpu.Ctx) {
			_ = ctx.SharedF32(8)
			if ctx.ThreadIdx()%2 == 0 {
				_ = ctx.SharedF32(8)
			}
		})
	})
	wantContains(t, msg,
		"divergent shared tile usage",
		"block 0",
		"thread 0 made 2 shared tile call(s), thread 1 made 1")
}

// TestAssumeBlockDimMismatch proves that a kernel written for a fixed block
// size refuses a launch with a different one.
func TestAssumeBlockDimMismatch(t *testing.T) {
	msg := mustPanic(t, func() {
		gpu.RunCPU(4, 8, func(ctx gpu.Ctx) {
			ctx.AssumeBlockDim(256)
		})
	})
	wantContains(t, msg, "block size mismatch", "AssumeBlockDim(256)", "blockDim=8")
}

// TestAssumeBlockDimAccepts covers the two cases that must stay quiet: the
// declared size matching the launch, and a zero-value Ctx, which belongs to no
// launch and therefore has nothing to check — the same no-op SyncThreads is
// there.
func TestAssumeBlockDimAccepts(t *testing.T) {
	const grid, block = 3, 16
	var ran int32
	gpu.RunCPU(grid, block, func(ctx gpu.Ctx) {
		ctx.AssumeBlockDim(block)
		if ctx.GlobalID() == 0 {
			ran = 1
		}
	})
	if ran != 1 {
		t.Fatal("kernel did not run")
	}

	var ctx gpu.Ctx
	ctx.AssumeBlockDim(block) // must not panic outside a launch
	ctx.SyncThreads()
	if got := len(ctx.SharedF32(4)); got != 4 {
		t.Fatalf("len(SharedF32(4)) = %d, want 4 outside a launch", got)
	}
}

// TestDifferentSizesAtDifferentCalls is the negative control for the size
// check: it keys on the call index, so successive calls of a block are free to
// ask for different sizes as long as all threads agree per call.
func TestDifferentSizesAtDifferentCalls(t *testing.T) {
	const grid, block = 4, 16
	out := make([]float32, grid*block)
	gpu.RunCPU(grid, block, func(ctx gpu.Ctx) {
		ctx.AssumeBlockDim(block)
		a := ctx.SharedF32(block)
		b := ctx.SharedF32(2 * block)
		if len(a) != block || len(b) != 2*block {
			t.Errorf("shared lengths = %d/%d, want %d/%d", len(a), len(b), block, 2*block)
		}
		a[ctx.ThreadIdx()] = 1
		b[block+ctx.ThreadIdx()] = 2
		ctx.SyncThreads()
		out[ctx.GlobalID()] = a[ctx.ThreadIdx()] + b[block+ctx.ThreadIdx()]
	})
	for i, v := range out {
		if v != 3 {
			t.Fatalf("out[%d] = %v, want 3", i, v)
		}
	}
}

// TestKernelPanicSurfaces proves that a panic raised by the kernel itself
// reaches the caller instead of killing the process, and that the threads left
// behind still get through their barriers rather than deadlocking.
func TestKernelPanicSurfaces(t *testing.T) {
	msg := mustPanic(t, func() {
		gpu.RunCPU(1, 8, func(ctx gpu.Ctx) {
			if ctx.ThreadIdx() == 3 {
				panic("boom")
			}
			ctx.SyncThreads()
			ctx.SyncThreads()
		})
	})
	wantContains(t, msg, "kernel panicked in block 0, thread 3", "boom")
}

// TestRunCPUDim covers a genuinely multi-dimensional launch: every thread of
// the grid must run exactly once, and each must see its own coordinates.
func TestRunCPUDim(t *testing.T) {
	const gx, gy, bx, by = 3, 2, 4, 5
	const w, h = gx * bx, gy * by

	// One counter per global position. Blocks run concurrently, so the visits
	// are counted atomically and `go test -race` has something to say if the
	// emulator ever hands two threads the same coordinates.
	visits := make([]int32, w*h)
	var bad atomic.Int32
	gpu.RunCPUDim(gpu.D2(gx, gy), gpu.D2(bx, by), func(c gpu.Ctx) {
		if c.BlockDim() != bx || c.BlockDimY() != by || c.GridDim() != gx || c.GridDimY() != gy {
			bad.Add(1)
			return
		}
		if c.BlockDimZ() != 1 || c.GridDimZ() != 1 || c.ThreadIdxZ() != 0 || c.BlockIdxZ() != 0 {
			bad.Add(1)
			return
		}
		x, y := c.GlobalIDX(), c.GlobalIDY()
		if x != c.BlockIdx()*bx+c.ThreadIdx() || y != c.BlockIdxY()*by+c.ThreadIdxY() {
			bad.Add(1)
			return
		}
		if c.GlobalIDZ() != 0 {
			bad.Add(1)
			return
		}
		atomic.AddInt32(&visits[y*w+x], 1)
	})
	if n := bad.Load(); n != 0 {
		t.Fatalf("%d threads disagreed with their own coordinates", n)
	}
	for i, n := range visits {
		if n != 1 {
			t.Fatalf("global position %d ran %d times, want 1", i, n)
		}
	}
}

// TestRunCPUIsOneDimensional keeps the 1-D entry point meaning what it always
// did, now that it is a wrapper.
func TestRunCPUIsOneDimensional(t *testing.T) {
	var seen atomic.Int32
	gpu.RunCPU(3, 4, func(c gpu.Ctx) {
		if c.GridDimY() == 1 && c.BlockDimY() == 1 && c.GlobalIDY() == 0 && c.GlobalID() == c.GlobalIDX() {
			seen.Add(1)
		}
	})
	if got := seen.Load(); got != 12 {
		t.Errorf("%d of 12 threads saw a one-dimensional launch", got)
	}
}

// TestSharedAcrossA2DBlock checks that the barrier and the shared slab span
// the whole block, not just its x extent.
func TestSharedAcrossA2DBlock(t *testing.T) {
	const bx, by = 4, 3
	out := make([]int, bx*by)
	gpu.RunCPUDim(gpu.D1(1), gpu.D2(bx, by), func(c gpu.Ctx) {
		s := c.SharedF32(bx * by)
		i := c.ThreadIdxY()*bx + c.ThreadIdx()
		s[i] = float32(i)
		c.SyncThreads()
		// Every thread reads what its neighbour wrote, which only works if
		// the barrier waited for all bx*by of them.
		out[i] = int(s[(i+1)%(bx*by)])
	})
	for i, v := range out {
		if want := (i + 1) % (bx * by); v != want {
			t.Errorf("thread %d read %d, want %d", i, v, want)
		}
	}
}

// TestAssumeBlockDimCountsTheWholeBlock pins what the launch contract counts:
// threads per block, not the x extent.
func TestAssumeBlockDimCountsTheWholeBlock(t *testing.T) {
	gpu.RunCPUDim(gpu.D1(1), gpu.D2(8, 4), func(c gpu.Ctx) { c.AssumeBlockDim(32) })

	msg := mustPanic(t, func() {
		gpu.RunCPUDim(gpu.D1(1), gpu.D2(8, 4), func(c gpu.Ctx) { c.AssumeBlockDim(8) })
	})
	if !strings.Contains(msg, "AssumeBlockDim") {
		t.Errorf("diagnosis %q does not mention AssumeBlockDim", msg)
	}
}

// The atomic tests below are all order-independent by construction. The
// emulator runs a goroutine per thread, so the order in which they win the
// lock is not reproducible; any assertion that depended on it would be a
// flaky test dressed up as a correctness test.

// TestAtomicAddI32Counts is the plain lost-update test: 256 threads each add
// one, and a non-atomic implementation loses some of them.
func TestAtomicAddI32Counts(t *testing.T) {
	c := make([]int32, 1)
	gpu.RunCPU(8, 32, func(_ gpu.Ctx) { gpu.AtomicAddI32(c, 0, 1) })
	if c[0] != 256 {
		t.Errorf("counter is %d, want 256", c[0])
	}
}

// TestAtomicReturnsTheOldValue is the stronger claim, and the one that makes
// the returned value worth having: across all threads the values handed back
// must be a permutation of 0..n-1. A lost update duplicates one, and an
// implementation that returned the new value instead would never produce 0.
//
// Each thread writes its own slot, so the bookkeeping itself races with
// nothing and the test stays clean under -race.
func TestAtomicReturnsTheOldValue(t *testing.T) {
	const grid, block = 8, 32
	c := make([]int32, 1)
	seen := make([]int32, grid*block)
	gpu.RunCPU(grid, block, func(ctx gpu.Ctx) {
		old := gpu.AtomicAddI32(c, 0, 1)
		seen[ctx.GlobalID()] = old
	})

	count := make([]int, grid*block)
	for i, v := range seen {
		if v < 0 || int(v) >= len(count) {
			t.Fatalf("thread %d was handed %d, which is outside 0..%d", i, v, len(count)-1)
		}
		count[v]++
	}
	for v, n := range count {
		if n != 1 {
			t.Errorf("the value %d was handed to %d threads, want exactly 1", v, n)
		}
	}
}

// TestAtomicAddF32Histogram bins into float32. The counts stay far below 2**24,
// so every partial sum is exact and the result does not depend on the order the
// threads arrived in -- which is what lets this assert equality rather than a
// tolerance. On the device the same kernel is exact for the same reason and
// stops being so as soon as the addends are not small integers.
func TestAtomicAddF32Histogram(t *testing.T) {
	const grid, block, bins = 8, 32, 16
	h := make([]float32, bins)
	gpu.RunCPU(grid, block, func(ctx gpu.Ctx) {
		gpu.AtomicAddF32(h, ctx.GlobalID()%bins, 1)
	})
	for i, v := range h {
		if want := float32(grid * block / bins); v != want {
			t.Errorf("bin %d holds %v, want %v", i, v, want)
		}
	}
}

// TestAtomicMinMaxI32 drives the extremes from every thread at once, and pins
// the half that a no-op implementation would pass: min must not write when the
// candidate is larger, and must still report what was there.
func TestAtomicMinMaxI32(t *testing.T) {
	const grid, block = 8, 32
	ext := []int32{1 << 30, -(1 << 30)}
	gpu.RunCPU(grid, block, func(ctx gpu.Ctx) {
		v := int32(ctx.GlobalID()) - 100
		gpu.AtomicMinI32(ext, 0, v)
		gpu.AtomicMaxI32(ext, 1, v)
	})
	if ext[0] != -100 {
		t.Errorf("min is %d, want -100", ext[0])
	}
	if want := int32(grid*block - 1 - 100); ext[1] != want {
		t.Errorf("max is %d, want %d", ext[1], want)
	}

	one := []int32{7}
	if old := gpu.AtomicMinI32(one, 0, 9); old != 7 || one[0] != 7 {
		t.Errorf("min against a larger candidate returned %d and left %d, want 7 and 7", old, one[0])
	}
	if old := gpu.AtomicMaxI32(one, 0, 3); old != 7 || one[0] != 7 {
		t.Errorf("max against a smaller candidate returned %d and left %d, want 7 and 7", old, one[0])
	}
}

// TestAtomicCASI32SingleWinner is the mutual-exclusion primitive doing the one
// job it exists for. Every thread races to claim a flag; exactly one may win,
// and the flag must hold that winner's identity rather than anybody else's.
func TestAtomicCASI32SingleWinner(t *testing.T) {
	const grid, block = 8, 32
	flag := []int32{0}
	wins := []int32{0}
	gpu.RunCPU(grid, block, func(ctx gpu.Ctx) {
		if gpu.AtomicCASI32(flag, 0, 0, int32(ctx.GlobalID())+1) == 0 {
			gpu.AtomicAddI32(wins, 0, 1)
		}
	})
	if wins[0] != 1 {
		t.Errorf("%d threads believed they won the flag, want exactly 1", wins[0])
	}
	if flag[0] < 1 || int(flag[0]) > grid*block {
		t.Errorf("the flag holds %d, which is no thread's identity", flag[0])
	}

	// A losing compare-and-swap reports what is actually there, not the value
	// it compared against -- which is how a caller retries without re-reading.
	held := []int32{5}
	if old := gpu.AtomicCASI32(held, 0, 1, 9); old != 5 || held[0] != 5 {
		t.Errorf("a losing CAS returned %d and left %d, want 5 and 5", old, held[0])
	}
}

// TestAtomicExchI32 pins that exch is unconditional and still reports what it
// displaced.
func TestAtomicExchI32(t *testing.T) {
	s := []int32{3}
	if old := gpu.AtomicExchI32(s, 0, 8); old != 3 {
		t.Errorf("exch returned %d, want 3", old)
	}
	if s[0] != 8 {
		t.Errorf("exch left %d, want 8", s[0])
	}
}

// TestAtomicOnASharedTile is the case the generated C reaches through a
// generic pointer rather than a global one. In the emulator a tile is an
// ordinary slice, so what this really pins is that the vocabulary composes
// with SharedF32 at all -- the device half is simt's parity test.
func TestAtomicOnASharedTile(t *testing.T) {
	const block = 32
	out := make([]float32, block)
	gpu.RunCPU(1, block, func(ctx gpu.Ctx) {
		s := ctx.SharedF32(1)
		if ctx.ThreadIdx() == 0 {
			s[0] = 0
		}
		ctx.SyncThreads()
		gpu.AtomicAddF32(s, 0, 1)
		ctx.SyncThreads()
		out[ctx.ThreadIdx()] = s[0]
	})
	for i, v := range out {
		if v != block {
			t.Errorf("thread %d saw %v in the tile, want %d", i, v, block)
		}
	}
}

// TestAtomicOutsideALaunch calls the vocabulary on a plain slice with no
// RunCPU at all. A kernel is ordinary Go, and the CPU reference in a parity
// test is often the kernel called directly, so these must not require a Ctx.
func TestAtomicOutsideALaunch(t *testing.T) {
	i := []int32{4}
	f := []float32{1.5}
	if old := gpu.AtomicAddI32(i, 0, 3); old != 4 || i[0] != 7 {
		t.Errorf("add returned %d and left %d, want 4 and 7", old, i[0])
	}
	if old := gpu.AtomicAddF32(f, 0, 0.5); old != 1.5 || f[0] != 2 {
		t.Errorf("add returned %v and left %v, want 1.5 and 2", old, f[0])
	}
}

// TestAtomicPanicDoesNotStrandTheLock is the test for the defer, and it is the
// reason the defer is not a stylistic choice.
//
// An out-of-range index panics with the package lock held. RunCPU's per-thread
// recover turns that panic into a diagnosis rather than a crash, so without the
// defer the process would carry on holding a mutex nobody will ever release,
// and the *next* launch to use an atomic would block forever. One bad kernel
// would hang the whole test binary, some distance from the kernel that did it.
//
// The second half is therefore the assertion: if it returns at all, the lock
// was released.
func TestAtomicPanicDoesNotStrandTheLock(t *testing.T) {
	c := make([]int32, 1)
	msg := mustPanic(t, func() {
		gpu.RunCPU(1, 4, func(_ gpu.Ctx) { gpu.AtomicAddI32(c, 999, 1) })
	})
	wantContains(t, msg, "index out of range")

	done := make(chan struct{})
	go func() {
		defer close(done)
		gpu.RunCPU(1, 8, func(_ gpu.Ctx) { gpu.AtomicAddI32(c, 0, 1) })
	}()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("a launch using atomics blocked after an earlier kernel panicked holding the lock")
	}
	if c[0] != 8 {
		t.Errorf("counter is %d, want 8", c[0])
	}
}

// TestFminFmaxFollowTheBuiltinTheyClaimToBe is a test about the instrument
// rather than about a kernel.
//
// gpu.Fmin lowers to fminf, so the emulator has to be fminf. Go's math.Min is
// not: it propagates NaN, which its own documentation states, while IEEE 754
// minNum -- which fminf follows -- returns the operand that is not NaN. A
// parity test built on the wrong one would compare NaN against a number, which
// no tolerance can reconcile and which would read as a kernel bug rather than
// as a bug in the comparison.
func TestFminFmaxFollowTheBuiltinTheyClaimToBe(t *testing.T) {
	nan := float32(math.NaN())

	cases := []struct {
		name     string
		got      float32
		want     float32
		wantSign bool // check the sign bit too, for the zeros
	}{
		{name: "Fmin ignores a NaN on the right", got: gpu.Fmin(1, nan), want: 1},
		{name: "Fmin ignores a NaN on the left", got: gpu.Fmin(nan, 1), want: 1},
		{name: "Fmax ignores a NaN on the right", got: gpu.Fmax(1, nan), want: 1},
		{name: "Fmax ignores a NaN on the left", got: gpu.Fmax(nan, 1), want: 1},
		{name: "Fmin of two", got: gpu.Fmin(2, 3), want: 2},
		{name: "Fmin of two, reversed", got: gpu.Fmin(3, 2), want: 2},
		{name: "Fmax of two", got: gpu.Fmax(2, 3), want: 3},
		{name: "Fmax of two, reversed", got: gpu.Fmax(3, 2), want: 3},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.got != tc.want {
				t.Errorf("got %v, want %v", tc.got, tc.want)
			}
		})
	}

	// Both NaN is the one case where there is nothing to return but a NaN.
	if !math.IsNaN(float64(gpu.Fmin(nan, nan))) {
		t.Errorf("Fmin(NaN, NaN) = %v, want NaN", gpu.Fmin(nan, nan))
	}
	if !math.IsNaN(float64(gpu.Fmax(nan, nan))) {
		t.Errorf("Fmax(NaN, NaN) = %v, want NaN", gpu.Fmax(nan, nan))
	}

	// Signed zeros compare equal, so only the sign bit can tell which came
	// back. fminf prefers the negative zero and fmaxf the positive one.
	negZero := float32(math.Copysign(0, -1))
	if !math.Signbit(float64(gpu.Fmin(negZero, 0))) {
		t.Error("Fmin(-0, +0) should be -0")
	}
	if !math.Signbit(float64(gpu.Fmin(0, negZero))) {
		t.Error("Fmin(+0, -0) should be -0")
	}
	if math.Signbit(float64(gpu.Fmax(negZero, 0))) {
		t.Error("Fmax(-0, +0) should be +0")
	}
	if math.Signbit(float64(gpu.Fmax(0, negZero))) {
		t.Error("Fmax(+0, -0) should be +0")
	}
}
