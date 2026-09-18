package gpu_test

import (
	"fmt"
	"strings"
	"sync/atomic"
	"testing"

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
		"divergent SharedF32 usage",
		"block 0",
		"thread 0 made 2 SharedF32 call(s), thread 1 made 1")
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
