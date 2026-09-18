package gpu_test

import (
	"strings"
	"testing"
	"time"

	"github.com/CWBudde/gocuda/gpu"
)

// TestSharedTileElementTypeMismatch is the size check's sibling, and the one
// the typed tiles made possible: two threads reaching the same call index with
// different constructors.
//
// It is the same class of contract violation and gets the same treatment. A
// __shared__ array's element type is fixed when the kernel is compiled, so a
// kernel whose threads disagree about it is not a kernel the device could run;
// the emulator records that, hands the offending thread a private buffer so it
// finishes instead of drowning the diagnosis in an out-of-range panic, and
// RunCPU raises it on its own caller's goroutine.
func TestSharedTileElementTypeMismatch(t *testing.T) {
	msg := mustPanic(t, func() {
		gpu.RunCPU(1, 8, func(ctx gpu.Ctx) {
			if ctx.ThreadIdx()%2 == 0 {
				_ = ctx.SharedF32(8)
			} else {
				_ = ctx.SharedI32(8)
			}
		})
	})
	wantContains(t, msg, "shared tile type mismatch", "call #0", "block 0")
	// Whichever constructor won the race for the first allocation, both have
	// to be named: the message is of no use if it says only one of them.
	if !strings.Contains(msg, "SharedF32") || !strings.Contains(msg, "SharedI32") {
		t.Errorf("panic message does not name both constructors:\n%s", msg)
	}
}

// TestTypedSharedTiles is the negative control for the check above, and the
// evidence that each constructor hands back a buffer of its own element type
// rather than one buffer reinterpreted.
func TestTypedSharedTiles(t *testing.T) {
	const grid, block = 2, 8
	out := make([]int64, grid*block)
	gpu.RunCPU(grid, block, func(ctx gpu.Ctx) {
		f32 := ctx.SharedF32(block)
		i32 := ctx.SharedI32(block)
		i64 := ctx.SharedI64(block)
		u64 := ctx.SharedU64(block)
		i := ctx.ThreadIdx()
		f32[i], i32[i], i64[i], u64[i] = 1, 2, 4, 8
		ctx.SyncThreads()
		// A neighbour's element rather than this thread's: what is checked is
		// that the buffers are shared across the block and distinct from each
		// other, not that a write can be read back.
		j := (i + 1) % block
		out[ctx.GlobalID()] = int64(f32[j]) + int64(i32[j]) + i64[j] + int64(u64[j])
	})
	for i, v := range out {
		if v != 15 {
			t.Fatalf("out[%d] = %d, want 15", i, v)
		}
	}
}

// TestAtomicAddI32OnASharedTile is the pair the Histogram kernel is built out
// of. A tile is an ordinary Go slice here, so what this pins is that the
// atomics and the int32 tile compose at all; the device half is simt's parity
// test.
func TestAtomicAddI32OnASharedTile(t *testing.T) {
	const grid, block, bins = 4, 32, 8
	total := make([]int32, bins)
	gpu.RunCPU(grid, block, func(ctx gpu.Ctx) {
		tile := ctx.SharedI32(bins)
		if ctx.ThreadIdx() < bins {
			tile[ctx.ThreadIdx()] = 0
		}
		ctx.SyncThreads()
		gpu.AtomicAddI32(tile, ctx.ThreadIdx()%bins, 1)
		ctx.SyncThreads()
		if ctx.ThreadIdx() < bins {
			gpu.AtomicAddI32(total, ctx.ThreadIdx(), tile[ctx.ThreadIdx()])
		}
	})
	for i, v := range total {
		if want := int32(grid * block / bins); v != want {
			t.Errorf("bin %d holds %d, want %d", i, v, want)
		}
	}
}

// TestDynamicSharedTile pins the emulator's half of the launch-sized tile: the
// length comes from RunCPUShared, every thread of a block sees one buffer, and
// len() reports what the launch gave.
func TestDynamicSharedTile(t *testing.T) {
	const grid, block, n = 3, 8, 5
	out := make([]int32, grid)
	gpu.RunCPUShared(grid, block, n, func(ctx gpu.Ctx) {
		s := ctx.SharedDynI32()
		if len(s) != n {
			t.Errorf("len(SharedDynI32()) = %d, want %d", len(s), n)
		}
		for i := ctx.ThreadIdx(); i < len(s); i += ctx.BlockDim() {
			s[i] = 0
		}
		ctx.SyncThreads()
		gpu.AtomicAddI32(s, ctx.ThreadIdx()%n, 1)
		ctx.SyncThreads()
		if ctx.ThreadIdx() == 0 {
			sum := int32(0)
			for _, v := range s {
				sum += v
			}
			out[ctx.BlockIdx()] = sum
		}
	})
	for b, v := range out {
		if v != block {
			t.Errorf("block %d counted %d threads, want %d", b, v, block)
		}
	}
}

// TestDynamicSharedTileWithoutASize proves that a kernel asking for the
// dynamic tile under a launch that never sized one is diagnosed rather than
// handed an empty slice it goes on to index.
//
// On the device such a launch passes sharedBytes = 0 and every access runs off
// the end of nothing, which is the silent wrong answer the emulator exists to
// turn into a message.
func TestDynamicSharedTileWithoutASize(t *testing.T) {
	msg := mustPanic(t, func() {
		gpu.RunCPU(1, 4, func(ctx gpu.Ctx) { _ = ctx.SharedDynF32() })
	})
	wantContains(t, msg, "SharedDynF32", "did not size the dynamic tile", "RunCPUShared")

	// A negative size is the caller's own mistake, so it is raised on the
	// caller's goroutine, which is already where it happened.
	neg := mustPanic(t, func() {
		gpu.RunCPUShared(1, 4, -1, func(ctx gpu.Ctx) { _ = ctx.SharedDynF32() })
	})
	wantContains(t, neg, "not a size")
}

// TestSharedTileOutsideALaunch covers the zero-value Ctx, which belongs to no
// launch: a kernel body has to stay callable as plain Go, which is how a
// parity test writes its CPU reference.
func TestSharedTileOutsideALaunch(t *testing.T) {
	var ctx gpu.Ctx
	if got := len(ctx.SharedI64(4)); got != 4 {
		t.Errorf("len(SharedI64(4)) = %d, want 4 outside a launch", got)
	}
	// A dynamic tile has no size of its own, and outside a launch there is
	// nobody to have given it one, so an empty tile is the honest answer
	// rather than a diagnosis about a launch that does not exist.
	if got := len(ctx.SharedDynI64()); got != 0 {
		t.Errorf("len(SharedDynI64()) = %d, want 0 outside a launch", got)
	}
}

// TestSharedTilePanicDoesNotStrandTheBlock is the test for the per-block lock
// and the barrier together, on the tile path.
//
// A thread that panics between two shared-tile calls must hold nothing by the
// time it unwinds -- sharedTile unlocks through defer -- and must be removed
// from the barrier, or its siblings wait at the next SyncThreads for a
// participant that no longer exists. Either failure is a hang rather than a
// wrong answer, so the assertion is that the second launch returns at all.
func TestSharedTilePanicDoesNotStrandTheBlock(t *testing.T) {
	msg := mustPanic(t, func() {
		gpu.RunCPUShared(1, 8, 8, func(ctx gpu.Ctx) {
			a := ctx.SharedI32(8)
			a[ctx.ThreadIdx()] = 1
			ctx.SyncThreads()
			if ctx.ThreadIdx() == 3 {
				_ = a[999] // out of range, with two more shared calls to come
			}
			b := ctx.SharedDynI32()
			// Each thread writes its own slot. Every thread writing b[0] would
			// be a genuine race -- on the device as much as here -- and the
			// emulator is built to keep reporting one under -race rather than
			// to make a tile look atomic. The subject of this test is the
			// barrier and the lock, so it must not smuggle in a race of its own.
			b[ctx.ThreadIdx()] = 1
			ctx.SyncThreads()
		})
	})
	wantContains(t, msg, "index out of range")

	done := make(chan struct{})
	go func() {
		defer close(done)
		gpu.RunCPUShared(4, 8, 8, func(ctx gpu.Ctx) {
			a := ctx.SharedI32(8)
			b := ctx.SharedDynI32()
			a[ctx.ThreadIdx()] = 1
			b[ctx.ThreadIdx()] = 1
			ctx.SyncThreads()
		})
	}()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("a launch using shared tiles blocked after an earlier kernel panicked between two of them")
	}
}
