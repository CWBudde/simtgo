package gpu_test

import (
	"testing"

	"github.com/CWBudde/gocuda/gpu"
)

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
