// Package gpu is the vocabulary kernels are written in.
//
// A kernel is an ordinary Go function taking a Ctx as its first parameter:
//
//	func VecAdd(ctx gpu.Ctx, c, a, b []float32) {
//		i := ctx.GlobalID()
//		if i < len(c) {
//			c[i] = a[i] + b[i]
//		}
//	}
//
// That function compiles, runs and can be tested as plain Go via RunCPU, and
// the transpiler in package simt lowers the very same source to CUDA C. One
// source, two backends.
package gpu

import (
	"runtime"
	"sync"
)

// Ctx carries a thread's position in the grid. On the GPU its methods become
// the corresponding CUDA built-ins; on the CPU they are served by RunCPU.
type Ctx struct {
	tid, bid   int
	bdim, gdim int
	block      *blockState
	thread     *threadState
}

// ThreadIdx is threadIdx.x.
func (c Ctx) ThreadIdx() int { return c.tid }

// BlockIdx is blockIdx.x.
func (c Ctx) BlockIdx() int { return c.bid }

// BlockDim is blockDim.x.
func (c Ctx) BlockDim() int { return c.bdim }

// GridDim is gridDim.x.
func (c Ctx) GridDim() int { return c.gdim }

// GlobalID is blockIdx.x*blockDim.x + threadIdx.x.
func (c Ctx) GlobalID() int { return c.bid*c.bdim + c.tid }

// SyncThreads is __syncthreads(): a barrier across the threads of one block.
func (c Ctx) SyncThreads() {
	if c.block != nil {
		c.block.bar.wait()
	}
}

// SharedF32 returns this block's shared scratch buffer of n float32 values,
// lowered to a __shared__ array on the GPU. All threads of a block see the
// same buffer, so n must be a constant expression the transpiler can fold.
//
// Calls are matched between threads by their order of execution, exactly as
// __shared__ declarations are: every thread of a block must reach every
// SharedF32 call.
func (c Ctx) SharedF32(n int) []float32 {
	if c.block == nil {
		return make([]float32, n)
	}
	i := c.thread.seq
	c.thread.seq++

	c.block.mu.Lock()
	defer c.block.mu.Unlock()
	for len(c.block.slabs) <= i {
		c.block.slabs = append(c.block.slabs, nil)
	}
	if c.block.slabs[i] == nil {
		c.block.slabs[i] = make([]float32, n)
	}
	return c.block.slabs[i]
}

type threadState struct{ seq int }

type blockState struct {
	bar   *barrier
	mu    sync.Mutex
	slabs [][]float32
}

// RunCPU executes fn over a grid of blocks, emulating a CUDA launch.
//
// Blocks run concurrently and in no particular order, which is what CUDA
// guarantees too, so `go test -race` reports genuine inter-block races.
// Threads within a block are goroutines sharing a real barrier, so
// SyncThreads and SharedF32 behave as they do on the device.
//
// This is a debugging and correctness tool, not a fast CPU backend: a
// goroutine per thread is nothing like a warp. Benchmarks should compare
// against an ordinary Go loop instead.
func RunCPU(grid, block int, fn func(Ctx)) {
	if grid <= 0 || block <= 0 {
		return
	}
	workers := min(runtime.GOMAXPROCS(0), grid)
	blocks := make(chan int, grid)
	for b := range grid {
		blocks <- b
	}
	close(blocks)

	var wg sync.WaitGroup
	wg.Add(workers)
	for range workers {
		go func() {
			defer wg.Done()
			for b := range blocks {
				runBlock(b, grid, block, fn)
			}
		}()
	}
	wg.Wait()
}

func runBlock(bid, grid, block int, fn func(Ctx)) {
	st := &blockState{bar: newBarrier(block)}
	var wg sync.WaitGroup
	wg.Add(block)
	for t := range block {
		go func() {
			defer wg.Done()
			fn(Ctx{
				tid: t, bid: bid, bdim: block, gdim: grid,
				block: st, thread: &threadState{},
			})
		}()
	}
	wg.Wait()
}

// barrier is a reusable barrier for a fixed number of participants.
type barrier struct {
	mu    sync.Mutex
	cond  *sync.Cond
	n     int
	count int
	gen   uint64
}

func newBarrier(n int) *barrier {
	b := &barrier{n: n}
	b.cond = sync.NewCond(&b.mu)
	return b
}

func (b *barrier) wait() {
	b.mu.Lock()
	defer b.mu.Unlock()
	gen := b.gen
	b.count++
	if b.count == b.n {
		b.count = 0
		b.gen++
		b.cond.Broadcast()
		return
	}
	for gen == b.gen {
		b.cond.Wait()
	}
}
