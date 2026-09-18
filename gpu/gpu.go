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
//
// Two corners of the vocabulary are emulated rather than modelled, and each
// begins with what its emulation cannot promise: atomic.go, where the CPU
// serialises every read-modify-write on one lock and imposes an order the
// device does not, and warp.go, where a warp is a rendezvous between
// goroutines rather than threads already in lockstep -- so a short warp and
// ActiveMask both answer questions the device answers differently. Those notes
// are worth reading before relying on either.
package gpu

import (
	"fmt"
	"runtime"
	"runtime/debug"
	"sync"
)

// Dim is a grid or block extent, in threads or blocks per axis.
//
// It duplicates cuda.Dim3 on purpose. A kernel may import nothing but this
// package, so the launch geometry a kernel can talk about cannot come from
// the driver bindings; the two meet at the launch boundary instead.
type Dim struct{ X, Y, Z int }

// D1, D2 and D3 build an extent of the given rank, leaving the axes above it
// at one, which is what CUDA treats an unused axis as.
func D1(x int) Dim       { return Dim{X: x, Y: 1, Z: 1} }
func D2(x, y int) Dim    { return Dim{X: x, Y: y, Z: 1} }
func D3(x, y, z int) Dim { return Dim{X: x, Y: y, Z: z} }

// count is how many threads (or blocks) the extent holds.
func (d Dim) count() int { return d.X * d.Y * d.Z }

// flat numbers a position within an extent, x fastest -- the inverse of
// coord. Diagnostics use it so that a one-dimensional launch, which is most
// of them, still reads as a plain thread number.
func flat(pos, extent Dim) int {
	return pos.X + extent.X*(pos.Y+extent.Y*pos.Z)
}

// coord maps a flat index back to a position, x fastest, as CUDA numbers
// threads within a block.
func (d Dim) coord(i int) Dim {
	return Dim{X: i % d.X, Y: (i / d.X) % d.Y, Z: i / (d.X * d.Y)}
}

// Ctx carries a thread's position in the grid. On the GPU its methods become
// the corresponding CUDA built-ins; on the CPU they are served by RunCPU.
//
// The unsuffixed methods are the x axis, which is the spelling CUDA itself
// uses and what every one-dimensional kernel wants. GlobalID is the exception
// that gains an explicit GlobalIDX alias: next to GlobalIDY, the bare name
// reads like an oversight rather than an axis.
type Ctx struct {
	tid, bid   Dim
	bdim, gdim Dim
	block      *blockState
	thread     *threadState
}

// ThreadIdx is threadIdx.x.
func (c Ctx) ThreadIdx() int { return c.tid.X }

// ThreadIdxY is threadIdx.y.
func (c Ctx) ThreadIdxY() int { return c.tid.Y }

// ThreadIdxZ is threadIdx.z.
func (c Ctx) ThreadIdxZ() int { return c.tid.Z }

// BlockIdx is blockIdx.x.
func (c Ctx) BlockIdx() int { return c.bid.X }

// BlockIdxY is blockIdx.y.
func (c Ctx) BlockIdxY() int { return c.bid.Y }

// BlockIdxZ is blockIdx.z.
func (c Ctx) BlockIdxZ() int { return c.bid.Z }

// BlockDim is blockDim.x.
func (c Ctx) BlockDim() int { return c.bdim.X }

// BlockDimY is blockDim.y.
func (c Ctx) BlockDimY() int { return c.bdim.Y }

// BlockDimZ is blockDim.z.
func (c Ctx) BlockDimZ() int { return c.bdim.Z }

// GridDim is gridDim.x.
func (c Ctx) GridDim() int { return c.gdim.X }

// GridDimY is gridDim.y.
func (c Ctx) GridDimY() int { return c.gdim.Y }

// GridDimZ is gridDim.z.
func (c Ctx) GridDimZ() int { return c.gdim.Z }

// GlobalID is blockIdx.x*blockDim.x + threadIdx.x.
func (c Ctx) GlobalID() int { return c.bid.X*c.bdim.X + c.tid.X }

// GlobalIDX is GlobalID, spelled so it reads as one axis of three.
func (c Ctx) GlobalIDX() int { return c.bid.X*c.bdim.X + c.tid.X }

// GlobalIDY is blockIdx.y*blockDim.y + threadIdx.y.
func (c Ctx) GlobalIDY() int { return c.bid.Y*c.bdim.Y + c.tid.Y }

// GlobalIDZ is blockIdx.z*blockDim.z + threadIdx.z.
func (c Ctx) GlobalIDZ() int { return c.bid.Z*c.bdim.Z + c.tid.Z }

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
// SharedF32 call, and every thread must ask for the same n at the same call.
// A __shared__ declaration on the device is a property of the compiled kernel,
// not of the thread that executes it, so a kernel whose threads disagree about
// how much shared memory exists is not a kernel the GPU could run at all. The
// emulator therefore does not try to give such a kernel a plausible answer: it
// records the disagreement and RunCPU turns it into a panic (see blockState
// for why the panic is deferred to RunCPU's caller).
func (c Ctx) SharedF32(n int) []float32 {
	if c.block == nil {
		// A zero-value Ctx is not part of a launch, so there is nobody to
		// share with and nothing to cross-check: hand out a private buffer.
		return make([]float32, n)
	}
	i := c.thread.seq
	c.thread.seq++

	c.block.mu.Lock()
	defer c.block.mu.Unlock()
	for len(c.block.slabs) <= i {
		c.block.slabs = append(c.block.slabs, slab{})
	}
	s := &c.block.slabs[i]
	if !s.alloc {
		s.alloc, s.size, s.buf = true, n, make([]float32, n)
		return s.buf
	}
	if s.size != n {
		c.block.reportLocked(fmt.Sprintf(
			"gpu: SharedF32 size mismatch at call #%d of block %d: an earlier thread asked for %d element(s), thread %d asked for %d. "+
				"Shared memory is sized once per block, so every thread of a block must pass the same (constant) size to the same SharedF32 call.",
			i, flat(c.bid, c.gdim), s.size, flat(c.tid, c.bdim), n))
		// Return a private buffer of the size that was actually requested so
		// the offending thread can run to completion without an out-of-range
		// panic drowning out the diagnosis above. Its results are meaningless,
		// but nobody will get to read them: RunCPU panics.
		return make([]float32, n)
	}
	return s.buf
}

// AssumeBlockDim declares the block size this kernel was written for.
//
// Kernels routinely size their shared memory against a block size baked into
// the source ("s := ctx.SharedF32(256)" next to an implicit assumption that
// blockDim.x is 256). Launched with a different block size such a kernel does
// not fail, it quietly computes the wrong thing: threads beyond the declared
// size run off the end of the tile, or part of the tile is never filled and
// contributes zeros. Stating the assumption turns that class of silent
// corruption into a check the CPU emulator performs here, and the GPU launch
// path performs against the kernel's recorded block size.
//
// Outside RunCPU (a zero-value Ctx, which belongs to no launch) it is a no-op,
// exactly as SyncThreads is, so that a kernel body can still be called as a
// plain function. On the device the transpiler emits nothing for it: the
// declared size has already been consumed at compile time.
//
// What is counted is threads per block, across all three axes: a shared tile
// is sized against how many threads fill it, not against how they are
// arranged.
func (c Ctx) AssumeBlockDim(n int) {
	if c.block != nil && c.bdim.count() != n {
		c.block.report(fmt.Sprintf(
			"gpu: block size mismatch: the kernel declares AssumeBlockDim(%d) but was launched with blockDim=%d. "+
				"Launch it with %d threads per block, or rewrite the kernel so it works for any block size.",
			n, c.bdim.count(), n))
	}
}

// threadState is the per-thread bookkeeping the emulator needs. seq is the
// number of SharedF32 calls this thread has made so far, which doubles as the
// index of its next call; comparing the final values across a block is how
// divergent shared-memory usage is detected.
//
// warp and lane say which rendezvous the warp-level primitives join and which
// slot of it this thread owns. They are fixed when the block is built, because
// which warp a thread belongs to is a property of its position and not of what
// it does.
type threadState struct {
	seq  int
	warp *warpState
	lane int
}

// slab is one shared buffer of a block together with the size it was created
// with. Keeping the size is what allows a later thread that asks for a
// different size at the same call index to be diagnosed rather than silently
// handed a buffer of the wrong length. alloc is tracked separately because a
// legitimately empty slab (n == 0) is indistinguishable from an unused one by
// looking at buf alone.
type slab struct {
	buf   []float32
	size  int
	alloc bool
}

// blockState is the state the threads of one block share.
//
// fail holds the first contract violation observed in the block. Violations
// are detected on a thread's own goroutine, but they deliberately do not panic
// there: a panic on a thread goroutine would take down the whole test process,
// because recover runs only on the stack of the goroutine that panicked, and
// it would strand the thread's siblings in the barrier forever. So the
// diagnosis is recorded, the offending call returns something harmless, the
// block runs to completion, and RunCPU re-raises the first recorded diagnosis
// as a panic on its own caller's goroutine — loud, and catchable by both the
// kernel author's recover and a test's.
type blockState struct {
	bar   *barrier
	warps []*warpState
	mu    sync.Mutex
	slabs []slab
	fail  string
}

// report records a contract violation. The first one wins: later violations
// are usually consequences of the first and would only bury it.
func (s *blockState) report(msg string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.reportLocked(msg)
}

// reportLocked is report for callers that already hold s.mu.
func (s *blockState) reportLocked(msg string) {
	if s.fail == "" {
		s.fail = msg
	}
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
//
// RunCPU panics if the kernel violates the block-level contract the emulator
// can check (see SharedF32 and AssumeBlockDim) or if the kernel itself panics
// on one of its thread goroutines. The panic is raised here, on RunCPU's own
// caller's goroutine, rather than where the problem was found, so that it is
// recoverable and testable instead of terminating the process from a goroutine
// nobody can recover on.
func RunCPU(grid, block int, fn func(Ctx)) {
	RunCPUDim(D1(grid), D1(block), fn)
}

// RunCPUDim is RunCPU over a grid of any rank.
//
// Blocks are drained from a queue in flat order, x fastest, and the threads of
// a block are one goroutine each, exactly as in the one-dimensional case: the
// only thing that changes is how many there are and what coordinates they see.
func RunCPUDim(grid, block Dim, fn func(Ctx)) {
	if grid.X <= 0 || grid.Y <= 0 || grid.Z <= 0 {
		return
	}
	if block.X <= 0 || block.Y <= 0 || block.Z <= 0 {
		return
	}
	n := grid.count()
	workers := min(runtime.GOMAXPROCS(0), n)
	blocks := make(chan int, n)
	for b := range n {
		blocks <- b
	}
	close(blocks)

	var (
		failMu sync.Mutex
		fail   string
	)
	var wg sync.WaitGroup
	wg.Add(workers)
	for range workers {
		go func() {
			defer wg.Done()
			for b := range blocks {
				// Once one block has produced a diagnosis, running the rest of
				// the grid can only produce more of the same, so drain the
				// queue instead. The launch is about to panic anyway.
				failMu.Lock()
				stop := fail != ""
				failMu.Unlock()
				if stop {
					continue
				}
				if msg := runBlock(b, grid, block, fn); msg != "" {
					failMu.Lock()
					if fail == "" {
						fail = msg
					}
					failMu.Unlock()
				}
			}
		}()
	}
	wg.Wait()

	// Safe without the mutex: wg.Wait happens after every write to fail.
	if fail != "" {
		panic(fail)
	}
}

// runBlock runs the threads of one block and returns the first contract
// violation it observed, or "" if the block was well behaved. It returns the
// diagnosis rather than panicking because it runs on one of RunCPU's worker
// goroutines, which is no more recoverable than a thread goroutine is.
func runBlock(bid int, grid, block Dim, fn func(Ctx)) string {
	n := block.count()
	st := &blockState{bar: newBarrier(n)}

	// A warp is 32 consecutive threads by flat index, which is how CUDA
	// partitions a block, so t is already the number that decides both the
	// warp and the lane. A block that is not a multiple of 32 ends in a short
	// warp, and the emulator sizes that one at what is left rather than
	// pretending the missing lanes are there: see gpu/warp.go for what the
	// device does instead.
	st.warps = make([]*warpState, (n+WarpSize-1)/WarpSize)
	for w := range st.warps {
		size := min(WarpSize, n-w*WarpSize)
		st.warps[w] = &warpState{bar: newWarpBarrier(size), block: st, bid: bid, id: w, size: size}
	}

	// The thread states outlive the goroutines that use them: after the block
	// has finished, their seq counters say how many SharedF32 calls each
	// thread made, and those counts must agree.
	threads := make([]*threadState, n)
	for t := range n {
		threads[t] = &threadState{warp: st.warps[t/WarpSize], lane: t % WarpSize}
	}

	var wg sync.WaitGroup
	wg.Add(n)
	for t := range n {
		go func() {
			defer wg.Done()
			// A thread that leaves the kernel stops being a participant in its
			// warp, whether it returned or panicked -- which is what keeps
			// ordinary divergence from stranding the threads waiting for it.
			// Registered first so that it runs last, after the recover below
			// has turned a panic into a diagnosis.
			defer threads[t].warp.bar.abandon()
			// A kernel that panics on a thread goroutine would otherwise kill
			// the process. Catch it, let the barrier forget the thread so its
			// siblings are not stranded, and hand the message upwards to be
			// re-raised by RunCPU. The original stack is kept, since that is
			// the part with the diagnostic value.
			defer func() {
				if r := recover(); r != nil {
					st.report(fmt.Sprintf("gpu: kernel panicked in block %d, thread %d: %v\n\n%s", bid, t, r, debug.Stack()))
					st.bar.abandon()
				}
			}()
			fn(Ctx{
				tid: block.coord(t), bid: grid.coord(bid), bdim: block, gdim: grid,
				block: st, thread: threads[t],
			})
		}()
	}
	wg.Wait()

	if msg := divergence(bid, threads); msg != "" {
		st.report(msg)
	}
	return st.fail // no lock needed: every writer has been joined
}

// divergence reports whether the threads of a finished block disagreed about
// how many SharedF32 calls to make. Since calls are matched by execution
// order, a thread that skipped one shares the wrong buffer with everybody from
// that point on, which is a wrong answer rather than a crash — hence the
// check.
func divergence(bid int, threads []*threadState) string {
	if len(threads) == 0 {
		return ""
	}
	want := threads[0].seq
	for t, ts := range threads {
		if ts.seq != want {
			return fmt.Sprintf(
				"gpu: divergent SharedF32 usage in block %d: thread 0 made %d SharedF32 call(s), thread %d made %d. "+
					"Shared buffers are matched across threads by call order, so every thread of a block must reach every SharedF32 call; "+
					"hoist the call out of the conditional that threads disagree about.",
				bid, want, t, ts.seq)
		}
	}
	return ""
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

// abandon removes one participant from the barrier for good. It exists for the
// one case where a thread stops taking part without reaching the end of the
// kernel: a panic. Without it the surviving threads would wait at the next
// barrier for a participant that no longer exists, and the deadlock would hide
// the panic that caused it.
func (b *barrier) abandon() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.n--
	// The departure may have been the one the others were waiting for.
	if b.n > 0 && b.count >= b.n {
		b.count = 0
		b.gen++
		b.cond.Broadcast()
	}
}
