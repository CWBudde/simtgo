package gpu

import "fmt"

// The shared-tile vocabulary: one constructor per element type, in the two
// shapes CUDA has.
//
// A statically sized tile -- SharedF32(256) -- lowers to a `__shared__ float
// name[256]` inside the kernel, so n has to be a constant expression the
// transpiler can fold. A dynamic one -- SharedDynF32() -- lowers to the
// `extern __shared__` block, whose size is given at the launch instead, which
// is why it takes no argument here and why a kernel may have at most one: CUDA
// has exactly one dynamic block per launch, and two names for it would be two
// views of the same bytes.
//
// There is no SharedBool, and it is a gap rather than a decision. Go's bool
// and C++'s bool are both one byte, so a tile of them would lower cleanly;
// what is missing is a reason to have one. Nothing in the vocabulary reads or
// writes a bool atomically, so such a tile could only ever be filled under a
// barrier, and the int32 tile that already does that is also the one the
// atomics work on. Nothing is in the way if a kernel turns up that wants one.

// slab is one shared buffer of a block together with what created it. The
// element type is kept as the buffer's own dynamic type and the constructor's
// name is kept beside it for the message, which is what lets a thread calling
// SharedI32 where a sibling called SharedF32 be diagnosed rather than handed a
// buffer of the wrong shape. alloc is tracked separately because a
// legitimately empty slab (n == 0) is indistinguishable from an unused one by
// looking at buf alone.
type slab struct {
	buf   any    // []float32, []int32, ... whichever constructor filled it
	kind  string // that constructor's name, for the diagnosis
	size  int
	alloc bool
}

// sharedTile is the body every constructor below shares.
//
// Calls are matched between threads by their order of execution, exactly as
// __shared__ declarations are: every thread of a block must reach every call,
// and must ask for the same element type and the same size at the same one. A
// __shared__ declaration on the device is a property of the compiled kernel,
// not of the thread that executes it, so a kernel whose threads disagree about
// what shared memory exists is not a kernel the GPU could run at all. The
// emulator therefore does not try to give such a kernel a plausible answer: it
// records the disagreement, hands the offending thread a private buffer so it
// runs to completion instead of drowning the diagnosis in an out-of-range
// panic, and RunCPU turns the record into a panic on its own caller's
// goroutine (see blockState for why it is deferred that far).
func sharedTile[T any](c Ctx, n int, kind string) []T {
	if c.block == nil {
		// A zero-value Ctx is not part of a launch, so there is nobody to
		// share with and nothing to cross-check: hand out a private buffer.
		return make([]T, n)
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
		buf := make([]T, n)
		s.alloc, s.kind, s.size, s.buf = true, kind, n, buf
		return buf
	}
	// The type assertion is the element-type check itself rather than a
	// comparison of the two names: a name is what the message needs, and
	// deciding on it as well would leave the buffer's real type unchecked.
	buf, ok := s.buf.([]T)
	if !ok {
		c.block.reportLocked(fmt.Sprintf(
			"gpu: shared tile type mismatch at call #%d of block %d: an earlier thread built it with %s, thread %d called %s. "+
				"A __shared__ array has one element type, fixed when the kernel is compiled, so every thread of a block must reach the same constructor at the same call.",
			i, flat(c.bid, c.gdim), s.kind, flat(c.tid, c.bdim), kind))
		return make([]T, n)
	}
	if s.size != n {
		c.block.reportLocked(fmt.Sprintf(
			"gpu: %s size mismatch at call #%d of block %d: an earlier thread asked for %d element(s), thread %d asked for %d. "+
				"Shared memory is sized once per block, so every thread of a block must pass the same (constant) size to the same %s call.",
			kind, i, flat(c.bid, c.gdim), s.size, flat(c.tid, c.bdim), n, kind))
		return make([]T, n)
	}
	return buf
}

// SharedF32 returns this block's shared scratch buffer of n float32 values,
// lowered to a __shared__ float array on the GPU. All threads of a block see
// the same buffer, so n must be a constant expression the transpiler can fold.
func (c Ctx) SharedF32(n int) []float32 { return sharedTile[float32](c, n, "SharedF32") }

// SharedF64 is SharedF32 for float64, and needs //simtgo:float64 on the kernel
// for the reason every other double on the device does: the permission is
// about what the launch costs, and a tile of doubles is also twice the shared
// memory a block has to find.
func (c Ctx) SharedF64(n int) []float64 { return sharedTile[float64](c, n, "SharedF64") }

// SharedI32 is SharedF32 for int32. It is the tile the atomics reach: a
// histogram privatised per block is an int32 tile, atomicAdd'd into by the
// block's threads and then folded into global memory once.
func (c Ctx) SharedI32(n int) []int32 { return sharedTile[int32](c, n, "SharedI32") }

// SharedI64 is SharedF32 for int64.
func (c Ctx) SharedI64(n int) []int64 { return sharedTile[int64](c, n, "SharedI64") }

// SharedU32 is SharedF32 for uint32.
func (c Ctx) SharedU32(n int) []uint32 { return sharedTile[uint32](c, n, "SharedU32") }

// SharedU64 is SharedF32 for uint64.
func (c Ctx) SharedU64(n int) []uint64 { return sharedTile[uint64](c, n, "SharedU64") }

// dynamicTile is sharedTile for the launch-sized block: the length comes from
// the launch rather than from the call, so there is nothing at the call site
// to disagree about and everything else -- one buffer per block, matched
// across threads by call order -- is unchanged.
//
// A launch that never sized the block is a contract violation rather than a
// tile of length zero. On the device that launch would pass sharedBytes = 0
// and every access would run off the end of nothing, so the emulator says so
// instead of handing back an empty slice the kernel will index.
func dynamicTile[T any](c Ctx, kind string) []T {
	if c.block == nil {
		// Outside a launch there is no size to have been given, so there is
		// nothing to diagnose either. An empty tile is the honest answer: a
		// dynamic tile has no size of its own.
		return nil
	}
	if !c.block.dynSized {
		c.block.report(fmt.Sprintf(
			"gpu: %s in block %d, but this launch did not size the dynamic tile. "+
				"Run the kernel with RunCPUShared(grid, block, n, fn), which is the emulator's half of Kernel.LaunchShared.",
			kind, flat(c.bid, c.gdim)))
		return nil
	}
	return sharedTile[T](c, c.block.dyn, kind)
}

// SharedDynF32 returns this block's dynamically sized shared buffer of
// float32, the one whose length the launch gives rather than the source.
//
// It takes no argument for that reason, and a kernel may declare at most one
// tile this way whatever its element type: CUDA has a single dynamic
// __shared__ block per launch, so a second name would be a second view of the
// same bytes. NVRTC compiles that quite happily, which is why the transpiler
// refuses it with a position instead of leaving the aliasing to be discovered.
func (c Ctx) SharedDynF32() []float32 { return dynamicTile[float32](c, "SharedDynF32") }

// SharedDynF64 is SharedDynF32 for float64, and needs //simtgo:float64 for the
// same reason SharedF64 does.
func (c Ctx) SharedDynF64() []float64 { return dynamicTile[float64](c, "SharedDynF64") }

// SharedDynI32 is SharedDynF32 for int32.
func (c Ctx) SharedDynI32() []int32 { return dynamicTile[int32](c, "SharedDynI32") }

// SharedDynI64 is SharedDynF32 for int64.
func (c Ctx) SharedDynI64() []int64 { return dynamicTile[int64](c, "SharedDynI64") }

// SharedDynU32 is SharedDynF32 for uint32.
func (c Ctx) SharedDynU32() []uint32 { return dynamicTile[uint32](c, "SharedDynU32") }

// SharedDynU64 is SharedDynF32 for uint64.
func (c Ctx) SharedDynU64() []uint64 { return dynamicTile[uint64](c, "SharedDynU64") }
