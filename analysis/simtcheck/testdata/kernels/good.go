package kernels

import "github.com/CWBudde/gocuda/gpu"

// VecAdd lowers cleanly and must draw no diagnostics at all.
func VecAdd(ctx gpu.Ctx, c, a, b []float32) {
	i := ctx.GlobalID()
	if i < len(c) {
		c[i] = a[i] + b[i]
	}
}

// scaled is an ordinary Go function, and a device function once a kernel
// calls it. Neither it nor its caller may draw a diagnostic.
func scaled(v, k float32) float32 { return v * k }

// Scale lowers cleanly through a device function.
func Scale(ctx gpu.Ctx, y, x []float32, k float32) {
	i := ctx.GlobalID()
	if i < len(y) {
		y[i] = scaled(x[i], k)
	}
}

// Histogram is the accepted half of the atomics rule, and the half that
// matters: the analyzer and the transpiler have to agree that a buffer and an
// index is the right shape, not merely that everything else is refused.
// Nothing else makes them agree.
func Histogram(ctx gpu.Ctx, bins []int32, x []int32) {
	i := ctx.GlobalID()
	if i < len(x) {
		gpu.AtomicAddI32(bins, int(x[i])%len(bins), 1)
	}
}

// Accumulate reaches the other addressable thing a kernel has: a shared tile.
func Accumulate(ctx gpu.Ctx, y []float32, x []float32) {
	s := ctx.SharedF32(1)
	if ctx.ThreadIdx() == 0 {
		s[0] = 0
	}
	ctx.SyncThreads()
	i := ctx.GlobalID()
	if i < len(x) {
		gpu.AtomicAddF32(s, 0, x[i])
	}
	ctx.SyncThreads()
	if ctx.ThreadIdx() == 0 {
		gpu.AtomicAddF32(y, 0, s[0])
	}
}

// Float64Math is the accepted half: with the directive the double-precision
// helpers are ordinary vocabulary.
//
//gocuda:float64
func Float64Math(ctx gpu.Ctx, y []float64) {
	i := ctx.GlobalID()
	if i < len(y) {
		y[i] = gpu.Hypot64(gpu.Sqrt64(y[i]), 1)
	}
}

// Privatise is the shared-memory work of this round in one kernel: a typed
// tile, an atomic into it, and a device function with a tile of its own.
// Each is a construct the analyzer newly has to accept rather than diagnose.
func Privatise(ctx gpu.Ctx, bins, x []int32) {
	tile := ctx.SharedI32(16)
	for k := ctx.ThreadIdx(); k < 16; k += ctx.BlockDim() {
		tile[k] = 0
	}
	ctx.SyncThreads()
	i := ctx.GlobalID()
	if i < len(x) {
		gpu.AtomicAddI32(tile, int(x[i])%16, 1)
	}
	ctx.SyncThreads()
	gpu.AtomicAddI32(bins, ctx.ThreadIdx()%16, staged(ctx))
}

// staged declares shared memory inside what becomes a device function, which
// is legal CUDA -- block-scoped storage that the compiler allocates once for
// the function -- and was refused until this round.
//
//gocuda:ignore
func staged(ctx gpu.Ctx) int32 {
	scratch := ctx.SharedI64(4)
	scratch[ctx.ThreadIdx()%4] = 1
	return int32(scratch[0])
}

// Windowed takes its tile's length from the launch, so the length reaches the
// kernel as a generated parameter rather than as a constant.
func Windowed(ctx gpu.Ctx, y []float32, x []float32) {
	s := ctx.SharedDynF32()
	for i := ctx.ThreadIdx(); i < len(s); i += ctx.BlockDim() {
		s[i] = x[i%len(x)]
	}
	ctx.SyncThreads()
	y[0] = s[0]
}

// Luma is the accepted half of the narrow storage rule, and the half that
// matters: the analyzer and the transpiler have to agree that a []uint8 with
// its arithmetic done in int32 lowers, not merely that everything else is
// refused.
func Luma(ctx gpu.Ctx, out, rgb []uint8) {
	i := ctx.GlobalID()
	if i >= len(out) || 3*i+2 >= len(rgb) {
		return
	}
	v := int32(rgb[3*i]) + int32(rgb[3*i+1]) + int32(rgb[3*i+2])
	out[i] = uint8(v / 3)
}

// Weights reaches the struct shapes the offset padding was added for: an array
// field, and a narrow field that leaves a hole after it.
type Weights struct {
	N uint8
	W [3]float32
}

// Weighted lowers a struct carrying both.
func Weighted(ctx gpu.Ctx, y []float32, ws []Weights) {
	i := ctx.GlobalID()
	if i < len(y) {
		y[i] = ws[0].W[2] * float32(int32(ws[0].N))
	}
}

// WarpSum is the accepted half of the warp vocabulary. The analyzer has to
// agree with the transpiler that a warp primitive is ordinary vocabulary
// wherever a kernel may put it, device functions included; nothing else makes
// them agree.
func WarpSum(ctx gpu.Ctx, out, x []float32) {
	i := ctx.GlobalID()
	v := float32(0)
	if i < len(x) {
		v = x[i]
	}
	for off := gpu.WarpSize / 2; off > 0; off /= 2 {
		v += ctx.ShuffleDownF32(v, off)
	}
	// The vote is taken before the lane is tested, not inside the same
	// condition. Go's && short-circuits and so does the C it lowers to, so
	// `ctx.LaneID() == 0 && ctx.Any(...)` would call __any_sync on lane 0
	// alone while the emitter passes the full-warp mask -- the participation
	// promise broken by the spelling itself. The barrier-divergence check
	// works over statements and does not see into a condition, so this one is
	// avoided by hand.
	nonzero := ctx.Any(v != 0)
	if ctx.LaneID() == 0 && nonzero {
		out[i/gpu.WarpSize] = v
	}
}

// Position is the helper the //gocuda:device marker exists for: it takes a
// gpu.Ctx because it wants the thread's index, and it is not a kernel. The
// analyzer has to agree with the transpiler about that, or vet reports a
// kernel the generator never emits.
//
//gocuda:device
func Position(ctx gpu.Ctx, xs []float32) int {
	return ctx.GlobalID() % len(xs)
}

// UsesPosition reaches it, which is what turns it into a __device__ function.
func UsesPosition(ctx gpu.Ctx, y, x []float32) {
	y[Position(ctx, y)] = x[0]
}

// BlockUniformBarrier is the accepted half of the barrier-divergence rules,
// and the half that decides whether they are worth having. blockIdx is the
// same in every thread of a block, so this branch is one the whole block takes
// together; refusing it would be the rule's worst failure, because giving one
// block a job is what it is for.
func BlockUniformBarrier(ctx gpu.Ctx, y []float32, n int32) {
	s := ctx.SharedF32(4)
	if ctx.BlockIdx() == 0 && n > 0 {
		s[0] = 1
	}
	ctx.SyncThreads()
	y[0] = s[0]
}

// GuardedNotReturned is the shape the rules exist to leave alone: a trip count
// that varies, a ragged-tail guard that varies, and a barrier neither of them
// encloses. It is what FIR, Histogram and Transpose all do, so the analyzer
// has to accept it or vet reports the kernels in kernels/.
func GuardedNotReturned(ctx gpu.Ctx, y, x []float32) {
	s := ctx.SharedF32(64)
	for k := ctx.ThreadIdx(); k < 64; k += ctx.BlockDim() {
		s[k] = 0
	}
	ctx.SyncThreads()
	i := ctx.GlobalID()
	if i < len(x) {
		s[i%64] = x[i]
	}
	ctx.SyncThreads()
	if i < len(y) {
		y[i] = s[i%64]
	}
}
