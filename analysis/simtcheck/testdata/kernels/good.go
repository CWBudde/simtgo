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
