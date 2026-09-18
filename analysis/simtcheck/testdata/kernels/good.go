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
	if ctx.LaneID() == 0 && ctx.Any(v != 0) {
		out[i/gpu.WarpSize] = v
	}
}
