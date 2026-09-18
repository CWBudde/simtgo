package kernels

import "github.com/CWBudde/gocuda/gpu"

// HistogramBins is how many bins Histogram counts into, and the extent of its
// shared tile: a __shared__ array needs a compile-time extent.
const HistogramBins = 256

// Histogram counts how many samples of x fall in each bin, privatised per
// block.
//
// Every block keeps its own int32 tile of counters in shared memory, adds into
// that, and folds the tile into the global bins once at the end. The point is
// the arithmetic that is avoided: without the tile every sample would be an
// atomic on global memory, and a block of 256 threads hitting the same popular
// bin would serialise on it across the whole grid rather than only within the
// block. It is also what a shared tile and the atomics are for together, which
// nothing else here demonstrates.
//
// Samples outside [0, HistogramBins) are dropped rather than wrapped. Wrapping
// would be a second, invisible decision about somebody's data; dropping is one
// the caller can see in the totals.
func Histogram(ctx gpu.Ctx, bins, x []int32) {
	tile := ctx.SharedI32(HistogramBins)

	// Strided by the block size rather than one bin per thread, so the kernel
	// works at any block size and needs no AssumeBlockDim: the tile is sized
	// against the number of bins, which is a constant, not against how many
	// threads there happen to be.
	for k := ctx.ThreadIdx(); k < HistogramBins; k += ctx.BlockDim() {
		tile[k] = 0
	}
	ctx.SyncThreads()

	i := ctx.GlobalID()
	if i < len(x) {
		v := int(x[i])
		if v >= 0 && v < HistogramBins {
			gpu.AtomicAddI32(tile, v, 1)
		}
	}
	ctx.SyncThreads()

	for k := ctx.ThreadIdx(); k < HistogramBins; k += ctx.BlockDim() {
		c := tile[k]
		if c != 0 && k < len(bins) {
			gpu.AtomicAddI32(bins, k, c)
		}
	}
}
