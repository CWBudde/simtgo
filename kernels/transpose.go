package kernels

import "github.com/CWBudde/simtgo/gpu"

// TransposeTile is the edge of the square tile Transpose stages, and with it
// the block geometry the kernel demands: TransposeTile x TransposeTile
// threads.
const TransposeTile = 16

// Transpose writes the transpose of an h-by-w matrix held row-major.
//
// It is the two-dimensional kernel: one thread per element, addressed by both
// axes of the grid. The tile is what the second axis buys -- reading a block
// by rows and writing it by rows means neither access runs down a column.
func Transpose(ctx gpu.Ctx, out, in []float32, w, h int32) {
	ctx.AssumeBlockDim(TransposeTile * TransposeTile)
	tile := ctx.SharedF32(TransposeTile * TransposeTile)

	tx := ctx.ThreadIdx()
	ty := ctx.ThreadIdxY()
	x := ctx.GlobalIDX()
	y := ctx.GlobalIDY()

	if x < int(w) && y < int(h) {
		tile[ty*TransposeTile+tx] = in[y*int(w)+x]
	}
	ctx.SyncThreads()

	// The output tile is the input tile transposed, so the block's origin
	// swaps axes and the thread's offsets swap with it. Reading the tile
	// across its other diagonal is what turns a strided write into a
	// contiguous one.
	ox := ctx.BlockIdxY()*TransposeTile + tx
	oy := ctx.BlockIdx()*TransposeTile + ty
	if ox < int(h) && oy < int(w) {
		out[oy*int(h)+ox] = tile[tx*TransposeTile+ty]
	}
}
