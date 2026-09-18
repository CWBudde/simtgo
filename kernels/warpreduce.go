package kernels

import "github.com/CWBudde/gocuda/gpu"

// WarpReduceBlock is the block size WarpReduceSum must be launched with.
//
// Any multiple of the warp size would do; the kernel names one so that both
// backends can refuse the launches that would not. A block that is *not* a
// whole number of warps ends in a short one, where the full participation mask
// the emitter writes names lanes that do not exist -- undefined on the device,
// and merely definite in the emulator, which is the worst kind of agreement.
const WarpReduceBlock = 128

// WarpReduceSum sums each warp's WarpSize inputs and writes one result per
// warp, for a one-dimensional launch.
//
// It is the reduction the warp vocabulary exists for: 32 values become one in
// five exchanges, with no shared memory, no __syncthreads and no round trip
// through either. Halving the distance each round leaves the total in lane 0,
// which is why lane 0 is the one that writes.
func WarpReduceSum(ctx gpu.Ctx, out, x []float32) {
	ctx.AssumeBlockDim(WarpReduceBlock)

	i := ctx.GlobalID()
	v := float32(0)
	if i < len(x) {
		v = x[i]
	}
	// Every thread of the warp reaches every exchange, including the threads
	// past the end of the input: they contribute a zero. Returning early
	// instead would leave the rest of the warp waiting at a rendezvous for a
	// lane that is not coming -- which is the contract these primitives are
	// written against, and what the emulator diagnoses.
	for off := gpu.WarpSize / 2; off > 0; off /= 2 {
		v += ctx.ShuffleDownF32(v, off)
	}
	if ctx.LaneID() == 0 {
		w := i / gpu.WarpSize
		if w < len(out) {
			out[w] = v
		}
	}
}
