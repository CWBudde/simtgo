package kernels

import "github.com/CWBudde/gocuda/gpu"

// FIRBlock is the block size FIR must be launched with: the shared tile is
// sized for it, and a __shared__ array needs a compile-time extent.
const FIRBlock = 256

// FIRMaxTaps is the largest filter FIR accepts.
const FIRMaxTaps = 64

// FIR is a direct-form FIR filter, y[n] = sum(h[k] * x[n-k]).
//
// Each block stages its own samples plus a halo of taps-1 preceding samples
// into shared memory, so every sample is read from global memory once instead
// of once per tap. Samples before the start of the signal read as zero.
func FIR(ctx gpu.Ctx, y, x, h []float32) {
	tile := ctx.SharedF32(FIRBlock + FIRMaxTaps)
	taps := len(h)
	base := ctx.BlockIdx() * ctx.BlockDim()
	t := ctx.ThreadIdx()

	for k := t; k < ctx.BlockDim()+taps-1; k += ctx.BlockDim() {
		src := base + k - (taps - 1)
		v := float32(0)
		if src >= 0 && src < len(x) {
			v = x[src]
		}
		tile[k] = v
	}
	ctx.SyncThreads()

	n := base + t
	if n < len(y) {
		acc := float32(0)
		for k := range h {
			acc += h[k] * tile[t+taps-1-k]
		}
		y[n] = acc
	}
}
