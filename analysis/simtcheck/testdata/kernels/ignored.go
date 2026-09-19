package kernels

import "github.com/CWBudde/simtgo/gpu"

// NotAKernel takes a Ctx but is never lowered, so it opts out.
//
//simtgo:ignore
func NotAKernel(ctx gpu.Ctx, a []float64) {
	a[0] = 1
}
