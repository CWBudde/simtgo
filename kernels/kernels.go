// Package kernels holds the example kernels.
//
// These are ordinary Go functions: they compile with the rest of the module,
// run on the CPU through gpu.RunCPU, and are lowered to CUDA C by package
// simt. Kernels may only import package gpu -- nothing else exists on the
// device.
package kernels

import "github.com/CWBudde/gocuda/gpu"

// VecAdd computes c = a + b elementwise.
func VecAdd(ctx gpu.Ctx, c, a, b []float32) {
	i := ctx.GlobalID()
	if i < len(c) {
		c[i] = a[i] + b[i]
	}
}

// Magnitude computes the magnitude of a complex spectrum held as separate
// real and imaginary parts.
func Magnitude(ctx gpu.Ctx, mag, re, im []float32) {
	i := ctx.GlobalID()
	if i < len(mag) {
		mag[i] = gpu.Hypot(re[i], im[i])
	}
}

// Scale multiplies every sample by k.
func Scale(ctx gpu.Ctx, y, x []float32, k float32) {
	i := ctx.GlobalID()
	if i < len(y) {
		y[i] = k * x[i]
	}
}
