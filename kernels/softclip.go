package kernels

import "github.com/CWBudde/gocuda/gpu"

// softclip is the transfer curve Softclip applies, as an ordinary Go
// function: below the threshold the signal passes through, above it the
// excess is compressed towards one so the curve stays continuous.
//
// It is here to be called, which is the point -- it lowers to a __device__
// function, and the CPU emulator simply runs it as the Go it is.
func softclip(v, threshold float32) float32 {
	a := gpu.Abs(v)
	if a <= threshold {
		return v
	}
	over := a - threshold
	shaped := threshold + over/(1+over)
	if v < 0 {
		return -shaped
	}
	return shaped
}

// Softclip applies a saturating curve to every sample.
func Softclip(ctx gpu.Ctx, y, x []float32, threshold float32) {
	i := ctx.GlobalID()
	if i < len(y) {
		y[i] = softclip(x[i], threshold)
	}
}
