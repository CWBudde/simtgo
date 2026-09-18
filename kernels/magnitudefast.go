package kernels

import "github.com/CWBudde/gocuda/gpu"

// MagnitudeFast is Magnitude with a normalisation divide, compiled with
// NVRTC's --use_fast_math.
//
// It exists to make fast math observable rather than merely available. The
// body reaches all three of the arithmetic forms the flag changes: a square
// root (--prec-sqrt), a division (--prec-div), and a multiply-add the
// compiler contracts (--fmad, which --use_fast_math leaves on). Compiling
// kernels/prebuilt/MagnitudeFast.cu with and without the flag is therefore a
// one-variable experiment, and NUMERICS.md records what it showed.
//
// The accuracy this gives up is real and is the point of the opt-in being an
// opt-in: NUMERICS.md says what a test may still assert about a kernel
// compiled this way.
//
//gocuda:fastmath
func MagnitudeFast(ctx gpu.Ctx, mag, re, im []float32, scale float32) {
	i := ctx.GlobalID()
	if i < len(mag) {
		r, m := re[i], im[i]
		mag[i] = gpu.Sqrt(r*r+m*m) / scale
	}
}
