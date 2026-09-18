package kernels

import "github.com/CWBudde/gocuda/gpu"

// Quantize rounds each sample to a step, and reports what that cost.
//
// It is the integer half of the widened type map. The code is an int32, because
// that is what a quantiser emits; the energy is an int64, because the square of
// an int32 code is not an int32 and saying so is the honest reason to want a
// 64-bit type at all; and clipped is a []bool, which lowered all along with
// nothing to prove it did.
//
// seed is a uint32 so the dither is written in unsigned arithmetic. Go and C
// agree that unsigned wraps, which is what makes a hash like this portable, and
// the kernel is here partly to keep them agreeing.
func Quantize(ctx gpu.Ctx, code []int32, energy []int64, clipped []bool, x []float32, step float32, seed uint32) {
	i := ctx.GlobalID()
	if i >= len(code) {
		return
	}

	// A cheap per-sample dither, so the rounding does not bias a constant
	// input. Every operation here wraps, deliberately.
	h := seed ^ uint32(i)
	h = h*2654435761 + 1
	h ^= h >> 15
	jitter := float32(int32(h%1024))/1024 - 0.5

	q := int32((x[i]/step + jitter))
	if q > 127 {
		q = 127
		clipped[i] = true
	} else if q < -128 {
		q = -128
		clipped[i] = true
	} else {
		clipped[i] = false
	}

	code[i] = q
	// int64(q)*int64(q) is the point: at q = 127 this is 16129, which fits an
	// int32, but the accumulation a caller does with it does not, and the
	// conversion has to happen before the multiply rather than after.
	energy[i] = int64(q) * int64(q)
}
