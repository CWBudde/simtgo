package kernels

import "github.com/CWBudde/gocuda/gpu"

// A Band is one entry of a gain curve: everything at or below Upper takes Gain.
//
// Two float32 fields, so it is 8 bytes in Go and 8 bytes in CUDA. The generated
// C says so with a static_assert rather than trusting it.
type Band struct {
	Upper float32
	Gain  float32
}

// Shape is the part of the curve that does not vary per band.
//
// The field order is deliberate: Floor at 0, Count at 4, Bias at 8, for a size
// of 16 and an alignment of 8. Mixed widths are the point -- an all-float32
// struct has only one layout either language could give it -- and the holes
// that Go leaves in a struct like this one are what the generated padding
// declares, so that sizeof pins every offset. This particular order happens to
// leave none: 4 and 4 fill the first eight bytes exactly. An earlier comment
// here claimed four bytes of padding before Bias, which was never true.
//
// Count is an int32 and not an int. An int field would be 8 bytes in Go and 4
// in CUDA, and cuda.Upload copies Go's layout, so it is refused.
type Shape struct {
	Floor float32
	Count int32
	Bias  float64
}

// BandGainTaps is the length of the smoothing window BandGain applies.
const BandGainTaps = 4

// BandGain applies a piecewise-constant gain curve, smoothed over a few taps.
//
// The accumulation is float64 because a gain curve multiplies numbers of very
// different magnitudes, and the point of the directive is that paying for that
// is a decision somebody wrote down rather than something that happened.
//
//gocuda:float64
func BandGain(ctx gpu.Ctx, y, x []float32, bands []Band, cfg Shape) {
	i := ctx.GlobalID()
	if i >= len(y) {
		return
	}

	v := x[i]
	// A literal with a field left off, which lowers positionally: C++17 has no
	// designated initialisers. Falling off the end of the curve means the floor
	// gain, so that is what the unmatched case already holds.
	best := Band{Gain: cfg.Floor}
	for _, b := range bands {
		if v <= b.Upper {
			best = b
			break
		}
	}

	// A fixed-size array as a local: declared, indexed and ranged over, which
	// is everything an array is allowed to be.
	var taps [BandGainTaps]float32
	for j := range BandGainTaps {
		k := i + j - BandGainTaps/2
		if k >= 0 && k < len(x) {
			taps[j] = x[k]
		}
	}
	var sum float32
	for _, tv := range taps {
		sum += tv
	}

	acc := float64(sum)/float64(BandGainTaps)*float64(best.Gain) + cfg.Bias
	y[i] = float32(acc) * float32(cfg.Count)
}
