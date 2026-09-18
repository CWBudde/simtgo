package kernels

import "github.com/CWBudde/gocuda/gpu"

// GrayShift is the number of fractional bits the luma weights are scaled by.
const GrayShift = 8

// The Rec. 601 luma weights, scaled by 1<<GrayShift and rounded so that they
// still sum to exactly 256. That is what keeps a white pixel white: 255 in
// every channel has to come back as 255 and not as 254.
const (
	GrayWeightR = 77
	GrayWeightG = 150
	GrayWeightB = 29
)

// Gray reduces an interleaved 8-bit RGB image to 8-bit luma.
//
// It is the narrow storage types in the position that asked for them. An image
// is bytes, and holding one as []int32 would quadruple both the transfer and
// the device footprint in order to store numbers that fit in a byte -- so the
// buffers are []uint8, which crosses unchanged because the widths do agree.
//
// What does not agree is arithmetic, so there is none on a uint8 here: every
// channel is converted to int32 first, the weighting happens there, and the
// result is converted back on the way into the output byte. That is the whole
// contract of a storage-only type, and it is not a formality -- Go would
// compute 77*r in 8 bits and wrap, while C would promote to int and only
// truncate at the store, which for a mid-grey pixel is two different pictures.
//
// The weights are integers rather than float32 for the usual reason an image
// pipeline uses integers: the answer is then exactly reproducible, and this
// kernel's parity test can assert equality instead of a tolerance.
func Gray(ctx gpu.Ctx, out, rgb []uint8) {
	i := ctx.GlobalID()
	if i >= len(out) {
		return
	}
	// Bounds are checked against the source as well, so a caller that passes a
	// short rgb buffer reads nothing rather than the tail of some other
	// allocation. The device has no bounds checking of its own.
	j := i * 3
	if j+2 >= len(rgb) {
		return
	}

	r := int32(rgb[j])
	g := int32(rgb[j+1])
	b := int32(rgb[j+2])
	// The rounding term is half a unit in the shifted scale, so this rounds to
	// nearest rather than truncating towards zero; all three channels are
	// non-negative, so there is no asymmetry to worry about.
	y := (GrayWeightR*r + GrayWeightG*g + GrayWeightB*b + 1<<(GrayShift-1)) >> GrayShift
	out[i] = uint8(y)
}
