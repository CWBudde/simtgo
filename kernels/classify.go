package kernels

import "github.com/CWBudde/gocuda/gpu"

// ClassifyGroup is how many edges the band search scans at a time.
const ClassifyGroup = 4

// Classify maps each sample to a gain, chosen by the first band it falls into.
//
// edges holds one ascending upper bound per band. The search is written as a
// nested scan over fixed-size groups, which is what makes the label load
// bearing: a plain break would only leave the group and the scan would carry
// on past the band it had already found.
func Classify(ctx gpu.Ctx, out, x, edges []float32) {
	i := ctx.GlobalID()
	if i >= len(out) {
		return
	}
	v := gpu.Abs(x[i])

	band := len(edges)
search:
	for base := 0; base < len(edges); base += ClassifyGroup {
		for j := base; j < base+ClassifyGroup; j++ {
			if j >= len(edges) {
				break search
			}
			if v <= edges[j] {
				band = j
				break search
			}
		}
	}

	// The gain depends on which band was found, not on where its edge fell.
	var gain float32
	switch band {
	case 0:
		gain = 0.25
	case 1, 2:
		gain = 0.5
	case 3:
		gain = 0.75
	default:
		gain = 1
	}

	// A bias drawn from the table as a whole, binding each edge rather than
	// its index.
	bias := float32(0)
	for _, e := range edges {
		bias += e
	}

	out[i] = gain*v + bias*0.001
}
