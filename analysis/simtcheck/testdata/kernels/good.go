package kernels

import "github.com/CWBudde/gocuda/gpu"

// VecAdd lowers cleanly and must draw no diagnostics at all.
func VecAdd(ctx gpu.Ctx, c, a, b []float32) {
	i := ctx.GlobalID()
	if i < len(c) {
		c[i] = a[i] + b[i]
	}
}
