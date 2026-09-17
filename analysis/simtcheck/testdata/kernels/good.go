package kernels

import "github.com/CWBudde/gocuda/gpu"

// VecAdd lowers cleanly and must draw no diagnostics at all.
func VecAdd(ctx gpu.Ctx, c, a, b []float32) {
	i := ctx.GlobalID()
	if i < len(c) {
		c[i] = a[i] + b[i]
	}
}

// scaled is an ordinary Go function, and a device function once a kernel
// calls it. Neither it nor its caller may draw a diagnostic.
func scaled(v, k float32) float32 { return v * k }

// Scale lowers cleanly through a device function.
func Scale(ctx gpu.Ctx, y, x []float32, k float32) {
	i := ctx.GlobalID()
	if i < len(y) {
		y[i] = scaled(x[i], k)
	}
}
