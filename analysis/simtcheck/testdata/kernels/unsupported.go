package kernels

import "github.com/CWBudde/gocuda/gpu"

func Float64Slice(ctx gpu.Ctx, a []float64) { // want `unsupported type float64 on the device`
	a[0] = 1
}

func Goroutine(ctx gpu.Ctx, a []float32) {
	go func() {}() // want `unsupported statement`
	a[0] = 1
}

// A kernel may call a Go function, but not one that reaches itself: there is
// no stack depth on the device to spend on recursion.
func down(x float32) float32 {
	if x > 0 {
		return down(x - 1) // want `down calls itself`
	}
	return x
}

func CallsItself(ctx gpu.Ctx, a []float32) {
	a[0] = down(a[1])
}

func RuntimeSharedSize(ctx gpu.Ctx, a []float32) {
	s := ctx.SharedF32(len(a)) // want `constant size`
	s[0] = 1
}

var gain float32 = 2

func PackageLevelVar(ctx gpu.Ctx, a []float32) {
	a[0] = gain // want `declared outside the kernel`
}
