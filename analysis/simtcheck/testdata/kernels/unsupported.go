package kernels

import "github.com/CWBudde/gocuda/gpu"

func Float64Slice(ctx gpu.Ctx, a []float64) { // want `unsupported type float64 on the device`
	a[0] = 1
}

func Goroutine(ctx gpu.Ctx, a []float32) {
	go func() {}() // want `unsupported statement`
	a[0] = 1
}

func helper(x float32) float32 { return x }

func CallsAGoFunction(ctx gpu.Ctx, a []float32) {
	a[0] = helper(a[1]) // want `device functions are not implemented`
}

func RuntimeSharedSize(ctx gpu.Ctx, a []float32) {
	s := ctx.SharedF32(len(a)) // want `constant size`
	s[0] = 1
}

var gain float32 = 2

func PackageLevelVar(ctx gpu.Ctx, a []float32) {
	a[0] = gain // want `declared outside the kernel`
}
