package kernels

import "github.com/CWBudde/simtgo/gpu"

func pair() (int, int) { return 0, 1 }

// ThreeProblems must draw three diagnostics, not one: a refusal in one
// statement does not silence the rest of the kernel.
func ThreeProblems(ctx gpu.Ctx, a []float32) {
	i, j := pair()    // want `assigning 2 values from one expression`
	go func() {}()    // want `unsupported statement`
	var m map[int]int // want `unsupported type map\[int\]int`
	a[i] = float32(j + len(m))
}

// Swap is the accepted half of the same feature, and the half that makes the
// analyzer and the transpiler agree about it: a parallel assignment lowers,
// and reporting it would be a diagnostic on code that builds.
func Swap(ctx gpu.Ctx, y []float32) {
	i, j := 0, len(y)-1
	y[i], y[j] = y[j], y[i]
}
