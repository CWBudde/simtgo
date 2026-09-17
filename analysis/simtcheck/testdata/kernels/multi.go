package kernels

import "github.com/CWBudde/gocuda/gpu"

// ThreeProblems must draw three diagnostics, not one: a refusal in one
// statement does not silence the rest of the kernel.
func ThreeProblems(ctx gpu.Ctx, a []float32) {
	i, j := 0, 1      // want `multiple assignment`
	go func() {}()    // want `unsupported statement`
	var m map[int]int // want `unsupported type map\[int\]int`
	a[i] = float32(j + len(m))
}
