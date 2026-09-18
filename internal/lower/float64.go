package lower

import (
	"go/ast"
)

// gpuFuncs64 maps package gpu's float64 helpers to CUDA's double-precision
// built-ins, which are the unsuffixed names: sqrt rather than sqrtf.
//
// They are a separate table from gpuFuncs rather than more rows in it because
// membership here is what requires //gocuda:float64. Keeping that structural
// means the permission cannot be forgotten for a name added later, which a
// second list of "the ones that need the directive" would eventually allow.
var gpuFuncs64 = map[string]string{
	"Sqrt64":  "sqrt",
	"Abs64":   "fabs",
	"Hypot64": "hypot",
	"Sin64":   "sin",
	"Cos64":   "cos",
	"Exp64":   "exp",
	"Log64":   "log",
	"Fmin64":  "fmin",
	"Fmax64":  "fmax",
}

// float64Call lowers a double-precision helper, refusing it unless the kernel
// opted into float64.
//
// The check has to be here and not in ctype, where every other float64
// refusal lives. ctype sees a type that was written down, and a call like
// `y[i] = float32(gpu.Sqrt64(2))` declares no float64 anywhere: the argument
// is an untyped constant and the result is converted away, so the kernel would
// have reached a double without ever naming one. The permission is about what
// the device is asked to execute, so the call is where it has to be asked for.
func (t *transpiler) float64Call(c *ast.CallExpr, goName, cfn string) cexpr {
	if !t.allowFloat64 {
		t.fail(c.Pos(), "gpu.%s is double precision and needs %s on kernel %s, or on its file's package comment: the device runs double at a fraction of the float32 rate, so it is opt-in", goName, Float64Directive, t.kernelName)
		return atom("")
	}
	return atom("%s(%s)", cfn, t.args(c))
}
