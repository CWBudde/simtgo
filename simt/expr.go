package simt

import (
	"fmt"
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
	"strconv"
	"strings"
)

// gpuFuncs maps package gpu's float32 helpers to CUDA's single-precision
// built-ins. NVRTC provides these without any header.
var gpuFuncs = map[string]string{
	"Sqrt":  "sqrtf",
	"Abs":   "fabsf",
	"Hypot": "hypotf",
	"Sin":   "sinf",
	"Cos":   "cosf",
	"Exp":   "expf",
	"Log":   "logf",
	"Fmin":  "fminf",
	"Fmax":  "fmaxf",
}

// ctxBuiltins maps gpu.Ctx methods to CUDA built-in variables. The casts keep
// the generated arithmetic signed: CUDA's thread indices are unsigned, and
// mixing them into Go's int expressions would silently change comparisons.
var ctxBuiltins = map[string]string{
	"ThreadIdx":   "(int)threadIdx.x",
	"BlockIdx":    "(int)blockIdx.x",
	"BlockDim":    "(int)blockDim.x",
	"GridDim":     "(int)gridDim.x",
	"GlobalID":    "(int)(blockIdx.x * blockDim.x + threadIdx.x)",
	"SyncThreads": "__syncthreads()",
}

func (t *transpiler) expr(e ast.Expr) string {
	if t.err != nil {
		return ""
	}
	// Anything go/types folded to a constant is emitted as a literal, which
	// covers literals, named constants and constant arithmetic alike.
	if tv, ok := t.info.Types[e]; ok && tv.Value != nil {
		return t.constant(tv, e.Pos())
	}

	switch e := e.(type) {
	case *ast.Ident:
		return cname(e.Name)
	case *ast.ParenExpr:
		// Binary expressions already parenthesise themselves, so Go's own
		// grouping parentheses would only add noise.
		return t.expr(e.X)
	case *ast.BinaryExpr:
		return t.binary(e)
	case *ast.UnaryExpr:
		switch e.Op {
		case token.SUB, token.ADD, token.NOT:
			return fmt.Sprintf("%s%s", e.Op, t.expr(e.X))
		case token.XOR:
			return "~" + t.expr(e.X)
		}
		t.fail(e.Pos(), "unsupported unary operator %s", e.Op)
		return ""
	case *ast.IndexExpr:
		if _, ok := t.info.Types[e.X].Type.Underlying().(*types.Slice); !ok {
			t.fail(e.Pos(), "only slices can be indexed in kernels")
			return ""
		}
		return fmt.Sprintf("%s[%s]", t.expr(e.X), t.expr(e.Index))
	case *ast.CallExpr:
		return t.call(e)
	}
	t.fail(e.Pos(), "unsupported expression %T", e)
	return ""
}

func (t *transpiler) binary(e *ast.BinaryExpr) string {
	switch e.Op {
	case token.ADD, token.SUB, token.MUL, token.QUO, token.REM,
		token.EQL, token.NEQ, token.LSS, token.LEQ, token.GTR, token.GEQ,
		token.LAND, token.LOR, token.AND, token.OR, token.XOR,
		token.SHL, token.SHR:
		return fmt.Sprintf("(%s %s %s)", t.expr(e.X), e.Op, t.expr(e.Y))
	case token.AND_NOT: // Go's &^ has no C equivalent
		return fmt.Sprintf("(%s & ~%s)", t.expr(e.X), t.expr(e.Y))
	}
	t.fail(e.Pos(), "unsupported operator %s", e.Op)
	return ""
}

func (t *transpiler) call(c *ast.CallExpr) string {
	fun := unparen(c.Fun)

	// A conversion such as float32(x) or int(x).
	if tv, ok := t.info.Types[fun]; ok && tv.IsType() {
		if len(c.Args) != 1 {
			t.fail(c.Pos(), "unsupported conversion")
			return ""
		}
		return fmt.Sprintf("(%s)(%s)", t.ctype(tv.Type, c.Pos()), t.expr(c.Args[0]))
	}

	switch f := fun.(type) {
	case *ast.Ident:
		switch f.Name {
		case "len":
			if len(c.Args) != 1 {
				t.fail(c.Pos(), "len takes one argument")
				return ""
			}
			return t.lengthOf(c.Args[0])
		case "min", "max":
			// CUDA provides overloaded min/max for int and float.
			return fmt.Sprintf("%s(%s)", f.Name, t.args(c))
		}
		t.fail(c.Pos(), "calls to %s are not supported in kernels (device functions are not implemented)", f.Name)
		return ""

	case *ast.SelectorExpr:
		if sel := t.info.Selections[f]; sel != nil && sel.Kind() == types.MethodVal && t.isCtx(sel.Recv()) {
			name := sel.Obj().Name()
			if name == "SharedF32" {
				t.fail(c.Pos(), "SharedF32 must be assigned to a variable, e.g. `s := ctx.SharedF32(256)`")
				return ""
			}
			builtin, ok := ctxBuiltins[name]
			if !ok {
				t.fail(c.Pos(), "gpu.Ctx.%s is not available on the device", name)
				return ""
			}
			return builtin
		}
		if id, ok := f.X.(*ast.Ident); ok {
			if pkg, ok := t.info.Uses[id].(*types.PkgName); ok && pkg.Imported().Path() == GPUPkgPath {
				if cfn, ok := gpuFuncs[f.Sel.Name]; ok {
					return fmt.Sprintf("%s(%s)", cfn, t.args(c))
				}
				t.fail(c.Pos(), "gpu.%s has no device equivalent", f.Sel.Name)
				return ""
			}
		}
	}
	t.fail(c.Pos(), "unsupported call")
	return ""
}

func (t *transpiler) args(c *ast.CallExpr) string {
	parts := make([]string, len(c.Args))
	for i, a := range c.Args {
		parts[i] = t.expr(a)
	}
	return strings.Join(parts, ", ")
}

// lengthOf resolves len(x). Slices lower to a pointer plus a length parameter,
// and shared buffers to a fixed-size array, so the length is always known
// without carrying a descriptor onto the device.
func (t *transpiler) lengthOf(e ast.Expr) string {
	if id, ok := unparen(e).(*ast.Ident); ok {
		if l, ok := t.lens[id.Name]; ok {
			return l
		}
	}
	t.fail(e.Pos(), "len() is only supported for slice parameters and shared buffers")
	return "0"
}

func (t *transpiler) constInt(e ast.Expr) (int, bool) {
	tv, ok := t.info.Types[e]
	if !ok || tv.Value == nil {
		return 0, false
	}
	v, ok := constant.Int64Val(constant.ToInt(tv.Value))
	return int(v), ok
}

func (t *transpiler) constant(tv types.TypeAndValue, pos token.Pos) string {
	basic, ok := tv.Type.Underlying().(*types.Basic)
	if !ok {
		t.fail(pos, "unsupported constant of type %s", tv.Type)
		return ""
	}
	switch info := basic.Info(); {
	case info&types.IsBoolean != 0:
		return strconv.FormatBool(constant.BoolVal(tv.Value))
	case info&types.IsFloat != 0:
		f, _ := constant.Float64Val(constant.ToFloat(tv.Value))
		s := strconv.FormatFloat(f, 'g', -1, 32)
		if !strings.ContainsAny(s, ".eE") {
			s += ".0"
		}
		return s + "f"
	case info&types.IsInteger != 0:
		v, ok := constant.Int64Val(constant.ToInt(tv.Value))
		if !ok {
			t.fail(pos, "constant %s does not fit in an int64", tv.Value)
			return ""
		}
		return strconv.FormatInt(v, 10)
	}
	t.fail(pos, "unsupported constant of type %s", tv.Type)
	return ""
}

// paren wraps s in parentheses unless it already is a parenthesised group,
// which keeps `if (i < n)` from being emitted as `if ((i < n))`.
func paren(s string) string {
	if len(s) > 1 && s[0] == '(' && s[len(s)-1] == ')' {
		depth := 0
		for i, c := range s {
			switch c {
			case '(':
				depth++
			case ')':
				depth--
				if depth == 0 && i != len(s)-1 {
					return "(" + s + ")"
				}
			}
		}
		return s
	}
	return "(" + s + ")"
}

func unparen(e ast.Expr) ast.Expr {
	for {
		p, ok := e.(*ast.ParenExpr)
		if !ok {
			return e
		}
		e = p.X
	}
}

// cppKeywords are valid Go identifiers that would be syntax errors in CUDA C++.
var cppKeywords = map[string]bool{
	"auto": true, "class": true, "const": true, "delete": true, "double": true,
	"extern": true, "float": true, "friend": true, "inline": true, "int": true,
	"long": true, "new": true, "operator": true, "private": true, "protected": true,
	"public": true, "register": true, "short": true, "signed": true, "sizeof": true,
	"static": true, "template": true, "this": true, "throw": true, "try": true,
	"typedef": true, "union": true, "unsigned": true, "virtual": true, "void": true,
	"volatile": true, "namespace": true, "using": true, "bool": true, "char": true,
}

func cname(name string) string {
	if cppKeywords[name] {
		return name + "_"
	}
	return name
}
