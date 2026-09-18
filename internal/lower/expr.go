package lower

import (
	"fmt"
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
	"math"
	"strconv"
	"strings"
)

// C's operator precedence, with higher numbers binding tighter.
//
// The generated code is C, and C's table is not Go's. Go parses `a & b == c`
// as `(a & b) == c` while C reads the very same spelling as `a & (b == c)`,
// and `a << b + c` differs the same way. The parse recorded in the Go AST is
// therefore authoritative, and these levels decide how that parse has to be
// spelled in C -- which is exactly where a parenthesis is needed, and nowhere
// else.
const (
	precArg     = 2  // a call argument: everything but a bare comma operator
	precOr      = 4  // ||
	precAnd     = 5  // &&
	precBitOr   = 6  // |
	precBitXor  = 7  // ^
	precBitAnd  = 8  // &
	precEq      = 9  // == !=
	precRel     = 10 // < <= > >=
	precShift   = 11 // << >>
	precAdd     = 12 // + -
	precMul     = 13 // * / %
	precPrefix  = 14 // ! - ~ and casts
	precPostfix = 15 // a[i]
	precAtom    = 16 // identifiers, literals and calls
)

// cexpr is a rendered C expression together with the precedence of its
// outermost operator. Carrying the precedence alongside the text is what lets
// the emitter parenthesise by construction instead of guessing from the
// rendered characters.
type cexpr struct {
	s    string
	prec int
}

// at renders c in a context that requires an operand binding at least as
// tightly as min.
func (c cexpr) at(min int) string {
	if c.prec < min {
		return "(" + c.s + ")"
	}
	return c.s
}

// atom renders an expression no operator can regroup: an identifier, a literal
// or a call. It is also the shape of the empty result returned after a
// failure, so that a discarded value never grows stray parentheses.
func atom(format string, args ...any) cexpr {
	return cexpr{fmt.Sprintf(format, args...), precAtom}
}

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
//
// Each entry carries its own precedence because the spellings differ: a cast
// binds like a prefix operator, while a call cannot be regrouped at all.
var ctxBuiltins = map[string]cexpr{
	"ThreadIdx":   {"(int)threadIdx.x", precPrefix},
	"ThreadIdxY":  {"(int)threadIdx.y", precPrefix},
	"ThreadIdxZ":  {"(int)threadIdx.z", precPrefix},
	"BlockIdx":    {"(int)blockIdx.x", precPrefix},
	"BlockIdxY":   {"(int)blockIdx.y", precPrefix},
	"BlockIdxZ":   {"(int)blockIdx.z", precPrefix},
	"BlockDim":    {"(int)blockDim.x", precPrefix},
	"BlockDimY":   {"(int)blockDim.y", precPrefix},
	"BlockDimZ":   {"(int)blockDim.z", precPrefix},
	"GridDim":     {"(int)gridDim.x", precPrefix},
	"GridDimY":    {"(int)gridDim.y", precPrefix},
	"GridDimZ":    {"(int)gridDim.z", precPrefix},
	"GlobalID":    {"(int)(blockIdx.x * blockDim.x + threadIdx.x)", precPrefix},
	"GlobalIDX":   {"(int)(blockIdx.x * blockDim.x + threadIdx.x)", precPrefix},
	"GlobalIDY":   {"(int)(blockIdx.y * blockDim.y + threadIdx.y)", precPrefix},
	"GlobalIDZ":   {"(int)(blockIdx.z * blockDim.z + threadIdx.z)", precPrefix},
	"SyncThreads": {"__syncthreads()", precAtom},
}

// cBinaryPrec gives the C precedence of every binary operator the transpiler
// emits. All of them are left-associative, so the right operand is rendered
// one level tighter than the left.
var cBinaryPrec = map[token.Token]int{
	token.MUL: precMul, token.QUO: precMul, token.REM: precMul,
	token.ADD: precAdd, token.SUB: precAdd,
	token.SHL: precShift, token.SHR: precShift,
	token.LSS: precRel, token.LEQ: precRel, token.GTR: precRel, token.GEQ: precRel,
	token.EQL: precEq, token.NEQ: precEq,
	token.AND:  precBitAnd,
	token.XOR:  precBitXor,
	token.OR:   precBitOr,
	token.LAND: precAnd,
	token.LOR:  precOr,
}

func (t *transpiler) expr(e ast.Expr) cexpr {
	if t.failed() {
		return atom("")
	}
	// Anything go/types folded to a constant is emitted as a literal, which
	// covers literals, named constants and constant arithmetic alike.
	if tv, ok := t.info.Types[e]; ok && tv.Value != nil {
		return t.constant(tv, e.Pos())
	}

	switch e := e.(type) {
	case *ast.Ident:
		return t.ident(e)
	case *ast.ParenExpr:
		// Go's own grouping parentheses carry no information the AST does not
		// already have: the emitter derives every parenthesis it needs from
		// the C precedence table.
		return t.expr(e.X)
	case *ast.BinaryExpr:
		return t.binary(e)
	case *ast.UnaryExpr:
		switch e.Op {
		case token.SUB, token.ADD, token.NOT:
			return cexpr{fmt.Sprintf("%s%s", e.Op, t.expr(e.X).at(precPrefix)), precPrefix}
		case token.XOR:
			return cexpr{"~" + t.expr(e.X).at(precPrefix), precPrefix}
		}
		t.fail(e.Pos(), "unsupported unary operator %s", e.Op)
		return atom("")
	case *ast.IndexExpr:
		typ := t.typeOf(e.X)
		if typ == nil {
			return atom("")
		}
		switch typ.Underlying().(type) {
		case *types.Slice, *types.Array:
			// A Go array and a C one index identically; the difference between
			// them is what happens when the whole thing is assigned or passed,
			// which is refused elsewhere.
		default:
			t.fail(e.Pos(), "only slices and arrays can be indexed in kernels")
			return atom("")
		}
		// The subscript itself is delimited by the brackets, so it needs no
		// precedence of its own; what is indexed does.
		return cexpr{fmt.Sprintf("%s[%s]", t.expr(e.X).at(precPostfix), t.expr(e.Index).s), precPostfix}
	case *ast.SelectorExpr:
		return t.selector(e)
	case *ast.CompositeLit:
		return t.composite(e)
	case *ast.CallExpr:
		return t.call(e)
	}
	t.fail(e.Pos(), "unsupported expression %T", e)
	return atom("")
}

func (t *transpiler) binary(e *ast.BinaryExpr) cexpr {
	if e.Op == token.EQL || e.Op == token.NEQ {
		// Go compares a struct or an array field by field. C++ gives a plain
		// aggregate no operator== at all, so emitting the same spelling would
		// produce an NVRTC error about generated code nobody wrote -- and, if
		// one ever were defined, a comparison that included padding.
		if typ := t.typeOf(e.X); typ != nil {
			switch typ.Underlying().(type) {
			case *types.Struct, *types.Array:
				t.fail(e.Pos(), "%s compares %s field by field in Go, which C cannot do; compare the fields you mean", e.Op, typ)
				return atom("")
			}
		}
	}
	if p, ok := cBinaryPrec[e.Op]; ok {
		return cexpr{fmt.Sprintf("%s %s %s", t.expr(e.X).at(p), e.Op, t.expr(e.Y).at(p+1)), p}
	}
	if e.Op == token.AND_NOT { // Go's &^ has no C equivalent
		return cexpr{
			fmt.Sprintf("%s & ~%s", t.expr(e.X).at(precBitAnd), t.expr(e.Y).at(precPrefix)),
			precBitAnd,
		}
	}
	t.fail(e.Pos(), "unsupported operator %s", e.Op)
	return atom("")
}

func (t *transpiler) call(c *ast.CallExpr) cexpr {
	fun := unparen(c.Fun)

	// A conversion such as float32(x) or int(x). The operand keeps its own
	// parentheses: a cast binds tighter than every binary operator, so without
	// them `(int)a + b` would cast a alone.
	if tv, ok := t.info.Types[fun]; ok && tv.IsType() {
		if len(c.Args) != 1 {
			t.fail(c.Pos(), "unsupported conversion")
			return atom("")
		}
		t.checkNarrowing(tv.Type, c.Args[0], c.Pos())
		return cexpr{fmt.Sprintf("(%s)(%s)", t.ctype(tv.Type, c.Pos()), t.expr(c.Args[0]).s), precPrefix}
	}

	switch f := fun.(type) {
	case *ast.Ident:
		switch f.Name {
		case "len":
			if len(c.Args) != 1 {
				t.fail(c.Pos(), "len takes one argument")
				return atom("")
			}
			return t.lengthOf(c.Args[0])
		case "min", "max":
			// CUDA provides overloaded min/max for int and float.
			return atom("%s(%s)", f.Name, t.args(c))
		}
		if obj, ok := t.info.Uses[f].(*types.Func); ok {
			name, ok := t.deviceFunc(c.Pos(), obj)
			if !ok {
				return atom("")
			}
			return atom("%s(%s)", name, t.deviceArgs(c, obj))
		}
		t.fail(c.Pos(), "calls to %s are not supported in kernels", f.Name)
		return atom("")

	case *ast.SelectorExpr:
		if sel := t.info.Selections[f]; sel != nil && sel.Kind() == types.MethodVal && !IsCtx(sel.Recv()) {
			// Reported here rather than left to the catch-all below, which
			// would say "unsupported call" and nothing about why. gpu.Ctx is
			// the one receiver with device meaning; every other method would
			// need a C++ member function, and the subset has no objects.
			t.fail(c.Pos(), "methods are not supported in kernels; %s cannot be called on the device", f.Sel.Name)
			return atom("")
		}
		if sel := t.info.Selections[f]; sel != nil && sel.Kind() == types.MethodVal && IsCtx(sel.Recv()) {
			name := sel.Obj().Name()
			switch name {
			case "SharedF32":
				t.fail(c.Pos(), "SharedF32 must be assigned to a variable, e.g. `s := ctx.SharedF32(256)`")
				return atom("")
			case "AssumeBlockDim":
				return t.assumeBlockDim(c)
			}
			builtin, ok := ctxBuiltins[name]
			if !ok {
				t.fail(c.Pos(), "gpu.Ctx.%s is not available on the device", name)
				return atom("")
			}
			return builtin
		}
		if id, ok := f.X.(*ast.Ident); ok {
			if pkg, ok := t.info.Uses[id].(*types.PkgName); ok && pkg.Imported().Path() == GPUPkgPath {
				if cfn, ok := gpuAtomics[f.Sel.Name]; ok {
					return t.atomic(c, f.Sel.Name, cfn)
				}
				if cfn, ok := gpuFuncs[f.Sel.Name]; ok {
					return atom("%s(%s)", cfn, t.args(c))
				}
				if cfn, ok := gpuFuncs64[f.Sel.Name]; ok {
					return t.float64Call(c, f.Sel.Name, cfn)
				}
				t.fail(c.Pos(), "gpu.%s has no device equivalent", f.Sel.Name)
				return atom("")
			}
		}
	}
	t.fail(c.Pos(), "unsupported call")
	return atom("")
}

// deviceArgs renders a call's arguments to match the device function's C
// signature: the gpu.Ctx receiver-in-all-but-name is dropped, and every slice
// is passed as the pointer and the length the signature splits it into.
func (t *transpiler) deviceArgs(c *ast.CallExpr, obj *types.Func) string {
	fd := t.declOf(obj)
	if fd == nil {
		return ""
	}
	params := t.params(fd.Type.Params)
	args := c.Args
	if len(params) > 0 && IsCtx(params[0].typ) && len(args) > 0 {
		params, args = params[1:], args[1:]
	}
	if len(params) != len(args) {
		// go/types has already reported the mismatch; saying so again here
		// would only describe the generated code.
		return ""
	}
	out := make([]string, 0, len(args))
	for i, p := range params {
		if _, ok := p.typ.(*types.Slice); ok {
			out = append(out, t.expr(args[i]).at(precArg), t.lengthOf(args[i]).at(precArg))
			continue
		}
		out = append(out, t.expr(args[i]).at(precArg))
	}
	return strings.Join(out, ", ")
}

// assumeBlockDim records ctx.AssumeBlockDim(n) and emits nothing.
//
// The call is a statement about the launch, not device code: it tells the host
// side which block size the kernel's shared tiles were sized for, so Launch
// can refuse any other geometry instead of letting threads read past a tile.
// Because that promise has to hold for the whole kernel it may only be made
// once, unconditionally, with a size the type checker can fold.
func (t *transpiler) assumeBlockDim(c *ast.CallExpr) cexpr {
	if t.inDevice {
		t.fail(c.Pos(), "AssumeBlockDim may only be called in a kernel, not in a device function")
		return atom("")
	}
	if t.ind != 1 {
		t.fail(c.Pos(), "AssumeBlockDim must be called at the top level of the kernel body")
		return atom("")
	}
	if len(c.Args) != 1 {
		t.fail(c.Pos(), "AssumeBlockDim takes one argument")
		return atom("")
	}
	n, ok := t.constInt(c.Args[0])
	if !ok {
		t.fail(c.Pos(), "AssumeBlockDim needs a constant block size (got a runtime value)")
		return atom("")
	}
	if n <= 0 {
		t.fail(c.Pos(), "AssumeBlockDim needs a positive block size, got %d", n)
		return atom("")
	}
	if t.requiredBlock != 0 && t.requiredBlock != n {
		t.fail(c.Pos(), "AssumeBlockDim already declared a block size of %d", t.requiredBlock)
		return atom("")
	}
	t.requiredBlock = n
	return atom("")
}

// args renders a call's arguments. An argument position accepts any full
// expression -- only C's comma operator binds looser, and the transpiler never
// emits one -- so no argument needs parentheses of its own.
func (t *transpiler) args(c *ast.CallExpr) string {
	parts := make([]string, len(c.Args))
	for i, a := range c.Args {
		parts[i] = t.expr(a).at(precArg)
	}
	return strings.Join(parts, ", ")
}

// lengthOf resolves len(x). Slices lower to a pointer plus a length parameter,
// and shared buffers to a fixed-size array, so the length is always known
// without carrying a descriptor onto the device.
//
// The lookup goes through the checked object rather than the identifier's
// text, so a declaration shadowing a slice parameter resolves to its own
// length and never to the outer one's.
func (t *transpiler) lengthOf(e ast.Expr) cexpr {
	if id, ok := unparen(e).(*ast.Ident); ok {
		obj := t.info.Uses[id]
		if obj == nil {
			obj = t.info.Defs[id]
		}
		if l, ok := t.lens[obj]; ok {
			return atom("%s", l)
		}
		// A buffer whose declaration was already refused has no length
		// because of that refusal, not because len() was misused here.
		if t.poisoned[obj] {
			return atom("0")
		}
	}
	t.fail(e.Pos(), "len() is only supported for slice parameters and shared buffers")
	return atom("0")
}

// ident renders a reference to a variable.
//
// The object is resolved rather than the name simply copied, because a kernel
// runs on a device where nothing exists but its own parameters and locals. A
// package-level variable emitted verbatim would compile here and then fail
// inside NVRTC as "identifier is undefined" -- a C error about code the author
// never wrote. Constants never reach this: go/types has already folded them
// into literals.
func (t *transpiler) ident(id *ast.Ident) cexpr {
	obj := t.info.Uses[id]
	if obj == nil {
		obj = t.info.Defs[id]
	}
	switch {
	case id.Name == "_":
		// go/types resolves the blank identifier to nothing, and C has no
		// equivalent; before this it was emitted verbatim as a C identifier.
		t.fail(id.Pos(), "the blank identifier is not supported in kernels")
		return atom("")
	case obj == nil:
		t.fail(id.Pos(), "%s has no resolved type", id.Name)
		return atom("")
	case t.poisoned[obj]:
		return atom("")
	case obj.Pkg() != nil && obj.Parent() == obj.Pkg().Scope():
		t.fail(id.Pos(), "%s is declared outside the kernel; a kernel can only use its parameters, its own variables and constants", id.Name)
		return atom("")
	case obj.Parent() == types.Universe && id.Name == "nil":
		t.fail(id.Pos(), "nil has no device equivalent")
		return atom("")
	}
	return atom("%s", cname(id.Name))
}

// typeOf reports e's checked type, refusing rather than panicking when the
// type checker left none -- which cannot happen behind Transpile, but can
// behind an analysis driver that tolerates type errors.
func (t *transpiler) typeOf(e ast.Expr) types.Type {
	tv, ok := t.info.Types[e]
	if !ok || tv.Type == nil {
		t.fail(e.Pos(), "cannot determine the type of this expression")
		return nil
	}
	return tv.Type
}

func (t *transpiler) constInt(e ast.Expr) (int, bool) {
	tv, ok := t.info.Types[e]
	if !ok || tv.Value == nil {
		return 0, false
	}
	v, ok := constant.Int64Val(constant.ToInt(tv.Value))
	return int(v), ok
}

func (t *transpiler) constant(tv types.TypeAndValue, pos token.Pos) cexpr {
	basic, ok := tv.Type.Underlying().(*types.Basic)
	if !ok {
		t.fail(pos, "unsupported constant of type %s", tv.Type)
		return atom("")
	}
	switch info := basic.Info(); {
	case info&types.IsBoolean != 0:
		return atom("%s", strconv.FormatBool(constant.BoolVal(tv.Value)))
	case info&types.IsFloat != 0:
		// The width matters twice over. Rendering a float64 constant at 32 bits
		// would quietly round it, and anything past a float's range would come
		// out +Inf; the trailing f would then tell C++ to store that in a float
		// anyway. An untyped constant keeps the single-precision spelling,
		// because that is what it becomes in a kernel without the directive.
		bits, suffix := 32, "f"
		if basic.Kind() == types.Float64 {
			bits, suffix = 64, ""
		}
		f, _ := constant.Float64Val(constant.ToFloat(tv.Value))
		s := strconv.FormatFloat(f, 'g', -1, bits)
		if !strings.ContainsAny(s, ".eE") {
			s += ".0"
		}
		return number(s + suffix)
	case info&types.IsUnsigned != 0:
		// Int64Val cannot represent a uint64 above MaxInt64, and refusing one
		// as "does not fit in an int64" would be a wrong answer about a legal
		// constant rather than a refusal of anything.
		v, ok := constant.Uint64Val(constant.ToInt(tv.Value))
		if !ok {
			t.fail(pos, "constant %s does not fit in a uint64", tv.Value)
			return atom("")
		}
		return number(strconv.FormatUint(v, 10) + intSuffix(basic.Kind()))
	case info&types.IsInteger != 0:
		v, ok := constant.Int64Val(constant.ToInt(tv.Value))
		if !ok {
			t.fail(pos, "constant %s does not fit in an int64", tv.Value)
			return atom("")
		}
		if v == math.MinInt64 {
			// C++ has no literal for this value: it tokenises the positive
			// magnitude first and applies unary minus afterwards, and
			// 9223372036854775808 fits no signed type. NVRTC and nvcc both
			// accept the straightforward spelling anyway, but the generated
			// .cu is a committed artifact people compile with other tools, so
			// it is written the way the standard allows.
			return cexpr{"(-9223372036854775807ll - 1)", precAtom}
		}
		return number(strconv.FormatInt(v, 10) + intSuffix(basic.Kind()))
	}
	t.fail(pos, "unsupported constant of type %s", tv.Type)
	return atom("")
}

// checkNarrowing refuses int(x) where x is 64 bits wide.
//
// On the host that conversion is exact -- Go's int is 64 bits -- and on the
// device it truncates, because C's int is 32. So it is a conversion that means
// one thing where it is read and another where it runs, which is the shape of
// mistranslation this package exists to refuse. int32(x) is lossy in Go too, so
// an author who writes that has said what they meant.
func (t *transpiler) checkNarrowing(to types.Type, from ast.Expr, pos token.Pos) {
	dst, ok := to.Underlying().(*types.Basic)
	if !ok || (dst.Kind() != types.Int && dst.Kind() != types.Uint) {
		return
	}
	src, ok := t.typeOf(from).Underlying().(*types.Basic)
	if !ok {
		return
	}
	switch src.Kind() {
	case types.Int64, types.Uint64:
		t.fail(pos, "converting %s to %s truncates on the device, where int is 32 bits, and does not on the host; write int32(x) if that is what you mean", src, dst)
	}
}

// intSuffix is the C++ suffix that gives a literal the type the Go constant
// has. Without it C++ picks the first type the value fits in, which is the
// right answer often enough to be a trap: a small constant in a long long
// expression is an int, and the expression's type then depends on what it was
// multiplied by rather than on what the author declared.
//
// An untyped constant gets no suffix. It has no Go type to preserve, and the
// existing goldens spell it plainly.
func intSuffix(k types.BasicKind) string {
	switch k {
	case types.Int64:
		return "ll"
	case types.Uint64:
		return "ull"
	case types.Uint32:
		return "u"
	}
	return ""
}

// number renders a folded literal. A negative one is really a minus sign
// applied to a literal, so it is given prefix precedence rather than atom
// precedence: contexts that demand something binding tighter than a prefix
// operator -- what is indexed, say -- then parenthesise the sign instead of
// absorbing it.
func number(s string) cexpr {
	if strings.HasPrefix(s, "-") {
		return cexpr{s, precPrefix}
	}
	return cexpr{s, precAtom}
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
