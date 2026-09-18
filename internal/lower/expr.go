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
		if typ := t.narrowOperand(e.X); typ != "" {
			t.fail(e.Pos(), "%s, so `%s` on one can give a different answer; write %sint32(x) and convert back with %s(...) when you store the result",
				narrowWhy(typ), e.Op, e.Op, typ)
			return atom("")
		}
		switch e.Op {
		case token.SUB, token.ADD, token.NOT:
			x := t.expr(e.X).at(precPrefix)
			return cexpr{e.Op.String() + unarySep(e.Op, x) + x, precPrefix}
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

// unsignedOf is the unsigned C type of the same width, for the round trip
// signedShiftLeft makes. Only the two signed types the subset allows an
// operator on are here; the narrow ones never reach an operator at all.
var unsignedOf = map[string]string{
	"int":       "unsigned int",
	"long long": "unsigned long long",
}

// signedShiftLeft emits a signed `<<` through its unsigned counterpart, which
// is the only spelling of it C defines.
//
// Go says a left shift of a signed integer wraps: int64(1) << 63 is
// MinInt64, and every bit shifted off the top is simply gone. C says a signed
// left shift whose result is not representable is *undefined*, which is not a
// wrong number but a licence for the compiler to assume it cannot happen. The
// two therefore differ on exactly the values Go defines and C does not, and
// the difference is invisible: the generated code compiles, and what it does
// depends on the optimiser.
//
// The differential fuzzer found it, on `(p ^ b*8) << 63` -- the host and the
// emulator disagreed on every element of the buffer, by whole powers of two.
//
// Casting to the unsigned type of the same width, shifting there and casting
// back is the standard way to write a defined wrapping shift, and it is what
// Go's semantics are: the conversion back is implementation-defined in C89 and
// defined as the two's-complement reinterpretation from C++20 on, which is
// what every target this emitter has runs. The right shift needs none of this,
// because it cannot overflow -- and neither do the unsigned kinds, which wrap
// by definition and are left spelled as they were.
func (t *transpiler) signedShiftLeft(e *ast.BinaryExpr) (cexpr, bool) {
	tv, ok := t.info.Types[e.X]
	if !ok || tv.Type == nil {
		return cexpr{}, false
	}
	basic, ok := tv.Type.Underlying().(*types.Basic)
	if !ok || basic.Info()&types.IsInteger == 0 || basic.Info()&types.IsUnsigned != 0 {
		return cexpr{}, false
	}
	signed := t.ctype(tv.Type, e.Pos())
	unsigned, ok := unsignedOf[signed]
	if !ok {
		return cexpr{}, false
	}
	return cexpr{fmt.Sprintf("(%s)((%s)(%s) << %s)",
		signed, unsigned, t.expr(e.X).s, t.expr(e.Y).at(cBinaryPrec[token.SHL]+1)), precPrefix}, true
}

// unarySep is the space that keeps a unary operator from fusing with its
// operand into a different token.
//
// Go's -(-c) and C's are the same expression, but the obvious spelling of it
// is "--c", which C++ lexes by maximal munch as the predecrement operator.
// Where c is a modifiable lvalue that compiles, so nothing reports it: the
// generated kernel decrements c and yields the decremented value, where the Go
// negated it twice and changed nothing. The compile error it gives on a
// prvalue -- "expression must be a modifiable lvalue" -- is the lucky half of
// the same bug.
//
// Only + and - can fuse. "!!x" is two logical nots and "~~x" two complements;
// neither pair is a token in C++, and neither needs the space.
//
// The test for it is the operand's rendered text rather than its AST, because
// what fuses is what is written: a negative constant folded to "-5", a
// parenthesised expression, and a nested unary all arrive here as strings, and
// only the first character decides.
func unarySep(op token.Token, operand string) string {
	if operand == "" {
		return ""
	}
	switch op {
	case token.SUB:
		if operand[0] == '-' {
			return " "
		}
	case token.ADD:
		if operand[0] == '+' {
			return " "
		}
	}
	return ""
}

func (t *transpiler) binary(e *ast.BinaryExpr) cexpr {
	if t.refuseNarrowBinary(e.X, e.Y, e.Op, e.Pos()) {
		return atom("")
	}
	if t.refuseWideShift(e.X, e.Y, e.Op, e.Pos()) {
		return atom("")
	}
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
	if e.Op == token.SHL {
		if c, ok := t.signedShiftLeft(e); ok {
			return c
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

// narrowOperand reports e's Go type when it is one of the storage-only narrow
// integers, and "" otherwise. It is asked at every position that would compute
// with the value rather than move it.
func (t *transpiler) narrowOperand(e ast.Expr) string {
	tv, ok := t.info.Types[e]
	if !ok || tv.Type == nil {
		return ""
	}
	if _, ok := narrowCType(tv.Type); !ok {
		return ""
	}
	return tv.Type.String()
}

// narrowWhy is the one reason every narrow-operand refusal gives. It is written
// once so that the messages cannot drift apart while describing one rule.
func narrowWhy(typ string) string {
	return fmt.Sprintf("%s is storage only on the device: Go computes %s arithmetic in its own width and C promotes it to int", typ, typ)
}

// narrowAnyway is the tail of the refusals for the shapes that, taken one at a
// time, do agree.
//
// It is worth being exact about which those are, because a message claiming a
// disagreement the reader cannot reproduce would be the same kind of thing this
// package refuses. A compound assignment and a ++ truncate straight back into
// the narrow slot, and every operator they can carry agrees under that
// truncation: the additive and bitwise ones are congruent modulo the narrow
// width, and the two that are not -- / and % -- take operands that already fit,
// so promoting them changes nothing. Comparisons agree as well, since promoting
// both sides cannot change which is larger. What does not agree is an operator
// whose result feeds another one, because the intermediate is 8 or 16 bits in
// Go and 32 in C: `a, b := int8(100), int8(3); a*b/2` is 22 in Go and -106 in
// C. There is no way to tell those apart at the operator without deciding what
// the surrounding expression may be, so the rule is every operator -- which a
// reader can hold -- instead of every operator but the ones that happen to be
// safe, which they would have to trust. Relaxing this later costs a line;
// retracting it would cost somebody a kernel that worked.
const narrowAnyway = "is refused anyway, because the subset refuses every operator on a narrow value rather than a list of the safe ones:"

// refuseNarrowBinary refuses an operator with a narrow operand on either side.
//
// Both sides, so `n << count` with a narrow count goes as well. That one is
// harmless -- a shift count is promoted without changing the result -- and it
// goes for the reason above.
func (t *transpiler) refuseNarrowBinary(x, y ast.Expr, op token.Token, pos token.Pos) bool {
	typ := t.narrowOperand(x)
	if typ == "" {
		typ = t.narrowOperand(y)
	}
	if typ == "" {
		return false
	}
	t.fail(pos, "%s, so `%s` on one can give a different answer; write int32(a) %s int32(b) and convert back with %s(...) when you store the result",
		narrowWhy(typ), op, op, typ)
	return true
}

// refuseWideShift refuses the two shifts whose answer depends on a width the
// two languages do not share.
//
// The first is the narrowing itself, caught where it escapes. Go's int is 64
// bits and the device's is 32, which SPEC.md calls the one deliberate
// infidelity and excuses on the grounds that an index is bounded by the grid
// anyway. A shift is where that stops being true: `o << (o & 31)` is computed
// in 64 bits by Go and in 32 by the device, and for o = 29 that is
// 15569256448 against -1610612736. The fuzzer wrote exactly that, fed it to an
// array index, and the two backends read different elements -- so the excuse
// had a hole in it and this is the hole closed. A *constant* shift is left
// alone: `x << 3` is what a real kernel writes, and its result is as bounded
// as x is, which is the case the excuse was actually about.
//
// The second is a constant shift the C type cannot take at all. Go defines
// x << 40 for a 32-bit x -- it is zero -- and C makes it undefined behaviour,
// so this is a wrong answer with nothing to report it. That one applies to
// every width, int32 and int64 alike, and to >> as much as to <<.
//
// What is deliberately not refused: int32 or int64 shifted by a non-constant.
// Both are the same width in both languages, so the only disagreement left is
// an amount that reaches the width at run time, and telling `x << (k & 31)` --
// which is how one writes it safely -- from `x << k` needs a range analysis
// this does not have. SPEC.md says so rather than leaving it to be found.
func (t *transpiler) refuseWideShift(x, y ast.Expr, op token.Token, pos token.Pos) bool {
	if op != token.SHL && op != token.SHR {
		return false
	}
	tv, ok := t.info.Types[x]
	if !ok || tv.Type == nil {
		return false
	}
	basic, ok := tv.Type.Underlying().(*types.Basic)
	if !ok {
		return false
	}
	// The width of the *C* type, which for Go's int is the whole point: 32,
	// not the 64 Go computes in. A kind not listed here is either refused
	// elsewhere -- the narrow integers, which no operator takes -- or not an
	// integer at all.
	var width int
	switch basic.Kind() {
	case types.Int, types.Uint, types.Int32, types.Uint32:
		width = 32
	case types.Int64, types.Uint64:
		width = 64
	default:
		return false
	}

	n, isConst := t.constInt(y)
	if !isConst {
		if op == token.SHL && (basic.Kind() == types.Int || basic.Kind() == types.Uint) {
			t.fail(pos, "%s is 64 bits in Go and 32 on the device, so `<<` by an amount this code computes can give a different answer on each; hold the value in an int32, which is 32 bits in both, or in an int64 when the high word is wanted", tv.Type)
			return true
		}
		return false
	}
	if n >= width {
		t.fail(pos, "shifting %s by %d is undefined on the device: it is %d bits there, and C leaves a shift that wide undefined where Go defines it", tv.Type, n, width)
		return true
	}
	return false
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
		// A conversion is the one place a narrow type is legal as a value,
		// which is what makes it usable at all: uint8(v) is how a computed
		// int32 gets back into a byte, and int32(b[i]) is how a byte gets out.
		// Both languages truncate a conversion to the low bits of the target,
		// so the two spellings mean the same thing -- for the signed targets
		// that is C++20's wording, and every C++ compiler NVRTC can be, as well
		// as every earlier standard in practice, already wrapped.
		cast, ok := narrowCType(tv.Type)
		if !ok {
			cast = t.ctype(tv.Type, c.Pos())
		}
		return cexpr{fmt.Sprintf("(%s)(%s)", cast, t.expr(c.Args[0]).s), precPrefix}
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
			// CUDA provides overloaded min/max for int and float. Not for the
			// narrow types: a `min(unsigned char, unsigned char)` would either
			// find no overload or silently pick the int one and hand back an
			// int, and either way it is an answer about generated code. The
			// result would happen to agree, since min of two values changes
			// neither, but it is refused for the reason the comparisons are.
			for _, a := range c.Args {
				if typ := t.narrowOperand(a); typ != "" {
					t.fail(c.Pos(), "%s, so %s has no overload for it; write %s(int32(a), int32(b)) and convert back with %s(...)",
						narrowWhy(typ), f.Name, f.Name, typ)
					return atom("")
				}
			}
			// A float operand is refused for a sharper reason: the two
			// builtins disagree about NaN. Go's min and max propagate one --
			// the specification says so -- and CUDA's fminf and fmaxf follow
			// IEEE minNum, which ignores a NaN operand and returns the number.
			// So min(0.0/0.0, x) is NaN in Go and x on the device, and the
			// same expression computes two different things with nothing to
			// report it.
			//
			// The differential fuzzer found this in six seconds of searching.
			// It had been recorded as known and untested since the numerics
			// round -- "nothing tests it and no committed kernel reaches it"
			// -- which is exactly the kind of claim a fuzzer is for.
			//
			// gpu.Fmin and gpu.Fmax are the spelling that means the device's
			// answer, and the emulator implements them to match, so the
			// refusal has somewhere to point. The integer overloads are
			// untouched: no integer is a NaN, so there is nothing to disagree
			// about.
			for _, a := range c.Args {
				if tv, ok := t.info.Types[a]; ok && tv.Type != nil {
					if b, ok := tv.Type.Underlying().(*types.Basic); ok && b.Info()&types.IsFloat != 0 {
						// gpu.Fmin / gpu.Fmin64, and the C they lower to,
						// which is fminf for a float and fmin for a double.
						fn, cfn := "Fmin", "fmin"
						if f.Name == "max" {
							fn, cfn = "Fmax", "fmax"
						}
						suffix := "f"
						if b.Kind() == types.Float64 {
							fn, suffix = fn+"64", ""
						}
						t.fail(c.Pos(), "%s propagates a NaN in Go and the device's %s%s ignores one, so the same expression computes two different things; write gpu.%s(a, b), which means the device's answer on both",
							f.Name, cfn, suffix, fn)
						return atom("")
					}
				}
			}
			return atom("%s(%s)", f.Name, t.args(c))
		}
		if obj, ok := t.info.Uses[f].(*types.Func); ok {
			name, ok := t.deviceFunc(c.Pos(), obj)
			if !ok {
				return atom("")
			}
			if decl := t.declOf(obj); decl != nil {
				// The same promise Launch checks, one level down: the callee's
				// pointers are __restrict__, so handing it one buffer twice is
				// undefined behaviour the moment it writes through either.
				if a, b, aliased := t.aliasedArgs(c, decl); aliased {
					t.fail(c.Pos(), "%s is passed the same buffer as both %s and %s, and it writes through one of them; "+
						"the generated C declares each parameter __restrict__, which promises they do not overlap",
						obj.Name(), a, b)
					return atom("")
				}
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
			if _, ok := sharedElems[name]; ok {
				// A tile is a declaration, not a value: it reaches the C as
				// the name of a __shared__ array, so there is nothing to
				// render at a call site that does not name one.
				t.fail(c.Pos(), "%s must be assigned to a variable, e.g. `s := ctx.%s(256)`", name, name)
				return atom("")
			}
			if _, ok := sharedDynElems[name]; ok {
				t.fail(c.Pos(), "%s must be assigned to a variable, e.g. `s := ctx.%s()`", name, name)
				return atom("")
			}
			if name == "AssumeBlockDim" {
				return t.assumeBlockDim(c)
			}
			// The warp-level vocabulary is a table of its own: it takes
			// arguments, and the mask CUDA wants in front of them is the
			// emitter's rather than the kernel's.
			if op, ok := ctxWarp[name]; ok {
				return t.warp(c, name, op)
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
	return atom("%s", t.cnameOf(obj, id.Name))
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
