package fuzz

import (
	"fmt"
	"go/format"
	"go/token"
	"strconv"
	"strings"
)

// Source renders the program as the Go text simt.Transpile is handed.
//
// The parentheses are minimal, and that is the point of the whole renderer:
// Go and C disagree about how tightly the shifts and the bitwise operators
// bind, and the emitter is correct only because it re-derives its grouping
// from C's table. A fully parenthesised source would hand it the answer and
// test nothing, so this writes exactly the parentheses Go's own precedence
// requires and no more.
//
// The result is run through go/format, which cannot change what the text
// means. A program whose text does not parse is returned unformatted rather
// than swallowed, so the test that reads it sees the broken source instead of
// an error about it.
func (p *Program) Source() string {
	var w srcWriter
	w.line("package %s", p.Pkg)
	w.blank()
	w.line("import %q", gpuImportPath)
	for _, sh := range p.Shapes {
		// Declared with the same fields in the same order as the Go type in
		// structs.go, so that Go computes one layout for both and the emitter's
		// padding has something true to be checked against.
		w.blank()
		w.line("type %s struct {", sh.Name)
		for _, f := range sh.Fields {
			w.line("\t%s %s", f.Name, f.Kind.goName())
		}
		w.line("}")
	}
	for _, fn := range p.Funcs {
		w.blank()
		p.writeFunc(&w, fn)
	}
	src := w.b.String()
	out, err := format.Source([]byte(src))
	if err != nil {
		return src
	}
	return string(out)
}

// gpuImportPath is the only import a kernel package may have.
const gpuImportPath = "github.com/CWBudde/gocuda/gpu"

// srcWriter accumulates Go text at an indentation.
type srcWriter struct {
	b     strings.Builder
	depth int
}

func (w *srcWriter) line(format string, args ...any) {
	w.b.WriteString(strings.Repeat("\t", w.depth))
	fmt.Fprintf(&w.b, format, args...)
	w.b.WriteByte('\n')
}

func (w *srcWriter) blank() { w.b.WriteByte('\n') }

func (w *srcWriter) in()  { w.depth++ }
func (w *srcWriter) out() { w.depth-- }

// writeFunc renders one declaration, directives included.
//
// //gocuda:float64 belongs on the kernel and nowhere else: it is a promise
// about what a launch costs, so it covers the whole translation unit including
// every helper the kernel reaches, and a helper carrying its own is refused.
func (p *Program) writeFunc(w *srcWriter, fn *Func) {
	if fn.Kernel && p.Float64 {
		w.line("//gocuda:float64")
	}
	if fn.Device {
		w.line("//gocuda:device")
	}
	var params []string
	if fn.Ctx {
		params = append(params, "ctx gpu.Ctx")
	}
	for _, v := range fn.Params {
		params = append(params, v.Name+" "+paramType(v))
	}
	result := ""
	if fn.Result != KInvalid {
		result = " " + fn.Result.goName()
	}
	w.line("func %s(%s)%s {", fn.Name, strings.Join(params, ", "), result)
	w.in()
	p.writeStmts(w, fn.Body)
	w.out()
	w.line("}")
}

// paramType is how a parameter's type is spelled. An array is not among them:
// Go passes one by value and C decays it to a pointer, so a write inside the
// function would reach the caller's array, and the subset refuses it.
func paramType(v *Var) string {
	if v.Slice {
		if v.Shape != nil {
			return "[]" + v.Shape.Name
		}
		return "[]" + v.Kind.goName()
	}
	return v.Kind.goName()
}

func (p *Program) writeStmts(w *srcWriter, body []Stmt) {
	for _, s := range body {
		p.writeStmt(w, s)
	}
}

func (p *Program) writeStmt(w *srcWriter, s Stmt) {
	switch s := s.(type) {
	case *Decl:
		switch s.Form {
		case DeclShort:
			w.line("%s := %s", s.V.Name, expand(s.Init))
		case DeclVarTyped:
			w.line("var %s %s = %s", s.V.Name, s.V.Kind.goName(), expand(s.Init))
		case DeclVarZero:
			w.line("var %s %s", s.V.Name, s.V.Kind.goName())
		}
	case *ArrayDecl:
		w.line("var %s [%d]%s", s.V.Name, s.V.Array, s.V.Kind.goName())
	case *Assign:
		w.line("%s %s %s", lvalueText(s.LHS), s.Op.String(), expand(s.RHS))
	case *ParAssign:
		lhs := make([]string, len(s.LHS))
		for i, l := range s.LHS {
			lhs[i] = lvalueText(l)
		}
		w.line("%s = %s", strings.Join(lhs, ", "), argsText(s.RHS))
	case *IncDec:
		w.line("%s%s", lvalueText(s.LHS), s.Op.String())
	case *If:
		p.writeIf(w, s, "")
	case *For:
		p.writeFor(w, s)
	case *Range:
		p.writeRange(w, s)
	case *Switch:
		p.writeSwitch(w, s)
	case *Branch:
		if s.Label == "" {
			w.line("%s", s.Tok.String())
		} else {
			w.line("%s %s", s.Tok.String(), s.Label)
		}
	case *Return:
		if s.X == nil {
			w.line("return")
		} else {
			w.line("return %s", expand(s.X))
		}
	case *Barrier:
		w.line("ctx.SyncThreads()")
	case *SyncWarp:
		w.line("ctx.SyncWarp()")
	case *SharedDecl:
		if s.Dyn {
			w.line("%s := ctx.SharedDyn%s()", s.V.Name, sharedSuffix(s.V.Kind))
		} else {
			w.line("%s := ctx.Shared%s(%d)", s.V.Name, sharedSuffix(s.V.Kind), s.N)
		}
	case *Assume:
		w.line("ctx.AssumeBlockDim(%d)", s.N)
	case *CallStmt:
		w.line("%s", callText(s.Fn, s.Args))
	case *Discard:
		w.line("%s", expand(s.X))
	default:
		panic(fmt.Sprintf("fuzz: no Go rendering for %T", s))
	}
}

// writeIf renders an if, folding `else { if ... }` back into `else if` when
// that is the shape the generator built. The two parse differently and the
// transpiler walks the AST, so which one is emitted is worth controlling.
func (p *Program) writeIf(w *srcWriter, s *If, prefix string) {
	w.line("%sif %s {", prefix, expand(s.Cond))
	w.in()
	p.writeStmts(w, s.Then)
	w.out()
	if len(s.Else) == 0 {
		w.line("}")
		return
	}
	if len(s.Else) == 1 {
		if inner, ok := s.Else[0].(*If); ok {
			p.writeIf(w, inner, "} else ")
			return
		}
	}
	w.line("} else {")
	w.in()
	p.writeStmts(w, s.Else)
	w.out()
	w.line("}")
}

func (p *Program) writeFor(w *srcWriter, s *For) {
	if s.Label != "" {
		w.line("%s:", s.Label)
	}
	head := "for "
	switch {
	case s.Init == nil && s.Post == nil && s.Cond != nil:
		head += expand(s.Cond)
	default:
		head += simpleText(s.Init) + "; " + condText(s.Cond) + "; " + simpleText(s.Post)
	}
	w.line("%s {", head)
	w.in()
	p.writeStmts(w, s.Body)
	w.out()
	w.line("}")
}

func (p *Program) writeRange(w *srcWriter, s *Range) {
	if s.Label != "" {
		w.line("%s:", s.Label)
	}
	names := s.Key.Name
	if s.Val != nil {
		names += ", " + s.Val.Name
	}
	var over string
	switch {
	case s.Over.Buf != nil:
		over = s.Over.Buf.Name
	default:
		over = expand(s.Over.N)
	}
	w.line("for %s := range %s {", names, over)
	w.in()
	p.writeStmts(w, s.Body)
	w.out()
	w.line("}")
}

func (p *Program) writeSwitch(w *srcWriter, s *Switch) {
	if s.Tag == nil {
		w.line("switch {")
	} else {
		w.line("switch %s {", expand(s.Tag))
	}
	for _, c := range s.Cases {
		if len(c.Vals) == 0 {
			w.line("default:")
		} else {
			vals := make([]string, len(c.Vals))
			for i, v := range c.Vals {
				vals[i] = expand(v)
			}
			w.line("case %s:", strings.Join(vals, ", "))
		}
		w.in()
		p.writeStmts(w, c.Body)
		if c.Fallthrough {
			w.line("fallthrough")
		}
		w.out()
	}
	w.line("}")
}

// simpleText renders the statement forms a for clause accepts. A parallel
// assignment is not among them: the temporaries it needs would have to be
// declarations, and C's comma operator carries only expressions.
func simpleText(s Stmt) string {
	switch s := s.(type) {
	case nil:
		return ""
	case *Decl:
		return s.V.Name + " := " + expand(s.Init)
	case *Assign:
		return lvalueText(s.LHS) + " " + s.Op.String() + " " + expand(s.RHS)
	case *IncDec:
		return lvalueText(s.LHS) + s.Op.String()
	}
	panic(fmt.Sprintf("fuzz: %T is not a simple statement", s))
}

func condText(e Expr) string {
	if e == nil {
		return ""
	}
	return expand(e)
}

func lvalueText(l Lvalue) string {
	if l.Idx == nil {
		return l.V.Name
	}
	if l.V.Shape != nil {
		return l.V.Name + "[" + expand(l.Idx) + "]." + l.V.Shape.Fields[l.F].Name
	}
	return l.V.Name + "[" + expand(l.Idx) + "]"
}

// sharedSuffix is the constructor suffix for a tile of this element type.
// There is deliberately no bool tile: nothing here reads or writes a bool
// atomically, so one could only be filled under a barrier, which the int32
// tile already does.
func sharedSuffix(k Kind) string {
	switch k {
	case KF32:
		return "F32"
	case KF64:
		return "F64"
	case KI32:
		return "I32"
	case KI64:
		return "I64"
	case KU32:
		return "U32"
	case KU64:
		return "U64"
	}
	panic("fuzz: no shared tile for " + k.goName())
}

// expand renders an expression at the outermost precedence, where nothing
// needs wrapping.
func expand(e Expr) string { return exprText(e, token.LowestPrec) }

// exprText renders e, parenthesising it when the context binds tighter than
// its own operator. minPrec is the precedence at or below which a binary node
// has to be wrapped; a unary operand passes unaryPrec, which is above every
// binary level, so any binary child of a unary operator is wrapped.
func exprText(e Expr, minPrec int) string {
	switch e := e.(type) {
	case *Lit:
		return litText(e)
	case *Ref:
		return e.V.Name
	case *Index:
		return e.Base.Name + "[" + expand(e.Idx) + "]"
	case *Field:
		return e.Base.Name + "[" + expand(e.Idx) + "]." + e.Base.Shape.Fields[e.F].Name
	case *Len:
		return "len(" + e.Base.Name + ")"
	case *Binary:
		prec := e.Op.Precedence()
		// Go's binary operators are all left-associative, so an equal-level
		// node is transparent on the left and has to be wrapped on the right.
		s := exprText(e.X, prec) + " " + e.Op.String() + " " + exprText(e.Y, prec+1)
		if prec < minPrec {
			return "(" + s + ")"
		}
		return s
	case *Unary:
		x := exprText(e.X, unaryPrec)
		if _, nested := e.X.(*Unary); nested {
			// `- -x` and `--x` are different tokens; the grouping costs the
			// precedence test nothing, since a unary operand is a leaf as far
			// as C's table is concerned.
			x = "(" + x + ")"
		}
		return e.Op.String() + x
	case *Conv:
		return e.To.goName() + "(" + expand(e.X) + ")"
	case *Pos:
		return "ctx." + e.Method + "()"
	case *MathCall:
		return "gpu." + e.Fn + "(" + argsText(e.Args) + ")"
	case *MinMax:
		return e.Fn + "(" + argsText(e.Args) + ")"
	case *WarpCall:
		return "ctx." + e.Method + "(" + argsText(e.Args) + ")"
	case *AtomicCall:
		args := append([]string{e.Buf.Name, expand(e.Idx)}, argStrings(e.Args)...)
		return "gpu." + e.Fn + "(" + strings.Join(args, ", ") + ")"
	case *CallExpr:
		return callText(e.Fn, e.Args)
	}
	panic(fmt.Sprintf("fuzz: no Go rendering for %T", e))
}

// unaryPrec is above every binary precedence, which is what makes a binary
// operand of a unary operator always parenthesised.
const unaryPrec = token.HighestPrec + 1

func callText(fn *Func, args []Expr) string {
	var parts []string
	if fn.Ctx {
		parts = append(parts, "ctx")
	}
	parts = append(parts, argStrings(args)...)
	return fn.Name + "(" + strings.Join(parts, ", ") + ")"
}

func argsText(args []Expr) string { return strings.Join(argStrings(args), ", ") }

func argStrings(args []Expr) []string {
	out := make([]string, len(args))
	for i, a := range args {
		out[i] = expand(a)
	}
	return out
}

// litText renders a constant. A negative one is parenthesised: it is a leaf,
// so the grouping costs the precedence test nothing, and it keeps a minus sign
// from running into a neighbouring operator.
func litText(l *Lit) string {
	switch {
	case l.K == KBool:
		return strconv.FormatBool(l.B)
	case l.K.float():
		bits := 64
		if l.K == KF32 {
			bits = 32
		}
		s := strconv.FormatFloat(l.F, 'g', -1, bits)
		if !strings.ContainsAny(s, ".eE") {
			// An integral value would otherwise be an untyped integer
			// constant, which is legal here but reads as an int; spelling the
			// point keeps the source saying what kind the node holds.
			s += ".0"
		}
		if l.F < 0 {
			return "(" + s + ")"
		}
		return s
	case l.K == KU32 || l.K == KU64:
		return strconv.FormatUint(l.U, 10)
	default:
		s := strconv.FormatInt(l.I, 10)
		if l.I < 0 {
			return "(" + s + ")"
		}
		return s
	}
}
