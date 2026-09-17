package lower

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"strconv"
)

func (t *transpiler) block(b *ast.BlockStmt) {
	t.line("{")
	t.ind++
	for _, s := range b.List {
		// Each statement is refused on its own: mark resets the window that
		// failed() reports on, so a refusal here does not silence the rest of
		// the kernel. The braces and t.ind stay balanced because nothing
		// returns early any more -- which matters, since t.ind is what
		// SharedF32 and AssumeBlockDim test to insist on the top level.
		t.mark = len(t.diags)
		t.stmt(s)
	}
	t.ind--
	t.line("}")
}

func (t *transpiler) stmt(s ast.Stmt) {
	switch s := s.(type) {
	case *ast.AssignStmt:
		t.assign(s)
	case *ast.IncDecStmt:
		t.line("%s;", t.simple(s))
	case *ast.ExprStmt:
		// ctx.AssumeBlockDim(n) describes the launch rather than the device
		// code and renders as nothing at all; a bare `;` line would be the
		// only trace of it in the generated source.
		if c := t.expr(s.X); c.s != "" {
			t.line("%s;", c.s)
		}
	case *ast.DeclStmt:
		t.decl(s)
	case *ast.IfStmt:
		t.ifStmt(s)
	case *ast.ForStmt:
		t.forStmt(s)
	case *ast.RangeStmt:
		t.rangeStmt(s)
	case *ast.BlockStmt:
		t.block(s)
	case *ast.BranchStmt:
		switch s.Tok {
		case token.BREAK, token.CONTINUE:
			t.line("%s;", s.Tok)
		default:
			t.fail(s.Pos(), "%s is not supported in kernels", s.Tok)
		}
	case *ast.ReturnStmt:
		if len(s.Results) > 0 {
			t.fail(s.Pos(), "a kernel cannot return a value")
			return
		}
		t.line("return;")
	case *ast.EmptyStmt:
	default:
		t.fail(s.Pos(), "unsupported statement %T", s)
	}
}

func (t *transpiler) assign(s *ast.AssignStmt) {
	if len(s.Lhs) != 1 || len(s.Rhs) != 1 {
		t.fail(s.Pos(), "multiple assignment is not supported in kernels")
		return
	}
	lhs, rhs := s.Lhs[0], s.Rhs[0]

	if s.Tok == token.DEFINE {
		id, ok := lhs.(*ast.Ident)
		if !ok {
			t.fail(s.Pos(), "unsupported declaration target %T", lhs)
			return
		}
		if n, isShared, ok := t.sharedSize(rhs); isShared {
			if ok {
				t.shared(id, n, s.Pos())
			} else {
				t.poison(t.info.Defs[id])
			}
			return
		}
		t.line("%s;", t.define(id, rhs))
		return
	}
	t.line("%s;", t.simple(s))
}

// define renders a `:=` declaration as `T name = rhs`, without a terminator.
// Statements and for-clauses both need it, and resolving the declared object
// in one place keeps the two from deriving the C type differently.
//
// The declared type is mapped before the initialiser is rendered, so a
// variable of a type the device has no answer for is reported as exactly that
// rather than through whatever its initialiser happens to be.
func (t *transpiler) define(id *ast.Ident, rhs ast.Expr) string {
	obj := t.info.Defs[id]
	if obj == nil {
		t.fail(id.Pos(), "%s has no resolved type", id.Name)
		return ""
	}
	before := len(t.diags)
	ctype := t.ctype(obj.Type(), id.Pos())
	if len(t.diags) > before {
		// The variable exists but has no device type. Every later use of it
		// would be a fresh complaint about the same declaration.
		t.poison(obj)
	}
	return fmt.Sprintf("%s %s = %s", ctype, cname(id.Name), t.expr(rhs).s)
}

// shared declares a block's __shared__ tile of n float32 values.
func (t *transpiler) shared(id *ast.Ident, n int, pos token.Pos) {
	if t.ind != 1 {
		t.fail(pos, "shared memory must be declared at the top level of the kernel")
		return
	}
	obj := t.info.Defs[id]
	if obj == nil {
		t.fail(id.Pos(), "%s has no resolved type", id.Name)
		return
	}
	t.line("__shared__ float %s[%d];", cname(id.Name), n)
	t.lens[obj] = strconv.Itoa(n)
	// Every block gets its own copy of the tile, so the running total is what
	// each block will ask the device for. SharedF32 is float32-only, hence the
	// fixed element size.
	t.sharedBytes += 4 * n
}

// sharedSize reports whether e is a ctx.SharedF32(n) call, and if so whether n
// folded to a constant.
//
// The two answers have to be separate. "Not a shared buffer" sends the caller
// down the ordinary declaration path; "a shared buffer whose size I have
// already complained about" must not, because that path would go on to reject
// the []float32 it declares as a type the device has no answer for -- a second
// diagnostic, about a different thing, for one mistake.
func (t *transpiler) sharedSize(e ast.Expr) (n int, isShared, ok bool) {
	call, callOK := unparen(e).(*ast.CallExpr)
	if !callOK {
		return 0, false, false
	}
	sel, selOK := unparen(call.Fun).(*ast.SelectorExpr)
	if !selOK {
		return 0, false, false
	}
	s := t.info.Selections[sel]
	if s == nil || s.Kind() != types.MethodVal || !IsCtx(s.Recv()) || s.Obj().Name() != "SharedF32" {
		return 0, false, false
	}
	if len(call.Args) != 1 {
		t.fail(call.Pos(), "SharedF32 takes one argument")
		return 0, true, false
	}
	n, constOK := t.constInt(call.Args[0])
	if !constOK {
		// A __shared__ array needs a compile-time extent, so this has to be
		// reported rather than silently mistranslated.
		t.fail(call.Pos(), "SharedF32 needs a constant size (got a runtime value)")
		return 0, true, false
	}
	return n, true, true
}

// simple renders an assignment or increment inline, without a terminator, so
// it can also serve as a for-loop clause.
func (t *transpiler) simple(s ast.Stmt) string {
	switch s := s.(type) {
	case nil:
		return ""
	case *ast.IncDecStmt:
		return fmt.Sprintf("%s%s", t.expr(s.X).s, s.Tok)
	case *ast.AssignStmt:
		if len(s.Lhs) != 1 || len(s.Rhs) != 1 {
			t.fail(s.Pos(), "multiple assignment is not supported in kernels")
			return ""
		}
		if s.Tok == token.DEFINE {
			id, ok := s.Lhs[0].(*ast.Ident)
			if !ok {
				t.fail(s.Pos(), "unsupported declaration target %T", s.Lhs[0])
				return ""
			}
			return t.define(id, s.Rhs[0])
		}
		switch s.Tok {
		case token.ASSIGN, token.ADD_ASSIGN, token.SUB_ASSIGN, token.MUL_ASSIGN,
			token.QUO_ASSIGN, token.REM_ASSIGN, token.AND_ASSIGN, token.OR_ASSIGN,
			token.XOR_ASSIGN, token.SHL_ASSIGN, token.SHR_ASSIGN:
			// Both sides stand in positions that accept a whole expression, so
			// neither needs parentheses of its own.
			return fmt.Sprintf("%s %s %s", t.expr(s.Lhs[0]).s, s.Tok, t.expr(s.Rhs[0]).s)
		default:
			t.fail(s.Pos(), "unsupported assignment %s", s.Tok)
			return ""
		}
	default:
		t.fail(s.Pos(), "unsupported clause %T", s)
		return ""
	}
}

func (t *transpiler) decl(s *ast.DeclStmt) {
	gd, ok := s.Decl.(*ast.GenDecl)
	if !ok {
		t.fail(s.Pos(), "unsupported declaration")
		return
	}
	switch gd.Tok {
	case token.CONST:
		// Constants are folded into their use sites by go/types, so nothing
		// needs to be emitted here.
		return
	case token.VAR:
		for _, spec := range gd.Specs {
			vs, ok := spec.(*ast.ValueSpec)
			if !ok {
				t.fail(spec.Pos(), "unsupported specification")
				return
			}
			if len(vs.Values) != 0 && len(vs.Values) != len(vs.Names) {
				t.fail(vs.Pos(), "unsupported var declaration")
				return
			}
			for i, n := range vs.Names {
				obj := t.info.Defs[n]
				if obj == nil {
					t.fail(n.Pos(), "%s has no resolved type", n.Name)
					return
				}
				ctype := t.ctype(obj.Type(), n.Pos())
				if len(vs.Values) == 0 {
					t.line("%s %s = 0;", ctype, cname(n.Name))
					continue
				}
				t.line("%s %s = %s;", ctype, cname(n.Name), t.expr(vs.Values[i]).s)
			}
		}
	default:
		t.fail(s.Pos(), "%s declarations are not supported in kernels", gd.Tok)
	}
}

func (t *transpiler) ifStmt(s *ast.IfStmt) {
	if s.Init != nil {
		// Give the init statement its own scope, as Go does.
		t.line("{")
		t.ind++
		t.line("%s;", t.simple(s.Init))
		defer func() {
			t.ind--
			t.line("}")
		}()
	}
	// C's statement grammar supplies the parentheses, so the condition is
	// rendered as the full expression it is.
	t.line("if (%s)", t.expr(s.Cond).s)
	t.block(s.Body)
	switch els := s.Else.(type) {
	case nil:
	case *ast.BlockStmt:
		t.line("else")
		t.block(els)
	case *ast.IfStmt:
		t.line("else")
		t.ifStmt(els)
	default:
		t.fail(s.Else.Pos(), "unsupported else branch %T", s.Else)
	}
}

func (t *transpiler) forStmt(s *ast.ForStmt) {
	switch {
	case s.Init == nil && s.Post == nil && s.Cond == nil:
		t.line("while (true)")
	case s.Init == nil && s.Post == nil:
		t.line("while (%s)", t.expr(s.Cond).s)
	default:
		cond := ""
		if s.Cond != nil {
			cond = t.expr(s.Cond).s
		}
		t.line("for (%s; %s; %s)", t.simple(s.Init), cond, t.simple(s.Post))
	}
	t.block(s.Body)
}

// rangeStmt lowers `for i := range x` where x is a slice or an integer.
func (t *transpiler) rangeStmt(s *ast.RangeStmt) {
	if s.Tok != token.DEFINE || s.Value != nil {
		t.fail(s.Pos(), "only `for i := range x` is supported in kernels")
		return
	}
	id, ok := s.Key.(*ast.Ident)
	if !ok {
		t.fail(s.Pos(), "unsupported range variable")
		return
	}
	var limit cexpr
	switch typ := t.info.Types[s.X].Type.Underlying().(type) {
	case *types.Slice:
		limit = t.lengthOf(s.X)
	case *types.Basic:
		if typ.Info()&types.IsInteger == 0 {
			t.fail(s.X.Pos(), "cannot range over %s", typ)
			return
		}
		limit = t.expr(s.X)
	default:
		t.fail(s.X.Pos(), "cannot range over %s", typ)
		return
	}
	// The limit becomes the right operand of a comparison, so anything binding
	// looser than one has to be parenthesised: `i < a & b` would otherwise
	// compare first and mask afterwards.
	name := cname(id.Name)
	t.line("for (int %s = 0; %s < %s; %s++)", name, name, limit.at(precRel+1), name)
	t.block(s.Body)
}
