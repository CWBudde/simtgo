package lower

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"strconv"
	"strings"
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
	case *ast.SwitchStmt:
		t.switchStmt(s)
	case *ast.TypeSwitchStmt:
		// Reported here rather than left to the catch-all, which would name
		// an AST node and say nothing about why: the device has no interfaces
		// to switch on in the first place.
		t.fail(s.Pos(), "type switches are not supported in kernels")
	case *ast.LabeledStmt:
		t.labeled(s)
	case *ast.BranchStmt:
		t.branch(s)
	case *ast.ReturnStmt:
		t.returnStmt(s)
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
		if req, isShared, ok := t.sharedCall(rhs); isShared {
			if ok {
				t.shared(id, req, s.Pos())
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
	if _, ok := obj.Type().Underlying().(*types.Array); ok {
		// `a := b` copies in Go and will not compile in C++, exactly as `a = b`
		// does not. A declaration with an initialiser is the same move.
		t.refuseArrayValue(rhs, id.Pos())
		t.poison(obj)
		return ""
	}
	before := len(t.diags)
	decl := t.cdecl(obj.Type(), cname(id.Name), id.Pos())
	if len(t.diags) > before {
		// The variable exists but has no device type. Every later use of it
		// would be a fresh complaint about the same declaration.
		t.poison(obj)
	}
	return fmt.Sprintf("%s = %s", decl, t.expr(rhs).s)
}

// returnStmt lowers a return. A kernel writes through its parameters and has
// nothing to return; a device function returns its one result.
func (t *transpiler) returnStmt(s *ast.ReturnStmt) {
	if len(s.Results) == 0 {
		t.line("return;")
		return
	}
	if !t.inDevice {
		t.fail(s.Pos(), "a kernel cannot return a value")
		return
	}
	if t.result == nil {
		t.fail(s.Pos(), "%s returns nothing", t.current.Name())
		return
	}
	if len(s.Results) > 1 {
		t.fail(s.Pos(), "a device function must return at most one value")
		return
	}
	t.line("return %s;", t.expr(s.Results[0]).s)
}

// sharedElems is package gpu's shared-tile vocabulary: the constructor's name,
// and the Go element type of the tile it hands back.
//
// Everything else about a tile is derived from that one entry -- the C element
// type through ctypeElem, the width through the same types.Sizes the rest of
// the emitter measures with -- so a new element type is a row here and nothing
// else. That is also what makes SharedF64 need //gocuda:float64 without a rule
// of its own: ctypeElem refuses float64 on a kernel that did not opt in,
// wherever the type came from, so membership of the double-precision
// vocabulary is what demands the permission rather than a second list that a
// later name could be left off.
var sharedElems = map[string]types.BasicKind{
	"SharedF32": types.Float32,
	"SharedF64": types.Float64,
	"SharedI32": types.Int32,
	"SharedI64": types.Int64,
	"SharedU32": types.Uint32,
	"SharedU64": types.Uint64,
}

// sharedDynElems is the same vocabulary for the tile CUDA sizes at the launch.
var sharedDynElems = map[string]types.BasicKind{
	"SharedDynF32": types.Float32,
	"SharedDynF64": types.Float64,
	"SharedDynI32": types.Int32,
	"SharedDynI64": types.Int64,
	"SharedDynU32": types.Uint32,
	"SharedDynU64": types.Uint64,
}

// sharedReq is one recognised shared-tile call: which constructor, what the
// tile holds, how long it is, and whether the launch is what says so.
type sharedReq struct {
	name    string
	elem    types.BasicKind
	n       int // the element count, and meaningless when dynamic
	dynamic bool
}

// shared declares a block's __shared__ tile.
//
// It is allowed in a device function as well as in a kernel. __shared__ inside
// a __device__ function is block-scoped storage that CUDA allocates once for
// the function, not once per call, and the accounting below follows suit
// because deviceFunc emits each function exactly once however many calls reach
// it. AssumeBlockDim stays refused there, because that one is a promise about
// a launch rather than a piece of storage.
func (t *transpiler) shared(id *ast.Ident, req sharedReq, pos token.Pos) {
	if t.ind != 1 {
		// The top level of the enclosing function, which is the kernel's body
		// or a device function's: a tile declared inside a conditional is one
		// the threads of a block could disagree about.
		t.fail(pos, "shared memory must be declared at the top level of the function")
		return
	}
	obj := t.info.Defs[id]
	if obj == nil {
		t.fail(id.Pos(), "%s has no resolved type", id.Name)
		return
	}
	elem := types.Typ[req.elem]
	// ctypeElem rather than ctype: a tile is memory with a stride, so it goes
	// through the same gate a slice element does -- which is also where
	// float64 without the directive is refused.
	ctype := t.ctypeElem(elem, pos)
	if t.failed() {
		t.poison(obj)
		return
	}
	name := cname(id.Name)
	width := int(t.sizes.Sizeof(elem))

	if req.dynamic {
		t.dynamicShared(obj, name, ctype, width, pos)
		return
	}
	t.line("__shared__ %s %s[%d];", ctype, name, req.n)
	t.lens[obj] = strconv.Itoa(req.n)
	// Every block gets its own copy of the tile, so the running total is what
	// each block will ask the device for.
	t.sharedBytes += width * req.n
}

// dynamicShared declares the tile CUDA sizes at the launch, and records what
// the host has to be told in order to size it.
//
// The length cannot be a constant the way a static tile's is, so it becomes an
// `int name_len` parameter of the kernel -- the convention a slice parameter
// already uses -- which Kernel.LaunchShared fills from the element count it
// was given. Everything downstream, len() included, then reads a tile exactly
// as it reads a slice.
func (t *transpiler) dynamicShared(obj types.Object, name, ctype string, width int, pos token.Pos) {
	if t.inDevice {
		// A static tile is fine in a device function; this one is not, and the
		// difference is worth stating rather than leaving as a C identifier
		// that was never declared. The length is a parameter of the kernel,
		// and a device function cannot see it.
		t.fail(pos, "a dynamically sized shared tile may only be declared in a kernel: its length is a launch parameter, which a device function cannot see")
		return
	}
	if t.dynShared != "" {
		// NVRTC accepts a second `extern __shared__` declaration without a
		// word -- of a different element type as readily as of the same one --
		// and every one of them names the same block of memory. Two names that
		// silently alias is exactly the mistranslation this emitter exists to
		// refuse, so it is refused here, where there is still a position to
		// report it against.
		t.fail(pos, "a kernel may declare at most one dynamically sized shared tile, and %s is already one: CUDA has a single dynamic __shared__ block per launch, so a second name would be another view of the same bytes", t.dynShared)
		return
	}
	lenName := name + "_len"
	if who, taken := t.sigNames[lenName]; taken {
		t.fail(pos, "the length generated for dynamic shared tile %s collides with %s; rename one of them", name, who)
		return
	}
	t.line("extern __shared__ %s %s[];", ctype, name)
	t.lens[obj] = lenName
	t.dynShared, t.dynSharedLen, t.dynSharedWidth = name, lenName, width
}

// sharedCall reports whether e is one of package gpu's shared-tile
// constructors, and if so whether the call was well formed.
//
// The two answers have to be separate. "Not a shared buffer" sends the caller
// down the ordinary declaration path; "a shared buffer I have already
// complained about" must not, because that path would go on to reject the
// []float32 it declares as a type the device has no answer for -- a second
// diagnostic, about a different thing, for one mistake.
func (t *transpiler) sharedCall(e ast.Expr) (req sharedReq, isShared, ok bool) {
	call, callOK := unparen(e).(*ast.CallExpr)
	if !callOK {
		return req, false, false
	}
	sel, selOK := unparen(call.Fun).(*ast.SelectorExpr)
	if !selOK {
		return req, false, false
	}
	s := t.info.Selections[sel]
	if s == nil || s.Kind() != types.MethodVal || !IsCtx(s.Recv()) {
		return req, false, false
	}
	name := s.Obj().Name()
	if elem, isDyn := sharedDynElems[name]; isDyn {
		if len(call.Args) != 0 {
			t.fail(call.Pos(), "%s takes no arguments; the launch is what gives its length", name)
			return req, true, false
		}
		return sharedReq{name: name, elem: elem, dynamic: true}, true, true
	}
	elem, isStatic := sharedElems[name]
	if !isStatic {
		return req, false, false
	}
	if len(call.Args) != 1 {
		t.fail(call.Pos(), "%s takes one argument", name)
		return req, true, false
	}
	n, constOK := t.constInt(call.Args[0])
	if !constOK {
		// A __shared__ array needs a compile-time extent, so this has to be
		// reported rather than silently mistranslated. A length only the host
		// knows is what the SharedDyn constructors are for, and saying so is
		// more use than naming the rule alone.
		t.fail(call.Pos(), "%s needs a constant size (got a runtime value); use %s, whose length the launch gives", name, dynSpelling(name))
		return req, true, false
	}
	return sharedReq{name: name, elem: elem, n: n}, true, true
}

// dynSpelling names the launch-sized constructor of the same element type, so
// that refusing a runtime size can also say what to write instead.
func dynSpelling(static string) string {
	return "SharedDyn" + strings.TrimPrefix(static, "Shared")
}

// refuseArrayValue reports, and refuses, an expression whose type is an array
// being moved as a whole.
//
// Go copies an array on assignment. C++ will not assign a raw array at all, so
// the same spelling becomes an NVRTC error about generated code. An array here
// is storage: declare it, index it, len it, range it, put it in a struct. Copy
// it element by element, or wrap it in a struct, which both languages copy.
func (t *transpiler) refuseArrayValue(e ast.Expr, pos token.Pos) bool {
	typ := t.typeOf(e)
	if typ == nil {
		return false
	}
	if _, ok := typ.Underlying().(*types.Array); ok {
		t.fail(pos, "a whole %s cannot be assigned: Go copies it and C++ will not assign an array at all; copy the elements, or wrap the array in a struct", typ)
		return true
	}
	return false
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
			if t.refuseArrayValue(s.Lhs[0], s.Pos()) {
				return ""
			}
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
					continue
				}
				if _, isArr := obj.Type().Underlying().(*types.Array); isArr && len(vs.Values) != 0 {
					// `var a [N]T = b` moves the array as a whole, which is the
					// same thing `a := b` and `a = b` do and which C++ refuses
					// for all three. Without this the initialiser reached NVRTC
					// as `float a[2] = b;`. An uninitialised `var` is fine: it
					// gets zeroValue's `{}`, which is how C++ spells it.
					t.refuseArrayValue(vs.Values[i], n.Pos())
					t.poison(obj)
					continue
				}
				before := len(t.diags)
				decl := t.cdecl(obj.Type(), cname(n.Name), n.Pos())
				if len(t.diags) > before {
					// Every later use of a variable with no device type would
					// repeat this one refusal.
					t.poison(obj)
					continue
				}
				if len(vs.Values) == 0 {
					t.line("%s = %s;", decl, t.zeroValue(obj.Type()))
					continue
				}
				t.line("%s = %s;", decl, t.expr(vs.Values[i]).s)
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

// breakable names what a bare `break` would leave, so that the emitter can
// tell the two switch lowerings apart. C's break binds to the nearest loop or
// switch, and a Go switch that had to become an if/else chain is neither.
type breakable int

const (
	breakLoop breakable = iota
	breakCSwitch
	breakIfSwitch
)

// labelState is one labelled loop, live while its body is lowered.
//
// Whether each target is reached is settled before the body is rendered,
// because the emitter writes forwards into a buffer and an unused C label is a
// warning -- and, being unreachable, would be a lie about the control flow.
type labelState struct {
	name         string // the label's C spelling
	usesBreak    bool
	usesContinue bool
}

// branch lowers break, continue and fallthrough.
//
// The labelled forms are why this is not two lines: C has no labelled break,
// so both become a goto. Before that was implemented the label was dropped
// and `break outer` left the innermost loop instead -- the generated kernel
// compiled and computed something else, which is the one thing the transpiler
// must never do.
func (t *transpiler) branch(s *ast.BranchStmt) {
	switch s.Tok {
	case token.BREAK:
		if s.Label != nil {
			t.jump(s, "break")
			return
		}
		if t.enclosing() == breakIfSwitch {
			t.fail(s.Pos(), "cannot break out of a switch whose cases are not constants: "+
				"it lowers to an if/else chain, where a C break would leave the enclosing loop instead")
			return
		}
		t.line("break;")
	case token.CONTINUE:
		if s.Label != nil {
			t.jump(s, "continue")
			return
		}
		t.line("continue;")
	case token.FALLTHROUGH:
		// A fallthrough that ends a case clause is consumed by cSwitch; one
		// reaching here is in a switch that had to become an if/else chain.
		t.fail(s.Pos(), "fallthrough is not supported in a switch whose cases are not constants")
	default:
		t.fail(s.Pos(), "%s is not supported in kernels", s.Tok)
	}
}

// jump emits the goto standing in for a labelled break or continue.
func (t *transpiler) jump(s *ast.BranchStmt, kind string) {
	lbl := t.labels[s.Label.Name]
	if lbl == nil {
		// go/types has already refused an undefined label; this only catches a
		// label on a statement the emitter declined to treat as a loop.
		t.fail(s.Pos(), "%s %s: %s does not label a for loop", s.Tok, s.Label.Name, s.Label.Name)
		return
	}
	t.line("goto %s_%s;", lbl.name, kind)
}

// enclosing reports what a bare break would leave.
func (t *transpiler) enclosing() breakable {
	if len(t.breakables) == 0 {
		return breakLoop
	}
	return t.breakables[len(t.breakables)-1]
}

// labeled lowers a labelled statement. Only a loop may carry a label: Go also
// allows one on a switch or a select, and neither has a meaning here that a
// goto could stand in for.
func (t *transpiler) labeled(s *ast.LabeledStmt) {
	switch s.Stmt.(type) {
	case *ast.ForStmt, *ast.RangeStmt:
	default:
		t.fail(s.Pos(), "a label may only be placed on a for loop in kernels")
		return
	}
	lbl := &labelState{name: cname(s.Label.Name)}
	ast.Inspect(s.Stmt, func(n ast.Node) bool {
		b, ok := n.(*ast.BranchStmt)
		if !ok || b.Label == nil || b.Label.Name != s.Label.Name {
			return true
		}
		switch b.Tok {
		case token.BREAK:
			lbl.usesBreak = true
		case token.CONTINUE:
			lbl.usesContinue = true
		}
		return true
	})

	if t.labels == nil {
		t.labels = map[string]*labelState{}
	}
	t.labels[s.Label.Name] = lbl
	t.pendingLabel = lbl
	t.stmt(s.Stmt)
	t.pendingLabel = nil
	delete(t.labels, s.Label.Name)

	if lbl.usesBreak {
		// C labels a statement, not a position, so the empty statement is the
		// target rather than decoration.
		t.line("%s_break: ;", lbl.name)
	}
}

// takeLabel claims the label waiting for the loop about to be rendered, so a
// nested loop can never inherit it.
func (t *transpiler) takeLabel() *labelState {
	lbl := t.pendingLabel
	t.pendingLabel = nil
	return lbl
}

// loopBody renders a loop body: head opens it, and a labelled loop closes it
// with its continue target.
//
// The target belongs at the very end of the body rather than after the loop,
// because falling off the end of a C for body still runs the post clause --
// which is exactly what Go's labelled continue does.
//
// The statements then need a scope of their own, so that the goto leaves that
// scope instead of jumping forward within it. C++ refuses a jump that enters
// the scope of an initialised variable without running its initialiser, and
// `if c { continue outer }; v := float32(1)` is exactly that shape: valid Go
// that NVRTC would reject with an error about generated code.
func (t *transpiler) loopBody(b *ast.BlockStmt, head string, lbl *labelState) {
	t.breakables = append(t.breakables, breakLoop)
	defer func() { t.breakables = t.breakables[:len(t.breakables)-1] }()

	scoped := lbl != nil && lbl.usesContinue
	t.line("{")
	t.ind++
	if scoped {
		t.line("{")
		t.ind++
	}
	if head != "" {
		t.line("%s", head)
	}
	for _, s := range b.List {
		t.mark = len(t.diags)
		t.stmt(s)
	}
	if scoped {
		t.ind--
		t.line("}")
		t.line("%s_continue: ;", lbl.name)
	}
	t.ind--
	t.line("}")
}

func (t *transpiler) forStmt(s *ast.ForStmt) {
	lbl := t.takeLabel()
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
	t.loopBody(s.Body, "", lbl)
}

// rangeStmt lowers `for i := range x` and `for i, v := range x`, where x is a
// slice or -- for the index-only form -- an integer.
func (t *transpiler) rangeStmt(s *ast.RangeStmt) {
	lbl := t.takeLabel()
	if s.Tok != token.DEFINE {
		t.fail(s.Pos(), "only `for i := range x` and `for i, v := range x` are supported in kernels")
		return
	}
	key, ok := s.Key.(*ast.Ident)
	if !ok {
		t.fail(s.Pos(), "unsupported range variable")
		return
	}
	value, ok := rangeValue(s.Value)
	if !ok {
		t.fail(s.Pos(), "unsupported range variable")
		return
	}

	var limit cexpr
	indexable := false
	// The counter's C type. len() is an int, so a slice or an array counts in
	// int; ranging over an integer gives the index that integer's own type,
	// and emitting int for a wider one would be a mistranslation rather than a
	// narrowing -- `i*i` at i = 50000 is 2500000000 in Go and -1794967296 in
	// 32-bit C, and a bound above MaxInt32 would never terminate at all.
	counter := "int"
	xt := t.typeOf(s.X)
	if xt == nil {
		return
	}
	switch typ := xt.Underlying().(type) {
	case *types.Slice:
		limit, indexable = t.lengthOf(s.X), true
	case *types.Array:
		// An array's length is part of its type, so the bound is a literal
		// rather than a companion parameter.
		limit, indexable = number(strconv.FormatInt(typ.Len(), 10)), true
	case *types.Basic:
		if typ.Info()&types.IsInteger == 0 {
			t.fail(s.X.Pos(), "cannot range over %s", typ)
			return
		}
		counter = t.ctype(xt, s.X.Pos())
		limit = t.expr(s.X)
	default:
		t.fail(s.X.Pos(), "cannot range over %s", typ)
		return
	}
	if value != nil && !indexable {
		t.fail(s.Pos(), "only a slice or an array can be ranged over with a value")
		return
	}

	// An index is needed even when the source discards it, because the value
	// is read out of the slice by hand. It is derived from the value's name
	// and stepped past anything the kernel already spells that way, so it can
	// neither collide with nor shadow a variable the author wrote.
	name := cname(key.Name)
	if key.Name == "_" {
		// Named after the value, which is what makes the generated index
		// readable. Go rejects a range clause that declares neither variable,
		// so there is always a value here -- but the name is chosen without
		// relying on that, because a nil dereference is a poor way to find
		// out otherwise.
		base := "range"
		if value != nil {
			base = cname(value.Name)
		}
		name = t.reserve(base + "_i")
		defer t.release(name)
	}

	// The value is a copy in Go, so it is a local here too: writing to it must
	// not reach the slice.
	head := ""
	if value != nil {
		obj := t.info.Defs[value]
		if obj == nil {
			t.fail(value.Pos(), "%s has no resolved type", value.Name)
			return
		}
		elem := t.cdecl(obj.Type(), cname(value.Name), value.Pos())
		head = fmt.Sprintf("%s = %s[%s];", elem, t.expr(s.X).at(precPostfix), name)
	}

	// The limit becomes the right operand of a comparison, so anything binding
	// looser than one has to be parenthesised: `i < a & b` would otherwise
	// compare first and mask afterwards.
	t.line("for (%s %s = 0; %s < %s; %s++)", counter, name, name, limit.at(precRel+1), name)
	t.loopBody(s.Body, head, lbl)
}

// rangeValue resolves the second range variable, reporting the blank one and
// an absent one alike as "no value to bind".
func rangeValue(e ast.Expr) (*ast.Ident, bool) {
	if e == nil {
		return nil, true
	}
	id, ok := e.(*ast.Ident)
	if !ok {
		return nil, false
	}
	if id.Name == "_" {
		return nil, true
	}
	return id, true
}

// reserve returns a generated name that nothing in the kernel already spells,
// and records it so a nested construct cannot pick the same one.
func (t *transpiler) reserve(base string) string {
	name := base
	for i := 2; t.names[name]; i++ {
		name = base + strconv.Itoa(i)
	}
	if t.names == nil {
		t.names = map[string]bool{}
	}
	t.names[name] = true
	return name
}

// release gives a reserved name back once the construct that held it is done,
// so two sibling loops read the same rather than counting upwards.
func (t *transpiler) release(name string) { delete(t.names, name) }

// switchStmt lowers a Go switch.
//
// There are two lowerings, and choosing the wrong one is a mistranslation
// rather than a compile error. C's switch takes constant case labels and falls
// through by default; Go's takes arbitrary expressions and does not. So a
// switch whose cases are all constant becomes a C switch with an explicit
// break closing each clause, and every other switch becomes the if/else chain
// that Go's semantics actually describe.
func (t *transpiler) switchStmt(s *ast.SwitchStmt) {
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
	if t.constantCases(s) {
		t.cSwitch(s)
		return
	}
	t.chainSwitch(s)
}

// constantCases reports whether s can be rendered as a C switch: an integral
// tag, and every case label a constant the emitter can fold.
func (t *transpiler) constantCases(s *ast.SwitchStmt) bool {
	if s.Tag == nil {
		return false
	}
	tt := t.typeOf(s.Tag)
	if tt == nil {
		return false
	}
	basic, ok := tt.Underlying().(*types.Basic)
	if !ok || basic.Info()&types.IsInteger == 0 {
		return false
	}
	for _, cc := range s.Body.List {
		clause, ok := cc.(*ast.CaseClause)
		if !ok {
			return false
		}
		for _, e := range clause.List {
			if _, ok := t.constInt(e); !ok {
				return false
			}
		}
	}
	return true
}

// cSwitch renders the constant form as a C switch.
func (t *transpiler) cSwitch(s *ast.SwitchStmt) {
	t.line("switch (%s)", t.expr(s.Tag).s)
	t.line("{")
	t.breakables = append(t.breakables, breakCSwitch)
	defer func() { t.breakables = t.breakables[:len(t.breakables)-1] }()

	for _, cc := range s.Body.List {
		clause := cc.(*ast.CaseClause)
		if len(clause.List) == 0 {
			t.line("default:")
		}
		for _, e := range clause.List {
			n, _ := t.constInt(e)
			t.line("case %d:", n)
		}
		// Each clause is braced, as a Go case clause already is: two clauses
		// may declare the same name, and C++ refuses both that redeclaration
		// and any jump to a later case across an initialisation in an
		// earlier one.
		t.ind++
		t.line("{")
		t.ind++
		body, falls := trimFallthrough(clause.Body)
		for _, st := range body {
			t.mark = len(t.diags)
			t.stmt(st)
		}
		// Go does not fall through; C does. The break is what makes the two
		// mean the same thing, and omitting it is how fallthrough is honoured.
		if !falls {
			t.line("break;")
		}
		t.ind--
		t.line("}")
		t.ind--
	}
	t.line("}")
}

// chainSwitch renders every other form as the if/else chain Go describes.
//
// The tag is bound to a local first, and the comparisons test that local. Go
// evaluates a switch tag exactly once, and a tag can now call a device
// function that writes through a slice parameter, so comparing the expression
// itself in every arm would run those writes once per arm and could pick a
// different branch than Go does.
func (t *transpiler) chainSwitch(s *ast.SwitchStmt) {
	t.breakables = append(t.breakables, breakIfSwitch)
	defer func() { t.breakables = t.breakables[:len(t.breakables)-1] }()

	tag := ""
	if s.Tag != nil {
		typ := t.typeOf(s.Tag)
		if typ == nil {
			return
		}
		// Each arm becomes `switch_tag == <case>`, which never passes through
		// binary() and so would slip past the refusal there. Go compares a
		// struct or an array field by field and C++ gives a plain aggregate no
		// operator== at all, so this reached NVRTC as an error about generated
		// code instead of being refused with a position.
		switch typ.Underlying().(type) {
		case *types.Struct, *types.Array:
			t.fail(s.Tag.Pos(), "cannot switch on %s: each case would compare it field by field in Go, which C cannot do; switch on the field you mean", typ)
			return
		}
		tag = t.reserve("switch_tag")
		defer t.release(tag)
		t.line("{")
		t.ind++
		defer func() {
			t.ind--
			t.line("}")
		}()
		t.line("%s = %s;", t.cdecl(typ, tag, s.Tag.Pos()), t.expr(s.Tag).s)
	}

	var dflt *ast.CaseClause
	emitted := 0
	for _, cc := range s.Body.List {
		clause, ok := cc.(*ast.CaseClause)
		if !ok {
			t.fail(cc.Pos(), "unsupported switch clause %T", cc)
			return
		}
		if len(clause.List) == 0 {
			// Go evaluates the default last however it is written, so it is
			// held back rather than emitted in place.
			dflt = clause
			continue
		}
		if emitted > 0 {
			t.line("else")
		}
		t.line("if (%s)", t.caseCond(tag, clause.List))
		t.clauseBlock(clause.Body)
		emitted++
	}
	if dflt == nil {
		return
	}
	if emitted > 0 {
		t.line("else")
	}
	t.clauseBlock(dflt.Body)
}

// caseCond renders one clause's condition: the case values or-ed together,
// each compared against the local holding the tag. An empty tag is a tagless
// switch, whose cases are conditions in their own right.
func (t *transpiler) caseCond(tag string, list []ast.Expr) string {
	parts := make([]string, 0, len(list))
	for _, e := range list {
		if tag == "" {
			parts = append(parts, t.expr(e).at(precOr+1))
			continue
		}
		eq := cexpr{fmt.Sprintf("%s == %s", tag, t.expr(e).at(precEq+1)), precEq}
		parts = append(parts, eq.at(precOr+1))
	}
	return strings.Join(parts, " || ")
}

// clauseBlock renders a case clause's statements as a braced block, which a
// Go case clause already is in every way that matters: it has its own scope.
func (t *transpiler) clauseBlock(body []ast.Stmt) {
	t.line("{")
	t.ind++
	for _, s := range body {
		t.mark = len(t.diags)
		t.stmt(s)
	}
	t.ind--
	t.line("}")
}

// trimFallthrough splits a case clause's trailing fallthrough off its body.
func trimFallthrough(body []ast.Stmt) ([]ast.Stmt, bool) {
	if n := len(body); n > 0 {
		if b, ok := body[n-1].(*ast.BranchStmt); ok && b.Tok == token.FALLTHROUGH {
			return body[:n-1], true
		}
	}
	return body, false
}
