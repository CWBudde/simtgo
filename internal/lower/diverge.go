package lower

import (
	"go/ast"
	"go/token"
	"go/types"
)

// This file answers one question: does every thread of a block reach the
// barriers this kernel executes? __syncthreads() that only some threads arrive
// at is undefined behaviour -- not a wrong number but a hang, a silently
// skipped rendezvous, or whatever the hardware happens to do -- so it is
// refused rather than emitted.
//
// Two facts about the subset make a structured walk over the Go AST enough,
// and both are properties of the lowering rather than of Go:
//
//   - goto is refused (stmt.go, branch) and a label may only be placed on a
//     for loop, so a function's control flow is exactly its statement tree.
//     The generated C does contain gotos -- they stand in for labelled break
//     and continue -- but they are the emitter's own, and an analysis over the
//     generated C would have to prove that. Over the Go source there is
//     nothing to prove.
//   - a barrier can only be an *ast.ExprStmt. gpu.Ctx.SyncThreads returns
//     nothing, so go/types keeps it out of every value position, and simple()
//     accepts only assignments and increments, so it cannot hide in a for
//     clause or an if initialiser either. Those three are each refused with
//     "unsupported clause", which is checked in simt/errors_test.go.
//
// The warp-level primitives are held to the same rule, which is stricter than
// they need. Their real requirement is warp uniformity, and the block lattice
// below over-approximates it: a condition that varies across a block may still
// be constant within each warp. It is one lattice and one message because the
// emitter writes 0xffffffff into every _sync built-in (warp.go), which is an
// unchecked promise that the whole warp participates, and because no kernel
// here writes the shape that costs -- `if ctx.All(p) { ctx.SyncWarp() }`. When
// one does, the lattice can gain a warp level without the rules changing.

// barrierUse is what one declaration does that the whole block has to do
// together. It is deliberately this small so that it is call-site independent:
// a helper reached by two paths is analysed once, and what the summary says is
// true wherever it is called from.
//
// barrier and warp cross a call, because a barrier inside a device function is
// a barrier the caller executes at the point of the call. returns does not,
// and that asymmetry is worth stating: a return from a device function hands
// control back to its caller and the thread goes on to whatever follows, so an
// early return there cannot make a thread miss the caller's barrier. It is the
// function's own answer, read by rule 3 while the function that produced it is
// being walked and by nothing above it. The kernel is the one function where a
// return does end the thread, and its barriers are in this same walk.
type barrierUse struct {
	barrier bool // reaches ctx.SyncThreads()
	warp    bool // reaches a _sync built-in, which carries the full-warp mask
	returns bool // may return under a thread-varying condition
}

// ctxUniform is the part of gpu.Ctx's vocabulary that answers the same in
// every thread of a block.
//
// It is the short list on purpose: everything not named here is taken to vary,
// which is how gpu.Ctx.ThreadIdx, the shuffles, the votes and anything added
// later default to the safe side. The three families here are the block's own
// coordinates and the grid's shape, which the launch fixes and no thread can
// change.
var ctxUniform = map[string]bool{
	"BlockIdx": true, "BlockIdxY": true, "BlockIdxZ": true,
	"BlockDim": true, "BlockDimY": true, "BlockDimZ": true,
	"GridDim": true, "GridDimY": true, "GridDimZ": true,
}

// The three constructs a barrier can end up inside, spelled as the refusal
// names them. They are constants rather than literals because report tells the
// loop apart from the other two by this value, and a typo would otherwise
// choose a message rather than fail to compile.
const (
	divIf     = "if"
	divSwitch = "switch"
	divLoop   = "loop"
)

// site is the innermost construct enclosing the walk that not every thread of
// the block enters. A zero site means the walk is on the block's common path.
type site struct {
	pos  token.Pos
	what string
}

func (s site) diverged() bool { return s.pos.IsValid() }

// divergeWalk carries the state of one function's analysis. It is a struct
// rather than a pile of arguments because the varying set and the running
// record of what has been found are read at every level of the walk.
type divergeWalk struct {
	t    *transpiler
	vary map[types.Object]bool
	use  *barrierUse
	// ret is where the first return this function may take under a
	// thread-varying condition is written. Whether there was one is use.returns;
	// this is only the position the message quotes, which is worth naming
	// because a barrier and the return that lets threads past it are rarely
	// next to each other. It goes from unset to set and never back: the walk is
	// in source order, so once threads have been let out of the function,
	// everything after is reached by fewer of them than started.
	ret token.Pos
}

// checkDivergence refuses the barriers in the kernel fd, and in every function
// it reaches, that not all of a block's threads arrive at.
func (t *transpiler) checkDivergence(fd *ast.FuncDecl) {
	t.barrierUse(fd, true)
}

// barrierUse analyses fd and reports what it does that the whole block has to
// do together.
//
// The result is memoised per declaration, and analysing is the cycle guard,
// set before descending: recursion is refused when the kernel is lowered, but
// this walk runs over declarations and has to terminate on its own. entry says
// whether fd is the kernel, which is the one function whose scalar parameters
// are known to hold the same value in every thread.
func (t *transpiler) barrierUse(fd *ast.FuncDecl, entry bool) barrierUse {
	if u, ok := t.uses[fd]; ok {
		return u
	}
	if fd == nil || fd.Body == nil || t.diverging[fd] {
		// A cycle, or a declaration with nothing to walk. The empty summary is
		// the one place this file is not pessimistic, and deliberately: a
		// cycle means the lowering is about to refuse the kernel for the
		// recursion, and a body that does not exist holds no barrier. Assuming
		// a barrier here would answer a kernel that cannot be built at all
		// with a second diagnostic about a call that is not the problem, which
		// is the noise t.mark and poison exist to keep out.
		return barrierUse{}
	}
	if t.diverging == nil {
		t.diverging = map[*ast.FuncDecl]bool{}
	}
	t.diverging[fd] = true

	var use barrierUse
	w := &divergeWalk{t: t, vary: t.varyingObjects(fd, entry), use: &use}
	w.check(fd.Body, site{})

	delete(t.diverging, fd)
	if t.uses == nil {
		t.uses = map[*ast.FuncDecl]barrierUse{}
	}
	t.uses[fd] = use
	return use
}

// check walks one statement, carrying the innermost divergent construct
// enclosing it.
func (w *divergeWalk) check(s ast.Stmt, at site) {
	switch s := s.(type) {
	case nil:
		return
	case *ast.ExprStmt:
		if w.isBarrier(s.X) {
			w.use.barrier = true
			w.report(s.Pos(), "ctx.SyncThreads()", at)
			return
		}
		w.checkExpr(s.X, at)
	case *ast.AssignStmt:
		for _, e := range s.Lhs {
			w.checkExpr(e, at)
		}
		for _, e := range s.Rhs {
			w.checkExpr(e, at)
		}
	case *ast.IncDecStmt:
		w.checkExpr(s.X, at)
	case *ast.DeclStmt:
		w.checkDecl(s, at)
	case *ast.BlockStmt:
		for _, st := range s.List {
			w.check(st, at)
		}
	case *ast.LabeledStmt:
		w.check(s.Stmt, at)
	case *ast.IfStmt:
		w.check(s.Init, at)
		// The condition is evaluated by everyone who reaches the if, so it
		// belongs to the enclosing context and not to the branch it guards.
		w.checkExpr(s.Cond, at)
		inner := at
		if !inner.diverged() && w.varyingExpr(s.Cond) {
			inner = site{s.Pos(), divIf}
		}
		w.check(s.Body, inner)
		w.check(s.Else, inner)
	case *ast.ForStmt:
		w.check(s.Init, at)
		inner := at
		if !inner.diverged() && w.forVaries(s) {
			inner = site{s.Pos(), divLoop}
		}
		// The condition and the post statement are re-run by whichever threads
		// are still going round, so both are inside the loop's divergence even
		// though neither is in its body.
		w.checkExpr(s.Cond, inner)
		w.check(s.Post, inner)
		w.check(s.Body, inner)
	case *ast.RangeStmt:
		w.checkExpr(s.X, at)
		inner := at
		if !inner.diverged() && w.rangeVaries(s) {
			inner = site{s.Pos(), divLoop}
		}
		w.check(s.Body, inner)
	case *ast.SwitchStmt:
		w.check(s.Init, at)
		w.checkExpr(s.Tag, at)
		inner := at
		if !inner.diverged() && w.switchVaries(s) {
			inner = site{s.Pos(), divSwitch}
		}
		for _, cl := range s.Body.List {
			cc, ok := cl.(*ast.CaseClause)
			if !ok {
				continue
			}
			// A case expression is compared only by the threads that got past
			// the cases before it, so it sits inside the switch's divergence.
			for _, e := range cc.List {
				w.checkExpr(e, inner)
			}
			for _, st := range cc.Body {
				w.check(st, inner)
			}
		}
	case *ast.ReturnStmt:
		for _, e := range s.Results {
			w.checkExpr(e, at)
		}
		if at.diverged() {
			w.use.returns = true
			if !w.ret.IsValid() {
				w.ret = s.Pos()
			}
		}
	case *ast.BranchStmt, *ast.EmptyStmt:
	default:
		// Something the lowering does not accept either. Nothing to walk, and
		// the statement's own refusal is the one worth reading.
	}
}

// checkDecl walks the initialisers of a var declaration.
func (w *divergeWalk) checkDecl(s *ast.DeclStmt, at site) {
	gen, ok := s.Decl.(*ast.GenDecl)
	if !ok {
		return
	}
	for _, spec := range gen.Specs {
		vs, ok := spec.(*ast.ValueSpec)
		if !ok {
			continue
		}
		for _, e := range vs.Values {
			w.checkExpr(e, at)
		}
	}
}

// checkExpr finds every call inside e that the whole block has to reach
// together.
//
// A barrier cannot appear here -- it is a statement, and only a statement --
// but a warp primitive is an ordinary value-returning call, and so is a call
// to a device function that reaches either.
func (w *divergeWalk) checkExpr(e ast.Expr, at site) {
	if e == nil {
		return
	}
	ast.Inspect(e, func(n ast.Node) bool {
		if c, ok := n.(*ast.CallExpr); ok {
			w.checkCall(c, at)
		}
		return true
	})
}

// checkCall reports a warp primitive, or a call that reaches one or a barrier,
// standing where not every thread of the block arrives.
func (w *divergeWalk) checkCall(c *ast.CallExpr, at site) {
	fun := unparen(c.Fun)
	if tv, ok := w.t.info.Types[fun]; ok && tv.IsType() {
		return // a conversion
	}
	switch f := fun.(type) {
	case *ast.Ident:
		obj, ok := w.t.info.Uses[f].(*types.Func)
		if !ok {
			return // len, min, max, or something the lowering refuses
		}
		decl := w.t.declOf(obj)
		if decl == nil {
			return // not declared in this package; refused where it is lowered
		}
		use := w.t.barrierUse(decl, false)
		if use.barrier {
			w.use.barrier = true
			w.report(c.Pos(), obj.Name()+", which reaches ctx.SyncThreads(),", at)
			return
		}
		if use.warp {
			w.use.warp = true
			w.report(c.Pos(), obj.Name()+", which reaches a warp-level primitive,", at)
		}
	case *ast.SelectorExpr:
		sel := w.t.info.Selections[f]
		if sel == nil || sel.Kind() != types.MethodVal || !IsCtx(sel.Recv()) {
			return
		}
		name := sel.Obj().Name()
		// op.mask is the precise test rather than membership of ctxWarp:
		// these are exactly the calls the emitter writes 0xffffffff into, and
		// that literal is the promise that every lane of the warp is here.
		// gpu.Ctx.LaneID is arithmetic on threadIdx and not a call at all, and
		// __activemask takes no mask -- it reports which lanes are converged
		// rather than requiring that they all are, so under divergence it has
		// a defined answer and merely an unhelpful one.
		if op, ok := ctxWarp[name]; ok && op.mask {
			w.use.warp = true
			w.report(c.Pos(), "ctx."+name, at)
		}
	}
}

// isBarrier reports whether e is the ctx.SyncThreads() call.
func (w *divergeWalk) isBarrier(e ast.Expr) bool {
	c, ok := unparen(e).(*ast.CallExpr)
	if !ok {
		return false
	}
	f, ok := unparen(c.Fun).(*ast.SelectorExpr)
	if !ok {
		return false
	}
	sel := w.t.info.Selections[f]
	return sel != nil && sel.Kind() == types.MethodVal && IsCtx(sel.Recv()) &&
		sel.Obj().Name() == "SyncThreads"
}

// uniformAdvice is the half of every refusal that says what to do instead. The
// vocabulary it names is ctxUniform plus the things a launch fixes, which
// together are the whole of what a kernel can branch on and still have every
// thread agree.
const uniformAdvice = "block-uniform -- ctx.BlockIdx, ctx.BlockDim, ctx.GridDim, " +
	"a scalar parameter, len() and the constants are the same in every thread of the block"

// report refuses subject standing where not every thread of the block arrives.
//
// The three rules share one function because they share one reason: the
// generated barrier is executed by fewer threads than the block has, and CUDA
// says nothing about what happens then. They are separate messages because the
// fix is different in each case.
func (w *divergeWalk) report(pos token.Pos, subject string, at site) {
	switch {
	case at.what == divLoop:
		w.t.fail(pos, "%s is inside a loop whose trip count differs between threads of the block, "+
			"so they do not all execute it the same number of times, which is undefined on the device. "+
			"Move it out of the loop, or make the loop's bound %s", subject, uniformAdvice)
	case at.diverged():
		w.t.fail(pos, "%s is under a thread-varying %s, so only some of the block's threads "+
			"arrive at it, which is undefined on the device. "+
			"Hoist it out of the %s, or make the condition %s", subject, at.what, at.what, uniformAdvice)
	case w.use.returns:
		w.t.fail(pos, "%s is preceded by a return at %s taken under a condition that differs between "+
			"threads of the block, so the threads that returned never arrive at it, which is undefined "+
			"on the device. Guard the work instead of returning from it, or make the condition %s",
			subject, w.t.fset.Position(w.ret), uniformAdvice)
	}
}

// forVaries reports whether the threads of a block may go round s a different
// number of times.
//
// Testing the condition is enough to cover the initialiser and the post
// statement as well: anything either of them does to the loop variable reaches
// the trip count only through the condition that reads it, and varyingObjects
// has already run to a fixpoint, so the variable is known to vary by the time
// the condition is asked about it. What the condition cannot see is a thread
// leaving early, which is the other half.
func (w *divergeWalk) forVaries(s *ast.ForStmt) bool {
	if w.varyingExpr(s.Cond) {
		return true
	}
	return w.exitsVarying(s.Body)
}

// rangeVaries is the same question for a range loop.
//
// Ranging over a slice or an array counts to a length the whole block agrees
// on, so only `for i := range n` over an integer can vary by itself.
func (w *divergeWalk) rangeVaries(s *ast.RangeStmt) bool {
	if xt := w.t.typeOf(s.X); xt != nil {
		if _, basic := xt.Underlying().(*types.Basic); basic && w.varyingExpr(s.X) {
			return true
		}
	}
	return w.exitsVarying(s.Body)
}

// switchVaries reports whether the threads of a block may land in different
// clauses of s. A tagless switch is a chain of conditions, so its cases are
// what has to agree.
func (w *divergeWalk) switchVaries(s *ast.SwitchStmt) bool {
	if w.varyingExpr(s.Tag) {
		return true
	}
	for _, cl := range s.Body.List {
		cc, ok := cl.(*ast.CaseClause)
		if !ok {
			continue
		}
		for _, e := range cc.List {
			if w.varyingExpr(e) {
				return true
			}
		}
	}
	return false
}

// exitsVarying reports whether a loop with this body can be left early by some
// threads and not others.
//
// A bare break or continue belongs to the innermost loop, so one inside a
// nested loop says nothing about this one; a labelled one is counted wherever
// it is, because it may name this loop or one further out, and either way this
// loop is among those it leaves. A return is counted everywhere: it ends every
// iteration of every loop it is inside.
func (w *divergeWalk) exitsVarying(body *ast.BlockStmt) bool {
	found := false
	var walk func(s ast.Stmt, div, nested bool)
	walk = func(s ast.Stmt, div, nested bool) {
		if s == nil || found {
			return
		}
		switch s := s.(type) {
		case *ast.BranchStmt:
			if div && (s.Label != nil || !nested) && s.Tok != token.FALLTHROUGH {
				found = true
			}
		case *ast.ReturnStmt:
			if div {
				found = true
			}
		case *ast.BlockStmt:
			for _, st := range s.List {
				walk(st, div, nested)
			}
		case *ast.LabeledStmt:
			walk(s.Stmt, div, nested)
		case *ast.IfStmt:
			inner := div || w.varyingExpr(s.Cond)
			walk(s.Body, inner, nested)
			walk(s.Else, inner, nested)
		case *ast.ForStmt:
			walk(s.Body, div || w.varyingExpr(s.Cond), true)
		case *ast.RangeStmt:
			walk(s.Body, div, true)
		case *ast.SwitchStmt:
			inner := div || w.switchVaries(s)
			for _, cl := range s.Body.List {
				if cc, ok := cl.(*ast.CaseClause); ok {
					for _, st := range cc.Body {
						walk(st, inner, nested)
					}
				}
			}
		}
	}
	walk(body, false, false)
	return found
}

// varyingObjects is the set of fd's variables that may hold different values
// in two threads of the same block.
//
// It is computed to a fixpoint because a variable can be read before the
// assignment that makes it vary is seen: `for { a = b; b = ctx.ThreadIdx() }`
// needs a second pass to learn that a varies too. The set only grows, and it
// is bounded by the number of objects, so the loop terminates.
//
// entry says whether fd is the kernel. A kernel's scalar parameters come from
// the launch and are the same in every thread. A device function's come from
// its caller, and the summary above has to hold at every call site, so they
// are assumed to vary -- the same pessimism readonly.go applies to a write it
// cannot see.
func (t *transpiler) varyingObjects(fd *ast.FuncDecl, entry bool) map[types.Object]bool {
	if v, ok := t.varying[fd]; ok {
		return v
	}
	vary := map[types.Object]bool{}
	if params, ok := t.paramObjects(fd); !entry || !ok {
		for _, obj := range params {
			if _, isSlice := obj.Type().Underlying().(*types.Slice); !isSlice {
				vary[obj] = true
			}
		}
	}
	w := &divergeWalk{t: t, vary: vary, use: &barrierUse{}}
	for n := -1; n != len(vary); {
		n = len(vary)
		w.mark(fd.Body, false)
	}
	if t.varying == nil {
		t.varying = map[*ast.FuncDecl]map[types.Object]bool{}
	}
	t.varying[fd] = vary
	return vary
}

// mark records the variables s may leave holding a thread-varying value.
//
// div is the control dependence: a variable assigned where only some threads
// go varies whatever it is assigned, because the threads that did not go there
// still hold what it had before. Without that, `n := 0; if ctx.ThreadIdx() ==
// 0 { n = 10 }` would look uniform -- both values are constants -- and a loop
// bounded by n would be accepted.
func (w *divergeWalk) mark(s ast.Stmt, div bool) {
	switch s := s.(type) {
	case nil:
		return
	case *ast.AssignStmt:
		for i, lhs := range s.Lhs {
			varies := div
			if !varies && len(s.Rhs) == len(s.Lhs) {
				varies = w.varyingExpr(s.Rhs[i])
			} else if !varies {
				// A shape the lowering refuses; assume the worst rather than
				// pair the wrong sides up.
				varies = true
			}
			if varies {
				w.markVarying(lhs)
			}
		}
	case *ast.IncDecStmt:
		if div {
			w.markVarying(s.X)
		}
	case *ast.DeclStmt:
		w.markDecl(s, div)
	case *ast.BlockStmt:
		for _, st := range s.List {
			w.mark(st, div)
		}
	case *ast.LabeledStmt:
		w.mark(s.Stmt, div)
	case *ast.IfStmt:
		w.mark(s.Init, div)
		inner := div || w.varyingExpr(s.Cond)
		w.mark(s.Body, inner)
		w.mark(s.Else, inner)
	case *ast.ForStmt:
		w.mark(s.Init, div)
		inner := div || w.forVaries(s)
		w.mark(s.Post, inner)
		w.mark(s.Body, inner)
	case *ast.RangeStmt:
		inner := div || w.rangeVaries(s)
		// The value is a copy of an element, so it varies with what is being
		// ranged over; the key counts from zero and varies only with the loop.
		if inner {
			w.markVarying(s.Key)
		}
		if inner || w.varyingExpr(s.X) {
			w.markVarying(s.Value)
		}
		w.mark(s.Body, inner)
	case *ast.SwitchStmt:
		w.mark(s.Init, div)
		inner := div || w.switchVaries(s)
		for _, cl := range s.Body.List {
			if cc, ok := cl.(*ast.CaseClause); ok {
				for _, st := range cc.Body {
					w.mark(st, inner)
				}
			}
		}
	}
}

// markDecl records the variables a var declaration leaves varying. One with no
// initialiser holds its zero value, which is the same in every thread.
func (w *divergeWalk) markDecl(s *ast.DeclStmt, div bool) {
	gen, ok := s.Decl.(*ast.GenDecl)
	if !ok {
		return
	}
	for _, spec := range gen.Specs {
		vs, ok := spec.(*ast.ValueSpec)
		if !ok {
			continue
		}
		for i, name := range vs.Names {
			varies := div
			if !varies && i < len(vs.Values) {
				varies = w.varyingExpr(vs.Values[i])
			}
			if varies {
				w.markVarying(name)
			}
		}
	}
}

// markVarying records the object an assignment target leaves varying.
//
// A slice is left alone deliberately. Writing s[i] from every thread does not
// make s[0] differ between them -- they all read the same address afterwards
// and see the same bytes -- so an element write says nothing about the values
// a later load produces. An array is the opposite: it is a per-thread local,
// so a divergent write to one element makes the whole array varying, because
// this lattice has nothing finer than the variable to say it about.
func (w *divergeWalk) markVarying(e ast.Expr) {
	switch e := unparen(e).(type) {
	case nil:
		return
	case *ast.Ident:
		if obj := w.t.objectOf(e); obj != nil {
			w.vary[obj] = true
		}
	case *ast.IndexExpr, *ast.SelectorExpr:
		if base := w.baseOf(e); base != nil {
			if _, isSlice := base.Type().Underlying().(*types.Slice); !isSlice {
				w.vary[base] = true
			}
		}
	}
}

// baseOf is the object an index or selector chain is ultimately rooted at.
func (w *divergeWalk) baseOf(e ast.Expr) types.Object {
	switch e := unparen(e).(type) {
	case *ast.Ident:
		return w.t.objectOf(e)
	case *ast.IndexExpr:
		return w.baseOf(e.X)
	case *ast.SelectorExpr:
		return w.baseOf(e.X)
	}
	return nil
}

// varyingExpr reports whether e may evaluate to different values in two
// threads of the same block.
//
// Everything it cannot read answers yes, which is the direction that can only
// cost a refusal. Saying no where the truth is yes would let a divergent
// barrier through, and that is the failure this whole file exists to prevent.
func (w *divergeWalk) varyingExpr(e ast.Expr) bool {
	if e == nil {
		return false
	}
	if tv, ok := w.t.info.Types[e]; ok && tv.Value != nil {
		// A constant expression, folded by go/types. gpu.WarpSize and every
		// literal arrive here.
		return false
	}
	switch e := e.(type) {
	case *ast.ParenExpr:
		return w.varyingExpr(e.X)
	case *ast.BasicLit:
		return false
	case *ast.Ident:
		obj := w.t.objectOf(e)
		if obj == nil {
			return true
		}
		if _, isSlice := obj.Type().Underlying().(*types.Slice); isSlice {
			// A slice parameter and a shared tile are both one piece of memory
			// the whole block addresses. What may differ is the index.
			return false
		}
		return w.vary[obj]
	case *ast.BinaryExpr:
		return w.varyingExpr(e.X) || w.varyingExpr(e.Y)
	case *ast.UnaryExpr:
		return w.varyingExpr(e.X)
	case *ast.IndexExpr:
		return w.varyingExpr(e.Index) || w.varyingExpr(e.X)
	case *ast.SelectorExpr:
		if sel := w.t.info.Selections[e]; sel != nil && sel.Kind() == types.FieldVal {
			return w.varyingExpr(e.X)
		}
		return true
	case *ast.CompositeLit:
		for _, el := range e.Elts {
			if w.varyingExpr(el) {
				return true
			}
		}
		return false
	case *ast.KeyValueExpr:
		return w.varyingExpr(e.Value)
	case *ast.CallExpr:
		return w.varyingCall(e)
	}
	return true
}

// varyingCall reports whether a call's result may differ between threads.
func (w *divergeWalk) varyingCall(c *ast.CallExpr) bool {
	fun := unparen(c.Fun)
	if tv, ok := w.t.info.Types[fun]; ok && tv.IsType() {
		// A conversion is as uniform as what it converts.
		return len(c.Args) != 1 || w.varyingExpr(c.Args[0])
	}
	switch fn := fun.(type) {
	case *ast.Ident:
		switch fn.Name {
		case "len":
			// A slice's length is its parameter's companion, and a tile's is
			// either a constant or the launch's: block-uniform either way.
			return false
		case "min", "max":
			for _, a := range c.Args {
				if w.varyingExpr(a) {
					return true
				}
			}
			return false
		}
		// A call into the package. Its result depends on its arguments and on
		// whatever thread-indexed vocabulary it reaches -- gpu.Ctx is passed
		// to it precisely so that it can reach some -- and summarising that
		// would be a fourth answer per declaration that nothing here needs.
		// Assuming it varies costs a refusal only where a device function's
		// result decides who reaches a barrier, which no kernel here does.
		return true
	case *ast.SelectorExpr:
		if sel := w.t.info.Selections[fn]; sel != nil && sel.Kind() == types.MethodVal && IsCtx(sel.Recv()) {
			return !ctxUniform[sel.Obj().Name()]
		}
		if id, ok := fn.X.(*ast.Ident); ok {
			if pkg, ok := w.t.info.Uses[id].(*types.PkgName); ok && pkg.Imported().Path() == GPUPkgPath {
				if _, isAtomic := gpuAtomics[fn.Sel.Name]; isAtomic {
					// An atomic hands back the value that was there before it,
					// which is whichever thread got there first.
					return true
				}
				// The math helpers are pure functions of their arguments.
				for _, a := range c.Args {
					if w.varyingExpr(a) {
						return true
					}
				}
				return false
			}
		}
	}
	return true
}
