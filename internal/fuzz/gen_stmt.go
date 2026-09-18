package fuzz

import (
	"fmt"
	"go/token"
)

// Statement generation.
//
// Two invariants run through all of it. The first is the divergence rule: a
// barrier and a masked warp primitive go only where every thread of the block
// arrives, so g.uniform is checked before one is built rather than after, and
// a variable that has to stay block-uniform is never assigned under a varying
// condition. The second is that nothing races: a global element is written
// only by the thread that owns it, a shared tile only during its fill, and the
// atomics only touch a buffer nothing else does.

// loopInfo is an enclosing loop. used records whether anything named its
// label, because Go rejects a label nothing branches to -- so the label is put
// on the loop only once something has asked for it.
type loopInfo struct {
	label string
	used  bool
}

func (g *gen) stmts(n int) []Stmt {
	var out []Stmt
	for range n {
		if g.budget <= 0 {
			break
		}
		g.budget--
		out = append(out, g.stmt()...)
	}
	return out
}

// stmt produces one statement, with whatever the form needed emitted ahead of
// it. g.pending is cleared around every attempt, so a form that gave up cannot
// leave its preamble behind for the next one.
func (g *gen) stmt() []Stmt {
	for range 8 {
		g.pending = nil
		if s := g.tryStmt(); s != nil {
			out := append(g.pending, s)
			g.pending = nil
			return out
		}
	}
	g.pending = nil
	return []Stmt{g.declStmt()}
}

// tryStmt picks a statement form and returns nil when this position cannot
// have one -- no variable of the right kind in scope, no enclosing loop to
// break out of, a barrier where the threads disagree. The caller retries,
// which is cheaper than threading every precondition into the choice.
func (g *gen) tryStmt() Stmt {
	switch g.r.IntN(22) {
	case 0, 1, 2:
		return g.declStmt()
	case 3, 4, 5, 6:
		return g.assignStmt()
	case 7:
		return g.incDecStmt()
	case 8:
		return g.parAssignStmt()
	case 9, 10:
		return g.ifStmt()
	case 11, 12:
		return g.forStmt()
	case 13:
		return g.rangeStmt()
	case 14:
		return g.switchStmt()
	case 15:
		return g.arrayDeclStmt()
	case 16:
		return g.branchStmt()
	case 17:
		return g.barrierStmt()
	case 18:
		return g.callStmt()
	case 19:
		return g.returnStmt()
	case 20:
		return g.arrayDeclStmt()
	}
	return g.assignStmt()
}

// --- declarations and assignments ------------------------------------------

func (g *gen) declStmt() Stmt {
	k := pick(g.r, g.valueKinds())
	// A variable is uniform only where it can stay uniform: a declaration
	// under a varying condition leaves whatever it holds varying, because the
	// threads that went elsewhere still hold what it had before.
	uniform := g.uniform && g.r.IntN(2) == 0
	v := g.local(k, uniform)
	prev := g.uni
	g.uni = uniform
	init := g.expr(k, 2)
	g.uni = prev

	form := pick(g.r, []DeclForm{DeclShort, DeclVarTyped, DeclVarZero})
	if form == DeclShort && untypedConst(init) && k != KInt && k != KBool && k != KF64 {
		// `x := 1 + 2` declares an int whatever was meant. The var form says
		// the type outright, and is a different AST node the transpiler has to
		// handle separately -- which is why all three spellings are generated.
		form = DeclVarTyped
	}
	if form == DeclVarZero {
		init = nil
	}
	g.declare(v)
	return &Decl{V: v, Init: init, Form: form}
}

func (g *gen) arrayDeclStmt() Stmt {
	k := pick(g.r, g.storeKinds())
	v := &Var{Name: g.freshName(g.r.IntN(4) == 0), Kind: k, Vary: true}
	v.Array = 1 + g.r.IntN(6)
	v.Len = v.Array
	g.fn.frame.alloc(v)
	g.declare(v)
	return &ArrayDecl{V: v}
}

// assignStmt stores into a scalar or into an element the thread owns.
func (g *gen) assignStmt() Stmt {
	if g.r.IntN(3) == 0 {
		if s := g.elemAssign(); s != nil {
			return s
		}
	}
	k := pick(g.r, g.valueKinds())
	vars := g.writableScalars(k)
	if len(vars) == 0 {
		return nil
	}
	v := pick(g.r, vars)
	prev := g.uni
	g.uni = !v.Vary
	rhs := g.expr(k, 2)
	op := token.ASSIGN
	if g.r.IntN(2) == 0 {
		op = g.compound(k, &rhs)
	}
	g.uni = prev
	return &Assign{LHS: Lvalue{V: v}, Op: op, RHS: rhs}
}

// compound picks a compound assignment operator and rewrites the right-hand
// side where the operator needs it: a divisor has to be non-zero and a shift
// count has to fit the width C gives the type.
func (g *gen) compound(k Kind, rhs *Expr) token.Token {
	if k == KBool {
		return token.ASSIGN
	}
	if k.float() {
		op := pick(g.r, []token.Token{token.ADD_ASSIGN, token.SUB_ASSIGN, token.MUL_ASSIGN, token.QUO_ASSIGN})
		if op == token.QUO_ASSIGN {
			*rhs = g.anchor(k)
		}
		return op
	}
	op := pick(g.r, []token.Token{
		token.ADD_ASSIGN, token.SUB_ASSIGN, token.MUL_ASSIGN, token.QUO_ASSIGN, token.REM_ASSIGN,
		token.AND_ASSIGN, token.OR_ASSIGN, token.XOR_ASSIGN, token.SHL_ASSIGN, token.SHR_ASSIGN,
	})
	switch op {
	case token.QUO_ASSIGN, token.REM_ASSIGN:
		*rhs = g.safeDivisor(k, 1)
	case token.SHL_ASSIGN, token.SHR_ASSIGN:
		*rhs = g.shiftCount(k, 1)
	}
	return op
}

// elemAssign writes one element of something the thread owns: a local array,
// which is per-thread storage, or its own slot of an output buffer.
func (g *gen) elemAssign() Stmt {
	var targets []*Var
	targets = append(targets, g.visible(func(v *Var) bool { return v.Array > 0 })...)
	if g.inTail {
		targets = append(targets, g.outBufs...)
	}
	if len(targets) == 0 {
		return nil
	}
	b := pick(g.r, targets)
	var idx Expr
	if b.Array > 0 {
		idx = g.wrapIndex(g.indexSeed(), b)
	} else {
		// An output element belongs to exactly one thread, which is what keeps
		// two of them from writing the same one.
		idx = &Ref{V: g.gid}
	}
	rhs := g.expr(b.Kind, 2)
	op := token.ASSIGN
	if g.r.IntN(3) == 0 {
		op = g.compound(b.Kind, &rhs)
	}
	return &Assign{LHS: Lvalue{V: b, Idx: idx}, Op: op, RHS: rhs}
}

func (g *gen) incDecStmt() Stmt {
	k := pick(g.r, []Kind{KInt, KI32, KI64, KU32, KU64, KF32})
	vars := g.writableScalars(k)
	if len(vars) == 0 {
		return nil
	}
	op := token.INC
	if g.r.IntN(2) == 0 {
		op = token.DEC
	}
	return &IncDec{LHS: Lvalue{V: pick(g.r, vars)}, Op: op}
}

// parAssignStmt is `a, b = e, f`, whose point is that everything is evaluated
// before anything is stored.
func (g *gen) parAssignStmt() Stmt {
	k := pick(g.r, g.valueKinds())
	vars := g.writableScalars(k)
	if len(vars) < 2 {
		return nil
	}
	a, b := pick(g.r, vars), pick(g.r, vars)
	if a == b {
		return nil
	}
	s := &ParAssign{LHS: []Lvalue{{V: a}, {V: b}}, Idxs: []*Var{nil, nil}}
	prev := g.uni
	g.uni = !a.Vary || !b.Vary
	s.RHS = []Expr{g.expr(k, 2), g.expr(k, 2)}
	g.uni = prev

	// `i, y[i] = 2, 7` is the shape worth generating: the index is evaluated
	// before anything is stored, so the element written is the old i's. An
	// element of a local array is the target, which is per-thread storage and
	// so cannot be written by anybody else.
	if arrs := g.visible(func(v *Var) bool { return v.Array > 0 && v.Kind == k }); len(arrs) > 0 && g.r.IntN(2) == 0 {
		arr := pick(g.r, arrs)
		idx := &Var{Kind: KInt}
		g.fn.frame.alloc(idx)
		s.LHS[1] = Lvalue{V: arr, Idx: g.wrapIndex(g.pureExpr(KInt, 1), arr)}
		s.Idxs[1] = idx
	}

	for range 2 {
		t := &Var{Kind: k}
		g.fn.frame.alloc(t)
		s.Vals = append(s.Vals, t)
	}
	return s
}

// writableScalars are the visible variables a store may target here.
//
// A variable declared uniform may only be written where every thread of the
// block is, because a store under a varying condition leaves it holding
// different values -- which is exactly what diverge.go's lattice says, and
// what a barrier guarded by it would then be undefined under.
func (g *gen) writableScalars(k Kind) []*Var {
	return g.visible(func(v *Var) bool {
		// The two thread numbers are left out on purpose: every global store
		// goes to the slot one of them names, and that is what keeps two
		// threads from writing the same element. A kernel that moved its own
		// gid would be racing with itself.
		if v == g.gid || v == g.tid || v.Fixed {
			return false
		}
		return !v.buffer() && v.Kind == k && (v.Vary || g.uniform)
	})
}

// --- control flow ----------------------------------------------------------

func (g *gen) ifStmt() Stmt {
	var cond Expr
	if g.uniform && g.r.IntN(2) == 0 {
		cond = g.uniExpr(KBool, 2)
	} else {
		cond = g.expr(KBool, 2)
	}
	prev := g.uniform
	g.uniform = prev && !cond.varies()
	g.push()
	then := g.stmts(1 + g.r.IntN(2))
	g.pop()
	var els []Stmt
	switch g.r.IntN(4) {
	case 0:
		g.push()
		els = g.stmts(1 + g.r.IntN(2))
		g.pop()
	case 1:
		// An else-if, which is one *ast.IfStmt in the else branch rather than
		// a block containing one. It is generated under this if's divergence
		// rather than the enclosing one: only the threads that failed the
		// condition reach an else, so it sits inside the same site the then
		// branch does.
		if inner := g.ifStmt(); inner != nil {
			els = []Stmt{inner}
		}
	}
	g.uniform = prev
	if len(then) == 0 && len(els) == 0 {
		return nil
	}
	if len(then) == 0 {
		then = []Stmt{&Decl{V: g.declared(KInt), Init: &Lit{K: KInt}, Form: DeclShort}}
	}
	return &If{Cond: cond, Then: then, Else: els}
}

// declared makes a variable and records it, for the places that need a
// statement rather than an empty block. Go accepts an empty block and C++
// accepts an empty one too, but an if with nothing in either arm says nothing.
func (g *gen) declared(k Kind) *Var {
	v := g.local(k, false)
	g.declare(v)
	return v
}

// maxLoopDepth bounds how deeply loops may nest.
//
// Trip counts multiply, and a corpus is worth more than one enormous program:
// five loops of seven iterations around a body is sixteen thousand executions
// per thread, and at two hundred threads that one program costs more than the
// hundred around it put together. The bound is on the depth rather than on the
// counts because a shallow loop with a real trip count is the interesting one.
const maxLoopDepth = 3

func (g *gen) forStmt() Stmt {
	if len(g.loops) >= maxLoopDepth {
		return nil
	}
	k := pick(g.r, []Kind{KInt, KInt, KI32, KI64})
	prevNoWarp := g.noWarp
	g.noWarp = true
	bound := g.loopBound(k)
	g.noWarp = prevNoWarp
	prev := g.uniform
	inner := prev && !bound.varies()

	// The bound is hoisted into a variable of its own, and neither it nor the
	// counter may be assigned to afterwards. Go re-evaluates a for condition
	// every time round, so a body that could reach into the bound -- or reset
	// the counter -- would be writing a loop whose trip count is not the
	// generator's to know.
	//
	// It is declared here, in the enclosing scope, which is why this form
	// never gives up afterwards: a name brought into scope for a statement
	// that was then thrown away is a name the rest of the block would go on
	// using and the source would never declare.
	var pre Stmt
	if !isConst(bound) {
		bv := &Var{Name: g.freshName(false), Kind: k, Vary: bound.varies(), Fixed: true}
		g.fn.frame.alloc(bv)
		form := DeclShort
		if untypedConst(bound) && k != KInt {
			form = DeclVarTyped
		}
		pre = &Decl{V: bv, Init: bound, Form: form}
		g.declare(bv)
		bound = &Ref{V: bv}
	}

	// The loop variable never shadows, although a declaration elsewhere may.
	// A for clause is the one place where it would change what the generator
	// already built: the variable's scope covers the condition, so a bound
	// written against an outer name of the same spelling would silently become
	// a bound on the counter.
	v := &Var{Name: g.freshName(false), Kind: k, Vary: !inner, Fixed: true}
	g.fn.frame.alloc(v)

	g.push()
	g.declare(v)
	g.uniform = inner
	loop := &loopInfo{}
	if g.r.IntN(3) == 0 {
		loop.label = g.newLabel()
	}
	g.loops = append(g.loops, loop)
	body := g.stmts(1 + g.r.IntN(3))
	if len(body) == 0 {
		// The budget ran out. The loop is emitted anyway, with something in
		// it, because the bound is already declared and abandoning the form
		// now would leave that name in scope with nothing declaring it.
		body = []Stmt{g.declStmt()}
	}
	g.loops = g.loops[:len(g.loops)-1]
	g.uniform = prev
	g.pop()

	label := ""
	if loop.used {
		label = loop.label
	}
	init := Stmt(&Decl{V: v, Init: g.zeroOf(k), Form: DeclShort})
	post := Stmt(&IncDec{LHS: Lvalue{V: v}, Op: token.INC})
	if g.r.IntN(3) == 0 {
		post = &Assign{LHS: Lvalue{V: v}, Op: token.ADD_ASSIGN, RHS: g.smallOf(k, 1+g.r.IntN(2))}
	}
	if pre != nil {
		g.pending = append(g.pending, pre)
	}
	return &For{
		Init:  init,
		Cond:  &Binary{Op: token.LSS, X: &Ref{V: v}, Y: bound, K: KBool},
		Post:  post,
		Body:  body,
		Label: label,
	}
}

// newLabel is a label unique within the function, which is the scope Go gives
// one.
func (g *gen) newLabel() string {
	g.labelN++
	return fmt.Sprintf("L%d", g.labelN)
}

// loopBound is a small, non-negative trip count. A loop whose bound came from
// the input would run for as long as the input said, and a fuzz run that never
// returns reports nothing.
func (g *gen) loopBound(k Kind) Expr {
	mask := 7
	if len(g.loops) > 0 {
		mask = 3
	}
	if g.r.IntN(2) == 0 {
		return g.smallOf(k, 1+g.r.IntN(mask))
	}
	var seed Expr
	if g.uniform && g.r.IntN(2) == 0 {
		seed = g.uniExpr(k, 1)
	} else {
		seed = g.expr(k, 1)
	}
	return &Binary{Op: token.AND, X: seed, Y: g.smallOf(k, mask), K: k}
}

// zeroOf and smallOf spell a constant of kind k so that a short declaration in
// a for clause gets the type meant: `i := 0` is an int, so anything else needs
// the conversion. A var declaration is not an option -- a for clause takes a
// simple statement, and that is not one.
func (g *gen) zeroOf(k Kind) Expr { return g.smallOf(k, 0) }

func (g *gen) smallOf(k Kind, n int) Expr {
	lit := Expr(&Lit{K: k, I: int64(n)})
	if k == KU32 || k == KU64 {
		lit = &Lit{K: k, U: uint64(n)}
	}
	if k == KInt {
		return lit
	}
	return &Conv{To: k, X: lit}
}

func (g *gen) rangeStmt() Stmt {
	if len(g.loops) >= maxLoopDepth {
		return nil
	}
	if g.r.IntN(2) == 0 {
		return g.rangeBuf()
	}
	return g.rangeInt()
}

func (g *gen) rangeBuf() Stmt {
	// A nested range counts to a length nobody chose, so only the outermost
	// one is allowed a long buffer.
	limit := 32
	if len(g.loops) > 0 {
		limit = 6
	}
	// Only a readable buffer: `for i, v := range out` would hand this thread
	// the element another one is writing.
	var small []*Var
	for _, b := range g.readable() {
		if b.Len > 0 && b.Len <= limit {
			small = append(small, b)
		}
	}
	if len(small) == 0 {
		return nil
	}
	b := pick(g.r, small)
	prev := g.uniform
	g.push()
	// Ranging over a buffer counts to a length the whole block agrees on, so
	// the loop itself does not diverge. The key is declared before the value's
	// name is drawn, so the two cannot come out the same.
	key := &Var{Name: g.freshName(false), Kind: KInt, Vary: !prev}
	g.fn.frame.alloc(key)
	g.declare(key)
	var val *Var
	if !b.Kind.narrow() && g.r.IntN(2) == 0 {
		// A narrow element would be a local of a storage-only type, which the
		// subset refuses; the key form is the only one available there.
		val = &Var{Name: g.freshName(false), Kind: b.Kind, Vary: true}
		g.fn.frame.alloc(val)
		g.declare(val)
	}
	loop := &loopInfo{}
	g.loops = append(g.loops, loop)
	body := g.stmts(1 + g.r.IntN(2))
	g.loops = g.loops[:len(g.loops)-1]
	g.pop()
	if len(body) == 0 {
		return nil
	}
	label := ""
	if loop.used {
		label = loop.label
	}
	return &Range{Key: key, Val: val, Over: RangeOver{Buf: b}, Body: body, Label: label}
}

// rangeInt is `for i := range n`, where the counter takes n's own type. That
// it does, rather than becoming an int, is the whole reason to generate this
// over an int64.
func (g *gen) rangeInt() Stmt {
	k := pick(g.r, []Kind{KInt, KI32, KI64})
	mask := 7
	if len(g.loops) > 0 {
		mask = 3
	}
	n := &Binary{Op: token.AND, X: g.expr(k, 1), Y: g.smallOf(k, mask), K: k}
	prev := g.uniform
	inner := prev && !n.varies()
	key := &Var{Name: g.freshName(false), Kind: k, Vary: !inner}
	g.fn.frame.alloc(key)
	g.push()
	g.declare(key)
	g.uniform = inner
	loop := &loopInfo{}
	g.loops = append(g.loops, loop)
	body := g.stmts(1 + g.r.IntN(2))
	g.loops = g.loops[:len(g.loops)-1]
	g.uniform = prev
	g.pop()
	if len(body) == 0 {
		return nil
	}
	label := ""
	if loop.used {
		label = loop.label
	}
	return &Range{Key: key, Over: RangeOver{N: n}, Body: body, Label: label}
}

// switchStmt generates both lowerings.
//
// A tag that is integral with constant cases becomes a C switch, where a
// fallthrough is honoured by leaving the closing break out. Anything else
// becomes the if/else chain Go's semantics describe, and there a bare break
// is refused -- in C it would leave the enclosing loop -- so this generates
// neither a break nor a fallthrough in that form.
func (g *gen) switchStmt() Stmt {
	if g.r.IntN(2) == 0 {
		return g.constSwitch()
	}
	return g.chainSwitch()
}

func (g *gen) constSwitch() Stmt {
	k := pick(g.r, []Kind{KInt, KI32, KI64, KU32})
	tag := g.pureExpr(k, 2)
	prev, prevSwitch := g.uniform, g.inSwitch
	g.uniform = prev && !tag.varies()
	g.inSwitch = true

	n := 2 + g.r.IntN(2)
	s := &Switch{Tag: tag, CSwitch: true}
	for i := range n {
		// Distinct constants: Go rejects a duplicate case in a constant switch.
		val := g.smallConst(k, int64(i))
		g.push()
		body := g.stmts(1 + g.r.IntN(2))
		g.pop()
		fall := i < n-1 && g.r.IntN(4) == 0
		s.Cases = append(s.Cases, Case{Vals: []Expr{val}, Body: body, Fallthrough: fall})
	}
	if g.r.IntN(2) == 0 {
		g.push()
		s.Cases = append(s.Cases, Case{Body: g.stmts(1 + g.r.IntN(2))})
		g.pop()
	}
	g.uniform, g.inSwitch = prev, prevSwitch
	return s
}

// smallConst is a case label. It is a plain literal rather than a conversion
// because a C switch needs its cases to fold to constants, which a conversion
// of a literal still does, but the plain form is what a kernel would write.
func (g *gen) smallConst(k Kind, v int64) Expr {
	if k == KU32 || k == KU64 {
		return &Lit{K: k, U: uint64(v)}
	}
	return &Lit{K: k, I: v}
}

func (g *gen) chainSwitch() Stmt {
	prev, prevSwitch := g.uniform, g.inSwitch
	g.inSwitch = true
	s := &Switch{}
	// Every condition first, then every body. A tagless switch diverges when
	// any of its cases does, not only the ones before the body in question: a
	// thread reaching the third clause has already disagreed with its
	// neighbours at the first, so the whole statement is one site.
	n := 1 + g.r.IntN(3)
	conds := make([]Expr, n)
	varying := false
	prevNoWarp := g.noWarp
	g.noWarp = true
	for i := range conds {
		conds[i] = g.expr(KBool, 2)
		varying = varying || conds[i].varies()
	}
	g.noWarp = prevNoWarp
	g.uniform = prev && !varying
	for _, cond := range conds {
		g.push()
		body := g.stmts(1 + g.r.IntN(2))
		g.pop()
		s.Cases = append(s.Cases, Case{Vals: []Expr{cond}, Body: body})
	}
	if g.r.IntN(2) == 0 {
		g.push()
		s.Cases = append(s.Cases, Case{Body: g.stmts(1 + g.r.IntN(2))})
		g.pop()
	}
	g.uniform, g.inSwitch = prev, prevSwitch
	return s
}

// branchStmt is a break or a continue.
//
// It is generated only where the block still agrees -- a branch under a
// varying condition is what makes a loop's trip count differ between threads,
// and a barrier inside such a loop is refused. A bare one is refused inside a
// switch that lowered to an if/else chain and would mean the switch rather
// than the loop inside a C switch, so a labelled one is used there.
func (g *gen) branchStmt() Stmt {
	if len(g.loops) == 0 {
		return nil
	}
	if g.sync && !g.uniform {
		return nil
	}
	tok := token.BREAK
	if g.r.IntN(2) == 0 {
		tok = token.CONTINUE
	}
	if !g.inSwitch && g.r.IntN(2) == 0 {
		return &Branch{Tok: tok}
	}
	// A labelled branch, which the emitter turns into a goto past the loop or
	// to the end of its body.
	outer := g.loops[g.r.IntN(len(g.loops))]
	if outer.label == "" {
		if g.inSwitch {
			return nil
		}
		return &Branch{Tok: tok}
	}
	outer.used = true
	return &Branch{Tok: tok, Label: outer.label}
}

func (g *gen) barrierStmt() Stmt {
	if !g.sync || !g.uniform || !g.fn.Ctx {
		return nil
	}
	if g.warpOK && g.r.IntN(3) == 0 {
		return &SyncWarp{}
	}
	return &Barrier{}
}

// callStmt calls a helper for its effect. A ctx-taking helper may hold a
// barrier, which is a barrier at this call site, so it goes only where the
// whole block is.
func (g *gen) callStmt() Stmt {
	if !g.fn.Ctx || !g.uniform || !g.sync {
		return nil
	}
	var fns []*Func
	for _, fn := range g.helpers {
		if fn.Ctx {
			fns = append(fns, fn)
		}
	}
	if len(fns) == 0 {
		return nil
	}
	fn := pick(g.r, fns)
	args := make([]Expr, len(fn.Params))
	for i, p := range fn.Params {
		if p.Slice {
			b := g.bufferArg(p.Kind)
			if b == nil {
				return nil
			}
			args[i] = &Ref{V: b}
			continue
		}
		args[i] = g.expr(p.Kind, 1)
	}
	return &CallStmt{Fn: fn, Args: args}
}

// returnStmt is a bare return from the kernel.
//
// Only where the kernel has no barrier at all: a thread that returned never
// arrives at one, which is the third of the three divergence rules, and the
// generator stays inside it by not writing both shapes into one kernel.
func (g *gen) returnStmt() Stmt {
	if g.sync || g.fn.Result != KInvalid || g.r.IntN(3) != 0 {
		return nil
	}
	return &Return{}
}
