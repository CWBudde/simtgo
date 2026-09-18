package fuzz

import (
	"fmt"
	"go/token"
)

// The statement half of the closure rendering. Where closure.go decides what
// an expression computes, this file decides in what order and how many times,
// which is the half the two renderings can most easily come to disagree about.

// loopCap bounds how many times a compiled loop may go round.
//
// Every loop the generator emits is bounded by construction, so reaching this
// is a generator bug. It is here because the alternative failure is a hung
// test rather than a reported one: a fuzz run that never returns says nothing
// and blocks everything behind it.
const loopCap = 1 << 22

func (c *compiler) stmts(list []Stmt) func(*frame) ctrl {
	fns := make([]func(*frame) ctrl, len(list))
	for i, s := range list {
		fns[i] = c.stmt(s)
	}
	switch len(fns) {
	case 0:
		return func(*frame) ctrl { return ctrl{} }
	case 1:
		return fns[0]
	}
	return func(f *frame) ctrl {
		for _, s := range fns {
			if r := s(f); r.k != ctrlNone {
				return r
			}
		}
		return ctrl{}
	}
}

func (c *compiler) stmt(s Stmt) func(*frame) ctrl {
	switch s := s.(type) {
	case *Decl:
		return effect(c.decl(s))
	case *ArrayDecl:
		return effect(c.arrayDecl(s.V))
	case *Assign:
		return effect(c.update(s.LHS, assignOp(s.Op), c.expr(s.RHS)))
	case *IncDec:
		return effect(c.incdec(s))
	case *ParAssign:
		return effect(c.parAssign(s))
	case *If:
		return c.ifStmt(s)
	case *For:
		return c.forStmt(s)
	case *Range:
		return c.rangeStmt(s)
	case *Switch:
		return c.switchStmt(s)
	case *Branch:
		return branch(s)
	case *Return:
		return c.returnStmt(s)
	case *Barrier:
		return func(f *frame) ctrl { f.ctx.SyncThreads(); return ctrl{} }
	case *SyncWarp:
		return func(f *frame) ctrl { f.ctx.SyncWarp(); return ctrl{} }
	case *SharedDecl:
		return effect(c.sharedDecl(s))
	case *Assume:
		n := s.N
		return func(f *frame) ctrl { f.ctx.AssumeBlockDim(n); return ctrl{} }
	case *CallStmt:
		return effect(discard(c.call(s.Fn, s.Args)))
	case *Discard:
		return effect(discard(c.expr(s.X)))
	}
	panic(fmt.Sprintf("fuzz: no closure for %T", s))
}

// discard runs a call for its effect and drops what it produced.
//
// Go allows a call in statement position whatever it returns, and both an
// atomic and a device function turn up there: an atomic because what it hands
// back depends on the order the threads arrived, which is not a thing two
// backends can be held to, and a device function because a kernel may want
// only what it did.
func discard(v code) func(*frame) {
	switch v.k {
	case KInvalid:
		g := as[struct{}](v)
		return func(f *frame) { _ = g(f) }
	case KF32:
		g := as[float32](v)
		return func(f *frame) { _ = g(f) }
	case KF64:
		g := as[float64](v)
		return func(f *frame) { _ = g(f) }
	case KI32:
		g := as[int32](v)
		return func(f *frame) { _ = g(f) }
	case KI64:
		g := as[int64](v)
		return func(f *frame) { _ = g(f) }
	case KU32:
		g := as[uint32](v)
		return func(f *frame) { _ = g(f) }
	case KU64:
		g := as[uint64](v)
		return func(f *frame) { _ = g(f) }
	case KBool:
		g := as[bool](v)
		return func(f *frame) { _ = g(f) }
	case KInt:
		g := as[int](v)
		return func(f *frame) { _ = g(f) }
	}
	panic("fuzz: nothing to discard of kind " + v.k.goName())
}

// effect lifts a statement with no control flow of its own.
func effect(fn func(*frame)) func(*frame) ctrl {
	return func(f *frame) ctrl { fn(f); return ctrl{} }
}

// assignOp turns a compound assignment token into the binary operator it
// applies, and token.ILLEGAL for a plain store.
func assignOp(t token.Token) token.Token {
	switch t {
	case token.ASSIGN:
		return token.ILLEGAL
	case token.ADD_ASSIGN:
		return token.ADD
	case token.SUB_ASSIGN:
		return token.SUB
	case token.MUL_ASSIGN:
		return token.MUL
	case token.QUO_ASSIGN:
		return token.QUO
	case token.REM_ASSIGN:
		return token.REM
	case token.AND_ASSIGN:
		return token.AND
	case token.OR_ASSIGN:
		return token.OR
	case token.XOR_ASSIGN:
		return token.XOR
	case token.SHL_ASSIGN:
		return token.SHL
	case token.SHR_ASSIGN:
		return token.SHR
	}
	panic("fuzz: " + t.String() + " is not an assignment")
}

func (c *compiler) decl(s *Decl) func(*frame) {
	if s.Form == DeclVarZero {
		// A declaration executed a second time -- inside a loop -- declares a
		// fresh variable holding its zero value, so the slot is cleared rather
		// than left holding the previous iteration's answer.
		return c.update(Lvalue{V: s.V}, token.ILLEGAL, c.expr(zeroLit(s.V.Kind)))
	}
	return c.update(Lvalue{V: s.V}, token.ILLEGAL, c.expr(s.Init))
}

// zeroLit is the zero value of a kind, as a literal node. Building one rather
// than writing a per-kind zero keeps the declaration path going through the
// same store as every other assignment.
func zeroLit(k Kind) Expr { return &Lit{K: k} }

func (c *compiler) arrayDecl(v *Var) func(*frame) {
	n, size := v.slot, v.Array
	switch v.Kind {
	case KF32:
		return func(f *frame) { f.bF32[n] = make([]float32, size) }
	case KF64:
		return func(f *frame) { f.bF64[n] = make([]float64, size) }
	case KI32:
		return func(f *frame) { f.bI32[n] = make([]int32, size) }
	case KI64:
		return func(f *frame) { f.bI64[n] = make([]int64, size) }
	case KU32:
		return func(f *frame) { f.bU32[n] = make([]uint32, size) }
	case KU64:
		return func(f *frame) { f.bU64[n] = make([]uint64, size) }
	case KBool:
		return func(f *frame) { f.bB[n] = make([]bool, size) }
	}
	panic("fuzz: no local array of " + v.Kind.goName())
}

func (c *compiler) sharedDecl(s *SharedDecl) func(*frame) {
	n, size, dyn := s.V.slot, s.N, s.Dyn
	switch s.V.Kind {
	case KF32:
		if dyn {
			return func(f *frame) { f.bF32[n] = f.ctx.SharedDynF32() }
		}
		return func(f *frame) { f.bF32[n] = f.ctx.SharedF32(size) }
	case KF64:
		if dyn {
			return func(f *frame) { f.bF64[n] = f.ctx.SharedDynF64() }
		}
		return func(f *frame) { f.bF64[n] = f.ctx.SharedF64(size) }
	case KI32:
		if dyn {
			return func(f *frame) { f.bI32[n] = f.ctx.SharedDynI32() }
		}
		return func(f *frame) { f.bI32[n] = f.ctx.SharedI32(size) }
	case KI64:
		if dyn {
			return func(f *frame) { f.bI64[n] = f.ctx.SharedDynI64() }
		}
		return func(f *frame) { f.bI64[n] = f.ctx.SharedI64(size) }
	case KU32:
		if dyn {
			return func(f *frame) { f.bU32[n] = f.ctx.SharedDynU32() }
		}
		return func(f *frame) { f.bU32[n] = f.ctx.SharedU32(size) }
	case KU64:
		if dyn {
			return func(f *frame) { f.bU64[n] = f.ctx.SharedDynU64() }
		}
		return func(f *frame) { f.bU64[n] = f.ctx.SharedU64(size) }
	}
	panic("fuzz: no shared tile of " + s.V.Kind.goName())
}

func (c *compiler) incdec(s *IncDec) func(*frame) {
	op := token.ADD
	if s.Op == token.DEC {
		op = token.SUB
	}
	return c.update(s.LHS, op, c.expr(oneLit(s.LHS.kind())))
}

func oneLit(k Kind) Expr {
	switch {
	case k.float():
		return &Lit{K: k, F: 1}
	case k == KU32 || k == KU64:
		return &Lit{K: k, U: 1}
	}
	return &Lit{K: k, I: 1}
}

// parAssign is the parallel assignment. Go evaluates the index operands on the
// left and the expressions on the right first, in that order, and only then
// carries the assignments out left to right -- which is what makes
// `i, y[i] = 2, 7` store into the old i's element.
//
// The three phases are literal here, and each temporary is a frame slot the
// generator reserved, so a launch's threads do not share one.
func (c *compiler) parAssign(s *ParAssign) func(*frame) {
	var phase1, phase2, phase3 []func(*frame)
	for i, l := range s.LHS {
		if l.Idx == nil {
			continue
		}
		phase1 = append(phase1, scalarStore(s.Idxs[i], token.ILLEGAL, c.expr(l.Idx)))
	}
	for i, rhs := range s.RHS {
		phase2 = append(phase2, scalarStore(s.Vals[i], token.ILLEGAL, c.expr(rhs)))
	}
	for i, l := range s.LHS {
		read := c.ref(s.Vals[i])
		if l.Idx == nil {
			phase3 = append(phase3, scalarStore(l.V, token.ILLEGAL, read))
			continue
		}
		idx := as[int](c.ref(s.Idxs[i]))
		phase3 = append(phase3, withBufStore(l.V, idx, token.ILLEGAL, read))
	}
	return func(f *frame) {
		for _, g := range phase1 {
			g(f)
		}
		for _, g := range phase2 {
			g(f)
		}
		for _, g := range phase3 {
			g(f)
		}
	}
}

// update stores rhs into l, applying op first when the assignment is a
// compound one. op is token.ILLEGAL for a plain store.
func (c *compiler) update(l Lvalue, op token.Token, rhs code) func(*frame) {
	if l.Idx == nil {
		return scalarStore(l.V, op, rhs)
	}
	return withBufStore(l.V, as[int](c.expr(l.Idx)), op, rhs)
}

// scalarStore writes a scalar slot. The read-modify-write form reads the slot
// before evaluating the right-hand side, which is the order Go's `x += e`
// has: the operand x is evaluated first.
func scalarStore(v *Var, op token.Token, rhs code) func(*frame) {
	n := v.slot
	switch v.Kind {
	case KF32:
		return storeCell(func(f *frame) []float32 { return f.sF32 }, n, op, as[float32](rhs), opFn[float32])
	case KF64:
		return storeCell(func(f *frame) []float64 { return f.sF64 }, n, op, as[float64](rhs), opFn[float64])
	case KI32:
		return storeCell(func(f *frame) []int32 { return f.sI32 }, n, op, as[int32](rhs), opFnInt[int32])
	case KI64:
		return storeCell(func(f *frame) []int64 { return f.sI64 }, n, op, as[int64](rhs), opFnInt[int64])
	case KU32:
		return storeCell(func(f *frame) []uint32 { return f.sU32 }, n, op, as[uint32](rhs), opFnInt[uint32])
	case KU64:
		return storeCell(func(f *frame) []uint64 { return f.sU64 }, n, op, as[uint64](rhs), opFnInt[uint64])
	case KInt:
		return storeCell(func(f *frame) []int { return f.sInt }, n, op, as[int](rhs), opFnInt[int])
	case KBool:
		s, g := as[bool](rhs), v.slot
		return func(f *frame) { f.sB[g] = s(f) }
	}
	panic("fuzz: no store to " + v.Kind.goName())
}

// withBufStore writes one element of a buffer, evaluating the index once.
func withBufStore(v *Var, idx func(*frame) int, op token.Token, rhs code) func(*frame) {
	g := bufOf(v)
	switch v.Kind {
	case KF32:
		return storeElem(g.(func(*frame) []float32), idx, op, as[float32](rhs), opFn[float32])
	case KF64:
		return storeElem(g.(func(*frame) []float64), idx, op, as[float64](rhs), opFn[float64])
	case KI32:
		return storeElem(g.(func(*frame) []int32), idx, op, as[int32](rhs), opFnInt[int32])
	case KI64:
		return storeElem(g.(func(*frame) []int64), idx, op, as[int64](rhs), opFnInt[int64])
	case KU32:
		return storeElem(g.(func(*frame) []uint32), idx, op, as[uint32](rhs), opFnInt[uint32])
	case KU64:
		return storeElem(g.(func(*frame) []uint64), idx, op, as[uint64](rhs), opFnInt[uint64])
	case KBool:
		get, s := g.(func(*frame) []bool), as[bool](rhs)
		return func(f *frame) { get(f)[idx(f)] = s(f) }
	case KI8:
		get, s := g.(func(*frame) []int8), as[int8](rhs)
		return func(f *frame) { get(f)[idx(f)] = s(f) }
	case KI16:
		get, s := g.(func(*frame) []int16), as[int16](rhs)
		return func(f *frame) { get(f)[idx(f)] = s(f) }
	case KU8:
		get, s := g.(func(*frame) []uint8), as[uint8](rhs)
		return func(f *frame) { get(f)[idx(f)] = s(f) }
	case KU16:
		get, s := g.(func(*frame) []uint16), as[uint16](rhs)
		return func(f *frame) { get(f)[idx(f)] = s(f) }
	}
	panic("fuzz: no element store to " + v.Kind.goName())
}

func storeCell[T any](get func(*frame) []T, n int, op token.Token, rhs func(*frame) T, mkOp func(token.Token) func(T, T) T) func(*frame) {
	if op == token.ILLEGAL {
		return func(f *frame) { get(f)[n] = rhs(f) }
	}
	apply := mkOp(op)
	return func(f *frame) {
		s := get(f)
		s[n] = apply(s[n], rhs(f))
	}
}

func storeElem[T any](get func(*frame) []T, idx func(*frame) int, op token.Token, rhs func(*frame) T, mkOp func(token.Token) func(T, T) T) func(*frame) {
	if op == token.ILLEGAL {
		return func(f *frame) { get(f)[idx(f)] = rhs(f) }
	}
	apply := mkOp(op)
	return func(f *frame) {
		s, i := get(f), idx(f)
		s[i] = apply(s[i], rhs(f))
	}
}

func (c *compiler) ifStmt(s *If) func(*frame) ctrl {
	cond := as[bool](c.expr(s.Cond))
	then := c.stmts(s.Then)
	if len(s.Else) == 0 {
		return func(f *frame) ctrl {
			if cond(f) {
				return then(f)
			}
			return ctrl{}
		}
	}
	els := c.stmts(s.Else)
	return func(f *frame) ctrl {
		if cond(f) {
			return then(f)
		}
		return els(f)
	}
}

func (c *compiler) forStmt(s *For) func(*frame) ctrl {
	var init, post func(*frame) ctrl
	if s.Init != nil {
		init = c.stmt(s.Init)
	}
	if s.Post != nil {
		post = c.stmt(s.Post)
	}
	var cond func(*frame) bool
	if s.Cond != nil {
		cond = as[bool](c.expr(s.Cond))
	}
	body := c.stmts(s.Body)
	label := s.Label
	return func(f *frame) ctrl {
		if init != nil {
			init(f)
		}
		for n := 0; ; n++ {
			if n > loopCap {
				panic("fuzz: generated loop did not terminate")
			}
			if cond != nil && !cond(f) {
				return ctrl{}
			}
			r := body(f)
			if out, stop := loopCtrl(r, label); stop {
				return out
			}
			if post != nil {
				post(f)
			}
		}
	}
}

// loopCtrl decides what a loop does with the control its body produced, and
// what it passes outwards. A labelled branch naming another loop travels on;
// one naming this loop, or naming none, is this loop's to act on.
func loopCtrl(r ctrl, label string) (out ctrl, stop bool) {
	switch r.k {
	case ctrlNone:
		return ctrl{}, false
	case ctrlContinue:
		if r.label == "" || r.label == label {
			return ctrl{}, false
		}
		return r, true
	case ctrlBreak:
		if r.label == "" || r.label == label {
			return ctrl{}, true
		}
		return r, true
	}
	return r, true // a return leaves every loop
}

func (c *compiler) rangeStmt(s *Range) func(*frame) ctrl {
	body := c.stmts(s.Body)
	label := s.Label
	if s.Over.Buf != nil {
		return c.rangeBuf(s, body, label)
	}
	return c.rangeInt(s, body, label)
}

// rangeBuf is `for i := range buf` and `for i, v := range buf`. The key counts
// to the buffer's length, which the whole block agrees on.
func (c *compiler) rangeBuf(s *Range, body func(*frame) ctrl, label string) func(*frame) ctrl {
	length := as[int](c.length(s.Over.Buf))
	setKey := scalarStoreFrom(s.Key)
	var setVal func(*frame)
	if s.Val != nil {
		// The value is a copy of the element at the key the loop has just set,
		// which is how Go's range hands it over.
		setVal = scalarStore(s.Val, token.ILLEGAL, c.index(&Index{Base: s.Over.Buf, Idx: &Ref{V: s.Key}}))
	}
	return func(f *frame) ctrl {
		n := length(f)
		for i := 0; i < n; i++ {
			setKey(f, i)
			if setVal != nil {
				setVal(f)
			}
			if out, stop := loopCtrl(body(f), label); stop {
				return out
			}
		}
		return ctrl{}
	}
}

// scalarStoreFrom writes a Go int into the key variable, converting to the
// variable's own kind -- which for a range over an integer is the kind of the
// thing being ranged over, and int for a range over a buffer.
func scalarStoreFrom(v *Var) func(*frame, int) {
	n := v.slot
	switch v.Kind {
	case KInt:
		return func(f *frame, i int) { f.sInt[n] = i }
	case KI32:
		return func(f *frame, i int) { f.sI32[n] = int32(i) }
	case KI64:
		return func(f *frame, i int) { f.sI64[n] = int64(i) }
	case KU32:
		return func(f *frame, i int) { f.sU32[n] = uint32(i) }
	case KU64:
		return func(f *frame, i int) { f.sU64[n] = uint64(i) }
	}
	panic("fuzz: no range key of kind " + v.Kind.goName())
}

// rangeInt is `for i := range n`, where i takes n's own type. That the counter
// keeps the ranged type rather than becoming an int is the whole point of
// generating this over an int64.
func (c *compiler) rangeInt(s *Range, body func(*frame) ctrl, label string) func(*frame) ctrl {
	bound := c.expr(s.Over.N)
	setKey := scalarStoreFrom(s.Key)
	var limit func(*frame) int
	switch bound.k {
	case KInt:
		g := as[int](bound)
		limit = g
	case KI32:
		g := as[int32](bound)
		limit = func(f *frame) int { return int(g(f)) }
	case KI64:
		g := as[int64](bound)
		limit = func(f *frame) int { return int(g(f)) }
	case KU32:
		g := as[uint32](bound)
		limit = func(f *frame) int { return int(g(f)) }
	case KU64:
		g := as[uint64](bound)
		limit = func(f *frame) int { return int(g(f)) }
	default:
		panic("fuzz: cannot range over " + bound.k.goName())
	}
	return func(f *frame) ctrl {
		n := limit(f)
		for i := 0; i < n; i++ {
			setKey(f, i)
			if out, stop := loopCtrl(body(f), label); stop {
				return out
			}
		}
		return ctrl{}
	}
}

func branch(s *Branch) func(*frame) ctrl {
	switch s.Tok {
	case token.BREAK:
		r := ctrl{k: ctrlBreak, label: s.Label}
		return func(*frame) ctrl { return r }
	case token.CONTINUE:
		r := ctrl{k: ctrlContinue, label: s.Label}
		return func(*frame) ctrl { return r }
	}
	panic("fuzz: fallthrough is carried by the clause, not by a statement")
}

func (c *compiler) returnStmt(s *Return) func(*frame) ctrl {
	if s.X == nil {
		return func(*frame) ctrl { return ctrl{k: ctrlReturn} }
	}
	set := retStore(s.X.kind(), c.expr(s.X))
	return func(f *frame) ctrl {
		set(f)
		return ctrl{k: ctrlReturn}
	}
}

func retStore(k Kind, v code) func(*frame) {
	switch k {
	case KF32:
		s := as[float32](v)
		return func(f *frame) { f.rF32 = s(f) }
	case KF64:
		s := as[float64](v)
		return func(f *frame) { f.rF64 = s(f) }
	case KI32:
		s := as[int32](v)
		return func(f *frame) { f.rI32 = s(f) }
	case KI64:
		s := as[int64](v)
		return func(f *frame) { f.rI64 = s(f) }
	case KU32:
		s := as[uint32](v)
		return func(f *frame) { f.rU32 = s(f) }
	case KU64:
		s := as[uint64](v)
		return func(f *frame) { f.rU64 = s(f) }
	case KBool:
		s := as[bool](v)
		return func(f *frame) { f.rB = s(f) }
	case KInt:
		s := as[int](v)
		return func(f *frame) { f.rInt = s(f) }
	}
	panic("fuzz: no result of kind " + k.goName())
}

// clause is one compiled case.
type clause struct {
	body func(*frame) ctrl
	fall bool
}

func (c *compiler) switchStmt(s *Switch) func(*frame) ctrl {
	clauses := make([]clause, len(s.Cases))
	def := -1
	for i, cc := range s.Cases {
		clauses[i] = clause{body: c.stmts(cc.Body), fall: cc.Fallthrough}
		if len(cc.Vals) == 0 {
			def = i
		}
	}
	pick := c.switchPick(s)
	return func(f *frame) ctrl {
		sel := pick(f)
		if sel < 0 {
			sel = def
		}
		for sel >= 0 && sel < len(clauses) {
			r := clauses[sel].body(f)
			if r.k == ctrlBreak && r.label == "" {
				// A bare break inside a switch leaves the switch, in Go and in
				// the C switch this lowers to alike.
				return ctrl{}
			}
			if r.k != ctrlNone {
				return r
			}
			if !clauses[sel].fall {
				return ctrl{}
			}
			sel++
		}
		return ctrl{}
	}
}

// switchPick chooses a clause by index, or -1 for none, evaluating the tag
// exactly once -- which is what the transpiler's temporary is for, and what
// makes a tag that calls a device function mean the same on both sides.
func (c *compiler) switchPick(s *Switch) func(*frame) int {
	if s.Tag == nil {
		conds := make([][]func(*frame) bool, len(s.Cases))
		for i, cc := range s.Cases {
			for _, v := range cc.Vals {
				conds[i] = append(conds[i], as[bool](c.expr(v)))
			}
		}
		return func(f *frame) int {
			for i, cs := range conds {
				for _, cond := range cs {
					if cond(f) {
						return i
					}
				}
			}
			return -1
		}
	}
	tag := c.expr(s.Tag)
	switch tag.k {
	case KI32:
		return switchOn(as[int32](tag), caseVals(c, s, as[int32]))
	case KI64:
		return switchOn(as[int64](tag), caseVals(c, s, as[int64]))
	case KU32:
		return switchOn(as[uint32](tag), caseVals(c, s, as[uint32]))
	case KU64:
		return switchOn(as[uint64](tag), caseVals(c, s, as[uint64]))
	case KInt:
		return switchOn(as[int](tag), caseVals(c, s, as[int]))
	case KBool:
		return switchOn(as[bool](tag), caseVals(c, s, as[bool]))
	}
	panic("fuzz: cannot switch on " + tag.k.goName())
}

// caseVals compiles every case expression at the tag's type. It is a function
// rather than a method because Go has no generic methods.
func caseVals[T comparable](c *compiler, s *Switch, conv func(code) func(*frame) T) [][]func(*frame) T {
	out := make([][]func(*frame) T, len(s.Cases))
	for i, cc := range s.Cases {
		for _, v := range cc.Vals {
			out[i] = append(out[i], conv(c.expr(v)))
		}
	}
	return out
}

func switchOn[T comparable](tag func(*frame) T, vals [][]func(*frame) T) func(*frame) int {
	return func(f *frame) int {
		got := tag(f)
		for i, vs := range vals {
			for _, v := range vs {
				if got == v(f) {
					return i
				}
			}
		}
		return -1
	}
}
