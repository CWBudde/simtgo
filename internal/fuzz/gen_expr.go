package fuzz

import (
	"go/token"
)

// Expression generation.
//
// Everything here is asked for by kind and built to fit, which is the whole
// reason the yield is what it is: the alternative -- build an expression and
// see whether the subset takes it -- is refused essentially always. The
// awkward cases are the ones where Go's own rules bite before the subset's
// does, and each is handled where it arises: an untyped constant has a default
// type that a short declaration would give the variable, a constant that
// overflows its kind is a compile error rather than a wrapping, and a constant
// division by zero is one too.

// expr builds an expression of kind k, with d levels of nesting left.
func (g *gen) expr(k Kind, d int) Expr {
	if d <= 0 {
		return g.leaf(k)
	}
	switch {
	case k == KBool:
		return g.boolExpr(d)
	case k.float():
		return g.floatExpr(k, d)
	default:
		return g.intExpr(k, d)
	}
}

// uniExpr builds a block-uniform expression: one every thread of the block
// agrees about, and therefore one a barrier may be guarded by.
func (g *gen) uniExpr(k Kind, d int) Expr {
	prev := g.uni
	g.uni = true
	e := g.expr(k, d)
	g.uni = prev
	return e
}

// pureExpr builds an expression free of calls, which is what a switch tag and
// a parallel assignment's index need: the generator evaluates those twice in
// one rendering and once in the other, and a call would make the two differ.
func (g *gen) pureExpr(k Kind, d int) Expr {
	prev := g.pure
	g.pure = true
	e := g.expr(k, d)
	g.pure = prev
	return e
}

// leaf is an expression with no operator in it.
func (g *gen) leaf(k Kind) Expr {
	var opts []func() Expr

	if vars := g.scalars(k); len(vars) > 0 {
		opts = append(opts, func() Expr { return &Ref{V: pick(g.r, vars)} })
		opts = append(opts, func() Expr { return &Ref{V: pick(g.r, vars)} })
	}
	if bufs := g.buffers(k); len(bufs) > 0 {
		opts = append(opts, func() Expr {
			b := pick(g.r, bufs)
			return &Index{Base: b, Idx: g.wrapIndex(g.indexSeed(), b)}
		})
	}
	if k == KInt && g.fn.Ctx {
		opts = append(opts, func() Expr { return g.position() })
		if all := g.allBuffers(); len(all) > 0 {
			opts = append(opts, func() Expr { return &Len{Base: pick(g.r, all)} })
		}
	}
	if len(opts) == 0 || g.r.IntN(4) == 0 {
		return g.lit(k)
	}
	return pick(g.r, opts)()
}

// lit is a constant of kind k.
//
// The magnitudes are small on purpose. A subtree of nothing but literals is a
// Go constant expression, evaluated at arbitrary precision and then required
// to fit its type, so a deep product of large constants is a compile error
// rather than the wrapping the same expression would do at run time.
func (g *gen) lit(k Kind) Expr {
	switch {
	case k == KBool:
		return &Lit{K: KBool, B: g.r.IntN(2) == 0}
	case k.float():
		vals := []float64{0.5, 1.5, 2, 3, 0.25, 7, 10, 0.125, 4, 1}
		v := pick(g.r, vals)
		if g.r.IntN(3) == 0 {
			v = -v
		}
		return &Lit{K: k, F: v}
	case k == KU32 || k == KU64:
		return &Lit{K: k, U: uint64(g.r.IntN(10))}
	}
	v := int64(g.r.IntN(10))
	if g.r.IntN(4) == 0 {
		v = -v
	}
	return &Lit{K: k, I: v}
}

// position is one of gpu.Ctx's built-ins. Under g.uni only the block's own
// coordinates and the grid's shape are offered, which is exactly the list
// diverge.go treats as uniform.
func (g *gen) position() Expr {
	uniform := []string{"BlockIdx", "BlockDim", "GridDim"}
	if g.p.Block.Y > 1 {
		uniform = append(uniform, "BlockDimY")
	}
	if g.uni {
		return &Pos{Method: pick(g.r, uniform)}
	}
	varying := []string{"ThreadIdx", "GlobalID", "GlobalIDX"}
	if g.p.Block.Y > 1 {
		varying = append(varying, "ThreadIdxY")
	}
	if g.warpOK {
		varying = append(varying, "LaneID")
	}
	if g.r.IntN(2) == 0 {
		return &Pos{Method: pick(g.r, uniform)}
	}
	return &Pos{Method: pick(g.r, varying), Vary: true}
}

// scalars are the visible variables of kind k that may be read here.
func (g *gen) scalars(k Kind) []*Var {
	return g.visible(func(v *Var) bool {
		return !v.buffer() && v.Kind == k && (!g.uni || !v.Vary)
	})
}

// buffers are the visible buffers of element kind k that may be indexed at an
// arbitrary position: the read-only parameters, the shared tiles once their
// fill barrier has passed, and the local arrays.
//
// An output parameter is deliberately absent. Every thread writes its own slot
// of one, so reading somebody else's would be a race, and a race is a
// difference between the two backends that says nothing about the emitter.
func (g *gen) buffers(k Kind) []*Var {
	var out []*Var
	for _, b := range g.readable() {
		// A local array is per-thread storage, so diverge.go marks the whole
		// variable varying -- it has nothing finer than a variable to say it
		// about -- and there is nothing uniform to read from one.
		if b.Kind == k && (!g.uni || b.Array == 0) {
			out = append(out, b)
		}
	}
	return out
}

// readable are the buffers whose elements may be read at an index other than
// this thread's own.
//
// An output parameter is deliberately absent, and so is the buffer the atomics
// use. Every thread writes its own slot of an output, so reading somebody
// else's is a race, and a race is a difference between the two backends that
// says nothing about the emitter. A local array is per-thread storage and a
// shared tile has been through its fill barrier, so both are safe.
func (g *gen) readable() []*Var {
	out := append([]*Var{}, g.inBufs...)
	out = append(out, g.tiles...)
	return append(out, g.visible(func(v *Var) bool { return v.Array > 0 })...)
}

// allBuffers is every buffer len() may be taken of, which includes the output
// parameters: a length is block-uniform however the buffer is used.
func (g *gen) allBuffers() []*Var {
	out := append([]*Var{}, g.inBufs...)
	out = append(out, g.outBufs...)
	out = append(out, g.tiles...)
	if g.atomBuf != nil {
		out = append(out, g.atomBuf)
	}
	return append(out, g.visible(func(v *Var) bool { return v.Array > 0 })...)
}

// indexSeed is the unwrapped part of an index: something worth indexing by,
// before it is folded into range.
func (g *gen) indexSeed() Expr {
	if g.r.IntN(3) == 0 {
		return g.expr(KInt, 1)
	}
	return g.leaf(KInt)
}

// wrapIndex folds e into [0, len(b)).
//
// Go's % and C's both take the sign of the dividend, so the second remainder
// is what makes a negative index positive in the same way on both backends.
// It is unconditional because an out-of-range index panics in the emulator and
// is undefined on the device, and neither outcome says anything about the
// emitter. Every buffer the generator makes has at least one element, so the
// divisor here is never zero.
func (g *gen) wrapIndex(e Expr, b *Var) Expr {
	n := func() Expr { return &Len{Base: b} }
	rem := &Binary{Op: token.REM, X: e, Y: n(), K: KInt}
	sum := &Binary{Op: token.ADD, X: rem, Y: n(), K: KInt}
	return &Binary{Op: token.REM, X: sum, Y: n(), K: KInt}
}

// safeDivisor forces a divisor into [1,128].
//
// Two things C leaves undefined and Go defines are closed by this one
// expression: division by zero, and the most negative value divided by -1,
// which Go says is itself.
func (g *gen) safeDivisor(k Kind, d int) Expr {
	masked := &Binary{Op: token.AND, X: g.expr(k, d), Y: g.mask(k, 127), K: k}
	return &Binary{Op: token.ADD, X: masked, Y: g.one(k), K: k}
}

// shiftCount masks a shift count to the width the kind has in the generated C.
//
// Go defines a shift wider than the type as zero and C leaves it undefined, so
// the mask is what makes the two agree. It is the C width rather than the Go
// one, which matters for int: 64 bits here and 32 there.
func (g *gen) shiftCount(k Kind, d int) Expr {
	return &Binary{Op: token.AND, X: g.expr(k, d), Y: g.mask(k, int64(k.cWidth()-1)), K: k}
}

func (g *gen) mask(k Kind, v int64) Expr {
	if k == KU32 || k == KU64 {
		return &Lit{K: k, U: uint64(v)}
	}
	return &Lit{K: k, I: v}
}

func (g *gen) one(k Kind) Expr {
	if k == KU32 || k == KU64 {
		return &Lit{K: k, U: 1}
	}
	return &Lit{K: k, I: 1}
}

// --- composites ------------------------------------------------------------

func (g *gen) intExpr(k Kind, d int) Expr {
	switch g.r.IntN(12) {
	case 0, 1:
		return g.leaf(k)
	case 2:
		return g.mixed(k, d)
	case 3:
		return g.unaryExpr(k, d)
	case 4:
		if e := g.conv(k, d); e != nil {
			return e
		}
	case 5:
		return &MinMax{Fn: pick(g.r, []string{"min", "max"}), K: k, Args: []Expr{g.expr(k, d-1), g.expr(k, d-1)}}
	case 6:
		if e := g.callExpr(k, d); e != nil {
			return e
		}
	case 7:
		if e := g.warpExpr(k, d); e != nil {
			return e
		}
	}
	return g.intBinary(k, d)
}

// intBinary picks one of the integer operators and builds operands that make
// it defined on both backends.
func (g *gen) intBinary(k Kind, d int) Expr {
	ops := []token.Token{token.ADD, token.SUB, token.MUL, token.QUO, token.REM,
		token.AND, token.OR, token.XOR, token.AND_NOT, token.SHR}
	// A Go int is never shifted left, and the subset refuses it for the reason
	// this generator found: int is 64 bits in Go and 32 on the device, so
	// 29 << 29 is 15569256448 on one and -1610612736 on the other. Masking the
	// count keeps the shift defined on both but does not make the answers
	// equal, and the narrowing's excuse -- that an int is an index, and an
	// index is bounded by the grid -- is exactly what a left shift breaks.
	//
	// `>>` stays, because it cannot grow a value: given an int that fits 32
	// bits the two agree, which is the premise holding rather than failing.
	// Every other integer kind keeps both, being the same width in both
	// languages.
	if k != KInt {
		ops = append(ops, token.SHL)
	}
	op := pick(g.r, ops)
	switch op {
	case token.QUO, token.REM:
		return &Binary{Op: op, X: g.expr(k, d-1), Y: g.safeDivisor(k, d-2), K: k}
	case token.SHL, token.SHR:
		return &Binary{Op: op, X: g.grounded(k, d-1), Y: g.shiftCount(k, d-2), K: k}
	case token.MUL:
		return &Binary{Op: op, X: g.grounded(k, d-1), Y: g.expr(k, d-1), K: k}
	case token.SUB:
		if k == KU32 || k == KU64 {
			return &Binary{Op: op, X: g.grounded(k, d-1), Y: g.expr(k, d-1), K: k}
		}
	}
	return &Binary{Op: op, X: g.expr(k, d-1), Y: g.expr(k, d-1), K: k}
}

// grounded is an expression that is certainly not a Go constant.
//
// Three operators need one on their left, and all three for the same reason: a
// subtree of nothing but literals is folded at arbitrary precision and then
// required to fit its type, so a shifted or multiplied constant is a compile
// error where the same expression at run time would simply wrap, and an
// unsigned constant subtraction that goes below zero is one too. Grounding the
// left operand is what makes the whole subtree a run-time expression.
func (g *gen) grounded(k Kind, d int) Expr {
	e := g.expr(k, d)
	if isConst(e) {
		return g.anchor(k)
	}
	return e
}

// mixed builds a chain that deliberately spans precedence levels.
//
// Go and C genuinely disagree here: Go binds << and >> tighter than + and -,
// C looser, and the bitwise operators go the other way against the
// comparisons. The emitter is correct only because it re-derives its
// parentheses from C's table, and a wrong table is invisible to NVRTC -- the
// generated code compiles and computes something else -- so a deep chain of
// operators from different levels is the pressure test.
func (g *gen) mixed(k Kind, d int) Expr {
	// Both shifts, except a left shift on a Go int; see intBinary for why. The
	// chain keeps its shift level either way, so the precedence this function
	// exists to press on -- a shift against the additive operators, which Go
	// and C order differently -- is still generated.
	shifts := []token.Token{token.SHR}
	if k != KInt {
		shifts = append(shifts, token.SHL)
	}
	levels := [][]token.Token{
		{token.MUL, token.AND},
		{token.ADD, token.SUB, token.OR, token.XOR},
		shifts,
	}
	// The chain starts from a value rather than a literal, which is what keeps
	// every subtree of it a run-time expression: a folded one would have to
	// fit its type at compile time, and a shifted or multiplied constant
	// quickly does not.
	e := g.anchor(k)
	for i := 0; i < 2+g.r.IntN(3); i++ {
		op := pick(g.r, pick(g.r, levels))
		switch op {
		case token.SHL, token.SHR:
			e = &Binary{Op: op, X: e, Y: g.shiftCount(k, 0), K: k}
		default:
			e = &Binary{Op: op, X: e, Y: g.leaf(k), K: k}
		}
		if g.r.IntN(3) == 0 {
			// Nesting on the right is what needs a parenthesis where nesting
			// on the left does not, since every one of these is left
			// associative in both languages.
			e = &Binary{Op: pick(g.r, levels[1]), X: g.anchor(k), Y: e, K: k}
		}
	}
	return e
}

func (g *gen) floatExpr(k Kind, d int) Expr {
	switch g.r.IntN(10) {
	case 0, 1:
		return g.leaf(k)
	case 2:
		return g.unaryExpr(k, d)
	case 3:
		if e := g.conv(k, d); e != nil {
			return e
		}
	case 4:
		// Go's builtin min and max, not gpu.Fmin: PLAN.md records that the two
		// backends disagree about a NaN operand and that nothing tests it.
		return &MinMax{Fn: pick(g.r, []string{"min", "max"}), K: k, Args: []Expr{g.grounded(k, d-1), g.expr(k, d-1)}}
	case 5:
		return g.mathCall(k, d)
	case 6:
		if e := g.callExpr(k, d); e != nil {
			return e
		}
	case 7:
		if e := g.warpExpr(k, d); e != nil {
			return e
		}
	}
	op := pick(g.r, []token.Token{token.ADD, token.SUB, token.MUL, token.QUO})
	// The left operand is grounded, which makes the whole expression a
	// run-time one. Go evaluates a floating-point constant expression exactly
	// and rounds once at the end, and exact arithmetic has neither a negative
	// zero nor an intermediate rounding: `(-3.0) * float32(0)` is a positive
	// zero where the compiler folds it and a negative one where the closure
	// multiplies it, and a chain of constant additions rounds a different
	// number of times on each side.
	if op == token.QUO {
		// The divisor is a value too: a constant divided by a constant zero is
		// a compile error. Dividing by a zero that arrives at run time is an
		// infinity on both backends and is left alone.
		return &Binary{Op: op, X: g.grounded(k, d-1), Y: g.anchor(k), K: k}
	}
	return &Binary{Op: op, X: g.grounded(k, d-1), Y: g.expr(k, d-1), K: k}
}

func (g *gen) mathCall(k Kind, d int) Expr {
	one := []string{"Sqrt", "Abs", "Sin", "Cos", "Exp", "Log"}
	two := []string{"Hypot", "Fmin", "Fmax"}
	suffix := ""
	if k == KF64 {
		suffix = "64"
	}
	if g.r.IntN(2) == 0 {
		return &MathCall{Fn: pick(g.r, one) + suffix, K: k, Args: []Expr{g.expr(k, d-1)}}
	}
	return &MathCall{Fn: pick(g.r, two) + suffix, K: k, Args: []Expr{g.expr(k, d-1), g.expr(k, d-1)}}
}

func (g *gen) boolExpr(d int) Expr {
	switch g.r.IntN(9) {
	case 0:
		return g.leaf(KBool)
	case 1:
		return &Unary{Op: token.NOT, X: g.expr(KBool, d-1)}
	case 2, 3:
		op := token.LAND
		if g.r.IntN(2) == 0 {
			op = token.LOR
		}
		// The right operand is generated without the masked warp primitives.
		// && and || short-circuit in C as they do in Go, so a __any_sync on
		// the right of one is called by whichever lanes got that far with a
		// mask that names all of them -- undefined on the device, and a
		// deadlock on the emulator. SPEC.md §6 states that the divergence
		// rules do not see into a condition and so do not catch this; the
		// generator stays out of it rather than reporting it as a defect.
		x := g.expr(KBool, d-1)
		prev := g.noWarp
		g.noWarp = true
		y := g.expr(KBool, d-1)
		g.noWarp = prev
		return &Binary{Op: op, X: x, Y: y, K: KBool}
	case 4:
		if e := g.callExpr(KBool, d); e != nil {
			return e
		}
	case 5:
		if e := g.warpExpr(KBool, d); e != nil {
			return e
		}
	}
	// A comparison, which is where a mixed-precedence operand ends up next to
	// an operator C binds the other way round: Go reads a & b == c as
	// (a & b) == c and C reads it as a & (b == c).
	k := pick(g.r, g.valueKinds())
	if k == KBool {
		op := token.EQL
		if g.r.IntN(2) == 0 {
			op = token.NEQ
		}
		return &Binary{Op: op, X: g.expr(KBool, d-1), Y: g.expr(KBool, d-1), K: KBool}
	}
	op := pick(g.r, []token.Token{token.EQL, token.NEQ, token.LSS, token.LEQ, token.GTR, token.GEQ})
	return &Binary{Op: op, X: g.expr(k, d-1), Y: g.expr(k, d-1), K: KBool}
}

// unaryExpr applies a prefix operator, and only to something that is not a
// constant: -x on an unsigned constant and ^x on one are both values outside
// the type, which Go rejects at compile time rather than wrapping.
func (g *gen) unaryExpr(k Kind, d int) Expr {
	x := g.expr(k, d-1)
	if isConst(x) {
		x = g.anchor(k)
	}
	if k.integer() && !k.signed() {
		return &Unary{Op: token.XOR, X: x}
	}
	if k.integer() && g.r.IntN(3) == 0 {
		return &Unary{Op: token.XOR, X: x}
	}
	return &Unary{Op: token.SUB, X: x}
}

// conv builds a conversion into k, or nil when no legal source is to hand.
func (g *gen) conv(k Kind, d int) Expr {
	if k == KBool {
		return nil // Go has no conversion between bool and anything else
	}
	// A narrow element is storage only: a conversion is the one thing that
	// accepts it, which is how a []uint8 is read at all.
	if g.r.IntN(2) == 0 {
		var narrow []*Var
		for _, b := range g.inBufs {
			if b.Kind.narrow() {
				narrow = append(narrow, b)
			}
		}
		if len(narrow) > 0 {
			b := pick(g.r, narrow)
			return &Conv{To: k, X: &Index{Base: b, Idx: g.wrapIndex(g.indexSeed(), b)}}
		}
	}
	from := pick(g.r, g.valueKinds())
	switch {
	case from == KBool:
		return nil
	case k == KInt && (from == KI64 || from == KU64):
		// int(x) from a 64-bit value truncates on the device and not on the
		// host, which the subset refuses rather than lowering.
		return nil
	case from.float() && !k.float():
		return &Conv{To: k, X: g.clampToInt(from, k, d)}
	case k == KU32 || k == KU64:
		// uint32(-6) is a constant outside the type, which Go rejects where
		// the same conversion of a value would simply wrap.
		return &Conv{To: k, X: g.grounded(from, d-1)}
	}
	return &Conv{To: k, X: g.expr(from, d-1)}
}

// clampToInt bounds a float so that converting it to the integer kind `to` is
// defined.
//
// Go leaves a conversion whose value does not fit implementation-dependent and
// C leaves it undefined, and the input distribution deliberately contains NaN
// and both infinities. gpu.Fmin and gpu.Fmax are the clamp because they follow
// IEEE minNum/maxNum: a NaN operand is ignored and the number is returned, so
// even a NaN comes out of this at a thousand.
//
// The floor depends on where the value is going, which is what this originally
// got wrong: clamping to -1000 and then converting to a uint32 is a negative
// float reaching an unsigned integer, which is exactly the case neither
// language defines. The host differential found it -- the two backends
// disagreed on every element of a buffer -- and it was the generator
// manufacturing a program whose answer nothing promises, not the emitter
// mistranslating one.
func (g *gen) clampToInt(from, to Kind, d int) Expr {
	suffix := ""
	if from == KF64 {
		suffix = "64"
	}
	floor := -1000.0
	if !to.signed() {
		floor = 0
	}
	lo := &MathCall{Fn: "Fmin" + suffix, K: from, Args: []Expr{g.expr(from, d-1), &Lit{K: from, F: 1000}}}
	return &MathCall{Fn: "Fmax" + suffix, K: from, Args: []Expr{lo, &Lit{K: from, F: floor}}}
}

// callExpr calls a helper for its value, or returns nil when none has that
// result kind or a call is not allowed here.
func (g *gen) callExpr(k Kind, d int) Expr {
	if g.pure || g.uni {
		// Under g.uni the answer has to be block-uniform, and a call is never
		// taken to be: see CallExpr.varies.
		return nil
	}
	var fns []*Func
	for _, fn := range g.pureFns {
		if fn.Result == k {
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
		args[i] = g.expr(p.Kind, d-1)
	}
	return &CallExpr{Fn: fn, Args: args}
}

// bufferArg is a buffer to hand a helper. Only read-only ones are offered: a
// helper that writes through two parameters bound to one buffer is refused at
// lowering, and the launch may have bound two of these to one buffer on
// purpose.
func (g *gen) bufferArg(k Kind) *Var {
	var out []*Var
	for _, b := range g.inBufs {
		if b.Kind == k {
			out = append(out, b)
		}
	}
	if len(out) == 0 {
		return nil
	}
	return pick(g.r, out)
}

// warpExpr builds a warp-level primitive, or nil where one would be undefined
// or refused.
//
// Three conditions have to hold at once. The block must be a whole number of
// warps, because the emitter writes 0xffffffff into every _sync built-in and
// in a short warp that mask names lanes which do not exist. The statement must
// be on the block's common path, because that mask is a promise every lane is
// here. And the expression must not have been asked to be block-uniform,
// because every one of these answers a question about this lane.
func (g *gen) warpExpr(k Kind, d int) Expr {
	if !g.sync || !g.warpOK || !g.uniform || g.uni || g.pure || g.noWarp || !g.fn.Ctx {
		return nil
	}
	lane := func() Expr {
		return &Binary{Op: token.AND, X: g.expr(KInt, 0), Y: &Lit{K: KInt, I: 31}, K: KInt}
	}
	switch k {
	case KF32:
		m := pick(g.r, []string{"ShuffleF32", "ShuffleXorF32", "ShuffleUpF32", "ShuffleDownF32"})
		return &WarpCall{Method: m, K: KF32, Mask: true, Args: []Expr{g.expr(KF32, d-1), lane()}}
	case KI32:
		m := pick(g.r, []string{"ShuffleI32", "ShuffleXorI32", "ShuffleUpI32", "ShuffleDownI32"})
		return &WarpCall{Method: m, K: KI32, Mask: true, Args: []Expr{g.expr(KI32, d-1), lane()}}
	case KU32:
		if g.r.IntN(2) == 0 {
			return &WarpCall{Method: "ActiveMask", K: KU32}
		}
		return &WarpCall{Method: "Ballot", K: KU32, Mask: true, Args: []Expr{g.expr(KBool, d-1)}}
	case KBool:
		m := "Any"
		if g.r.IntN(2) == 0 {
			m = "All"
		}
		return &WarpCall{Method: m, K: KBool, Mask: true, Args: []Expr{g.expr(KBool, d-1)}}
	case KInt:
		return &WarpCall{Method: "LaneID", K: KInt}
	}
	return nil
}

// anchor is an expression of kind k that is certainly not a constant.
//
// Several rules need one. A multiplication, a shift and an unsigned
// subtraction each have to have a run-time left operand, because the folded
// form would be a constant required to fit its type rather than a value
// allowed to wrap. A short declaration needs one for a different reason: an
// untyped constant would give the variable its default type instead of the one
// meant.
func (g *gen) anchor(k Kind) Expr {
	if vars := g.scalars(k); len(vars) > 0 {
		return &Ref{V: pick(g.r, vars)}
	}
	if bufs := g.buffers(k); len(bufs) > 0 {
		b := pick(g.r, bufs)
		return &Index{Base: b, Idx: g.wrapIndex(&Lit{K: KInt}, b)}
	}
	base := g.anchorInt()
	switch k {
	case KInt:
		return base
	case KBool:
		return &Binary{Op: token.LSS, X: base, Y: &Lit{K: KInt, I: 3}, K: KBool}
	}
	return &Conv{To: k, X: base}
}

// anchorInt is a non-constant int, which every other anchor is built from.
//
// There is always one. A kernel has the thread's own number; a helper's first
// parameter is an int for this reason, and neither can be shadowed. Under
// g.uni the answer has to be block-uniform too, which is why a helper without
// a gpu.Ctx never asks: its parameters are assumed to vary, so it has no
// uniform value to offer beyond a length.
func (g *gen) anchorInt() Expr {
	if vars := g.scalars(KInt); len(vars) > 0 {
		return &Ref{V: pick(g.r, vars)}
	}
	if g.fn.Ctx {
		if g.uni {
			return &Pos{Method: "BlockDim"}
		}
		return &Pos{Method: "ThreadIdx", Vary: true}
	}
	if bufs := g.allBuffers(); len(bufs) > 0 {
		return &Len{Base: pick(g.r, bufs)}
	}
	// Unreachable: every helper is given an int parameter, and it cannot be
	// shadowed. Spelled as a length rather than a literal anyway, so that a
	// future change here cannot reintroduce a constant that folds.
	return &Lit{K: KInt, I: 1}
}

// isConst reports whether e is a Go constant expression, which decides both
// whether it may overflow at compile time and what type a short declaration
// would give it.
func isConst(e Expr) bool {
	switch e := e.(type) {
	case *Lit:
		return true
	case *Len:
		// len of an array is a constant in Go and len of a slice is not, which
		// is the difference between uint32(-len(a)) being a compile error and
		// being a value that wraps.
		return e.Base.Array > 0
	case *Unary:
		return isConst(e.X)
	case *Binary:
		return isConst(e.X) && isConst(e.Y)
	case *Conv:
		return isConst(e.X)
	case *MinMax:
		return isConst(e.Args[0]) && isConst(e.Args[1])
	}
	return false
}

// untypedConst reports whether e is a constant with no type of its own, which
// is the case a short declaration cannot be given: `x := 1 + 2` declares an
// int whatever the generator meant it to be. A conversion anywhere inside
// gives the constant a type, and then the declaration takes that.
func untypedConst(e Expr) bool {
	switch e := e.(type) {
	case *Lit:
		return true
	case *Unary:
		return untypedConst(e.X)
	case *Binary:
		return untypedConst(e.X) && untypedConst(e.Y)
	case *MinMax:
		return untypedConst(e.Args[0]) && untypedConst(e.Args[1])
	}
	return false
}
