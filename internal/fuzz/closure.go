package fuzz

import (
	"fmt"
	"go/token"

	"github.com/CWBudde/gocuda/gpu"
)

// This file is the second rendering of the IR: a func(gpu.Ctx) that gpu.RunCPU
// executes.
//
// It is not an interpreter of Go. Every node compiles once, ahead of the
// launch, into a Go closure that performs that node's operation on that node's
// type -- an int32 addition in the IR becomes a Go int32 addition, a float32
// division becomes a Go float32 division -- so the oracle's arithmetic is the
// language's rather than a second definition of it that somebody would have to
// check. What the compilation does decide is control flow and evaluation
// order, and those are the places to look when the two renderings disagree.
//
// Compiling ahead of the launch also means every type assertion happens once,
// on a closure, and none happens per thread.

// Closure renders the program as the function gpu.RunCPU takes.
//
// a supplies the kernel's arguments and is read once here, not per thread, so
// the returned closure can be handed to a launch of any size.
func (p *Program) Closure(a *Args) func(gpu.Ctx) {
	c := &compiler{bodies: map[*Func]func(*frame) ctrl{}}
	body := c.body(p.Main)
	binders := make([]func(*frame), len(p.Main.Params))
	for i, v := range p.Main.Params {
		binders[i] = bindArg(v, a.Vals[i])
	}
	layout := p.Main.frame
	return func(ctx gpu.Ctx) {
		f := newFrame(layout, ctx)
		for _, bind := range binders {
			bind(f)
		}
		body(f)
	}
}

// compiler memoises each function's compiled body. Recursion is refused by the
// subset, so a plain recursive walk terminates.
type compiler struct {
	bodies map[*Func]func(*frame) ctrl
	fn     *Func
}

func (c *compiler) body(fn *Func) func(*frame) ctrl {
	if b, ok := c.bodies[fn]; ok {
		return b
	}
	prev := c.fn
	c.fn = fn
	b := c.stmts(fn.Body)
	c.fn = prev
	c.bodies[fn] = b
	return b
}

// frame is one call's storage: the thread's position, one array per kind for
// scalars and one per kind for buffers, and the slot a return writes into.
//
// The arrays are typed rather than []any because a boxed write would allocate
// on every assignment in a loop. Which array a variable lives in and at which
// index is fixed when the program is generated (frameLayout), so nothing here
// searches for anything.
type frame struct {
	ctx gpu.Ctx

	sF32 []float32
	sF64 []float64
	sI32 []int32
	sI64 []int64
	sU32 []uint32
	sU64 []uint64
	sB   []bool
	sInt []int

	bF32 [][]float32
	bF64 [][]float64
	bI32 [][]int32
	bI64 [][]int64
	bU32 [][]uint32
	bU64 [][]uint64
	bB   [][]bool
	bI8  [][]int8
	bI16 [][]int16
	bU8  [][]uint8
	bU16 [][]uint16
	// bStruct holds the struct buffers as any, because there is no one Go type
	// to declare them as -- the catalogue has four. Which shape each slot holds
	// is fixed when the program is generated, so the type assertion that gets
	// at it is made once, while the closure is compiled, and never in the loop.
	bStruct []any

	rF32 float32
	rF64 float64
	rI32 int32
	rI64 int64
	rU32 uint32
	rU64 uint64
	rB   bool
	rInt int
}

// newFrame allocates exactly the slots the function's body uses. A kind the
// body never mentions costs nothing.
func newFrame(l frameLayout, ctx gpu.Ctx) *frame {
	f := &frame{ctx: ctx}
	if n := l.scalars[KF32]; n > 0 {
		f.sF32 = make([]float32, n)
	}
	if n := l.scalars[KF64]; n > 0 {
		f.sF64 = make([]float64, n)
	}
	if n := l.scalars[KI32]; n > 0 {
		f.sI32 = make([]int32, n)
	}
	if n := l.scalars[KI64]; n > 0 {
		f.sI64 = make([]int64, n)
	}
	if n := l.scalars[KU32]; n > 0 {
		f.sU32 = make([]uint32, n)
	}
	if n := l.scalars[KU64]; n > 0 {
		f.sU64 = make([]uint64, n)
	}
	if n := l.scalars[KBool]; n > 0 {
		f.sB = make([]bool, n)
	}
	if n := l.scalars[KInt]; n > 0 {
		f.sInt = make([]int, n)
	}
	if n := l.buffers[KF32]; n > 0 {
		f.bF32 = make([][]float32, n)
	}
	if n := l.buffers[KF64]; n > 0 {
		f.bF64 = make([][]float64, n)
	}
	if n := l.buffers[KI32]; n > 0 {
		f.bI32 = make([][]int32, n)
	}
	if n := l.buffers[KI64]; n > 0 {
		f.bI64 = make([][]int64, n)
	}
	if n := l.buffers[KU32]; n > 0 {
		f.bU32 = make([][]uint32, n)
	}
	if n := l.buffers[KU64]; n > 0 {
		f.bU64 = make([][]uint64, n)
	}
	if n := l.buffers[KBool]; n > 0 {
		f.bB = make([][]bool, n)
	}
	if n := l.buffers[KI8]; n > 0 {
		f.bI8 = make([][]int8, n)
	}
	if n := l.buffers[KI16]; n > 0 {
		f.bI16 = make([][]int16, n)
	}
	if n := l.buffers[KU8]; n > 0 {
		f.bU8 = make([][]uint8, n)
	}
	if n := l.buffers[KU16]; n > 0 {
		f.bU16 = make([][]uint16, n)
	}
	if n := l.buffers[KStruct]; n > 0 {
		f.bStruct = make([]any, n)
	}
	return f
}

// code is one compiled expression: fn is a func(*frame) T with T the Go type k
// stands for. The assertion that recovers T happens once, while compiling the
// node above this one.
type code struct {
	k  Kind
	fn any
}

func mk[T any](k Kind, f func(*frame) T) code { return code{k: k, fn: f} }

// as recovers the typed closure. A mismatch here is a generator bug rather
// than anything a fuzz target could produce, so it panics rather than
// reporting.
func as[T any](c code) func(*frame) T {
	f, ok := c.fn.(func(*frame) T)
	if !ok {
		panic(fmt.Sprintf("fuzz: compiled %s where %T was wanted", c.k.goName(), *new(func(*frame) T)))
	}
	return f
}

// ctrlKind is how a statement finished.
type ctrlKind uint8

const (
	ctrlNone ctrlKind = iota
	ctrlBreak
	ctrlContinue
	ctrlReturn
)

// ctrl carries a break or continue out to the loop it names. A label travels
// with it because a labelled branch may leave more than one loop, which is
// exactly what the emitter turns into a goto.
type ctrl struct {
	k     ctrlKind
	label string
}

// --- expressions -----------------------------------------------------------

func (c *compiler) expr(e Expr) code {
	switch e := e.(type) {
	case *Lit:
		return c.lit(e)
	case *Ref:
		return c.ref(e.V)
	case *Index:
		return c.index(e)
	case *Field:
		return c.field(e)
	case *Len:
		return c.length(e.Base)
	case *Binary:
		return c.binary(e)
	case *Unary:
		return c.unary(e)
	case *Conv:
		return c.conv(e)
	case *Pos:
		return mk(KInt, posFn(e.Method))
	case *MathCall:
		return c.math(e)
	case *MinMax:
		return c.minmax(e)
	case *WarpCall:
		return c.warp(e)
	case *AtomicCall:
		return c.atomic(e)
	case *CallExpr:
		return c.call(e.Fn, e.Args)
	}
	panic(fmt.Sprintf("fuzz: no closure for %T", e))
}

func (c *compiler) lit(l *Lit) code {
	switch l.K {
	case KF32:
		v := float32(l.F)
		return mk(KF32, func(*frame) float32 { return v })
	case KF64:
		v := l.F
		return mk(KF64, func(*frame) float64 { return v })
	case KI32:
		v := int32(l.I)
		return mk(KI32, func(*frame) int32 { return v })
	case KI64:
		v := l.I
		return mk(KI64, func(*frame) int64 { return v })
	case KU32:
		v := uint32(l.U)
		return mk(KU32, func(*frame) uint32 { return v })
	case KU64:
		v := l.U
		return mk(KU64, func(*frame) uint64 { return v })
	case KBool:
		v := l.B
		return mk(KBool, func(*frame) bool { return v })
	case KInt:
		v := int(l.I)
		return mk(KInt, func(*frame) int { return v })
	}
	panic("fuzz: no literal of kind " + l.K.goName())
}

// ref reads a scalar. scalarsOf is the one place a kind is turned into a
// frame field, and every other scalar operation goes through it.
func (c *compiler) ref(v *Var) code {
	n := v.slot
	switch v.Kind {
	case KF32:
		return mk(KF32, func(f *frame) float32 { return f.sF32[n] })
	case KF64:
		return mk(KF64, func(f *frame) float64 { return f.sF64[n] })
	case KI32:
		return mk(KI32, func(f *frame) int32 { return f.sI32[n] })
	case KI64:
		return mk(KI64, func(f *frame) int64 { return f.sI64[n] })
	case KU32:
		return mk(KU32, func(f *frame) uint32 { return f.sU32[n] })
	case KU64:
		return mk(KU64, func(f *frame) uint64 { return f.sU64[n] })
	case KBool:
		return mk(KBool, func(f *frame) bool { return f.sB[n] })
	case KInt:
		return mk(KInt, func(f *frame) int { return f.sInt[n] })
	}
	panic("fuzz: no scalar of kind " + v.Kind.goName())
}

// bufOf returns the closure that finds a variable's buffer in a frame. The
// narrow kinds appear only here and in a conversion, which is exactly the
// storage-only rule SPEC.md §3 states.
func bufOf(v *Var) any {
	n := v.slot
	switch v.Kind {
	case KF32:
		return func(f *frame) []float32 { return f.bF32[n] }
	case KF64:
		return func(f *frame) []float64 { return f.bF64[n] }
	case KI32:
		return func(f *frame) []int32 { return f.bI32[n] }
	case KI64:
		return func(f *frame) []int64 { return f.bI64[n] }
	case KU32:
		return func(f *frame) []uint32 { return f.bU32[n] }
	case KU64:
		return func(f *frame) []uint64 { return f.bU64[n] }
	case KBool:
		return func(f *frame) []bool { return f.bB[n] }
	case KI8:
		return func(f *frame) []int8 { return f.bI8[n] }
	case KI16:
		return func(f *frame) []int16 { return f.bI16[n] }
	case KU8:
		return func(f *frame) []uint8 { return f.bU8[n] }
	case KU16:
		return func(f *frame) []uint16 { return f.bU16[n] }
	}
	panic("fuzz: no buffer of kind " + v.Kind.goName())
}

// field compiles p[i].A.
//
// The kind is the field's, so what comes back is an ordinary typed code of
// that kind and every compiler above this one is unchanged. The catalogue's
// accessor is asserted to its type here, once, rather than in the closure: a
// type assertion in the loop would be a boxed read on every iteration, which
// is exactly what the typed frame exists to avoid.
func (c *compiler) field(x *Field) code {
	idx := as[int](c.expr(x.Idx))
	n := x.Base.slot
	f := x.Base.Shape.Fields[x.F]
	switch f.Kind {
	case KF32:
		g := f.get.(func(any, int) float32)
		return mk(KF32, func(fr *frame) float32 { return g(fr.bStruct[n], idx(fr)) })
	case KF64:
		g := f.get.(func(any, int) float64)
		return mk(KF64, func(fr *frame) float64 { return g(fr.bStruct[n], idx(fr)) })
	case KI32:
		g := f.get.(func(any, int) int32)
		return mk(KI32, func(fr *frame) int32 { return g(fr.bStruct[n], idx(fr)) })
	case KI64:
		g := f.get.(func(any, int) int64)
		return mk(KI64, func(fr *frame) int64 { return g(fr.bStruct[n], idx(fr)) })
	case KU32:
		g := f.get.(func(any, int) uint32)
		return mk(KU32, func(fr *frame) uint32 { return g(fr.bStruct[n], idx(fr)) })
	case KU64:
		g := f.get.(func(any, int) uint64)
		return mk(KU64, func(fr *frame) uint64 { return g(fr.bStruct[n], idx(fr)) })
	case KBool:
		g := f.get.(func(any, int) bool)
		return mk(KBool, func(fr *frame) bool { return g(fr.bStruct[n], idx(fr)) })
	case KI8:
		g := f.get.(func(any, int) int8)
		return mk(KI8, func(fr *frame) int8 { return g(fr.bStruct[n], idx(fr)) })
	case KI16:
		g := f.get.(func(any, int) int16)
		return mk(KI16, func(fr *frame) int16 { return g(fr.bStruct[n], idx(fr)) })
	case KU8:
		g := f.get.(func(any, int) uint8)
		return mk(KU8, func(fr *frame) uint8 { return g(fr.bStruct[n], idx(fr)) })
	case KU16:
		g := f.get.(func(any, int) uint16)
		return mk(KU16, func(fr *frame) uint16 { return g(fr.bStruct[n], idx(fr)) })
	}
	panic("fuzz: no struct field of kind " + f.Kind.goName())
}

// fieldStore compiles `p[i].A = v` for a value of the field's kind.
func (c *compiler) fieldStore(l Lvalue, val code) func(*frame) {
	idx := as[int](c.expr(l.Idx))
	n := l.V.slot
	f := l.V.Shape.Fields[l.F]
	switch f.Kind {
	case KF32:
		st, v := f.set.(func(any, int, float32)), as[float32](val)
		return func(fr *frame) { st(fr.bStruct[n], idx(fr), v(fr)) }
	case KF64:
		st, v := f.set.(func(any, int, float64)), as[float64](val)
		return func(fr *frame) { st(fr.bStruct[n], idx(fr), v(fr)) }
	case KI32:
		st, v := f.set.(func(any, int, int32)), as[int32](val)
		return func(fr *frame) { st(fr.bStruct[n], idx(fr), v(fr)) }
	case KI64:
		st, v := f.set.(func(any, int, int64)), as[int64](val)
		return func(fr *frame) { st(fr.bStruct[n], idx(fr), v(fr)) }
	case KU32:
		st, v := f.set.(func(any, int, uint32)), as[uint32](val)
		return func(fr *frame) { st(fr.bStruct[n], idx(fr), v(fr)) }
	case KU64:
		st, v := f.set.(func(any, int, uint64)), as[uint64](val)
		return func(fr *frame) { st(fr.bStruct[n], idx(fr), v(fr)) }
	case KBool:
		st, v := f.set.(func(any, int, bool)), as[bool](val)
		return func(fr *frame) { st(fr.bStruct[n], idx(fr), v(fr)) }
	}
	panic("fuzz: cannot store into a struct field of kind " + f.Kind.goName())
}

func (c *compiler) index(x *Index) code {
	idx := as[int](c.expr(x.Idx))
	return withBuf(x.Base, func(get func(*frame) []float32) code {
		return mk(KF32, func(f *frame) float32 { return get(f)[idx(f)] })
	}, func(get func(*frame) []float64) code {
		return mk(KF64, func(f *frame) float64 { return get(f)[idx(f)] })
	}, func(get func(*frame) []int32) code {
		return mk(KI32, func(f *frame) int32 { return get(f)[idx(f)] })
	}, func(get func(*frame) []int64) code {
		return mk(KI64, func(f *frame) int64 { return get(f)[idx(f)] })
	}, func(get func(*frame) []uint32) code {
		return mk(KU32, func(f *frame) uint32 { return get(f)[idx(f)] })
	}, func(get func(*frame) []uint64) code {
		return mk(KU64, func(f *frame) uint64 { return get(f)[idx(f)] })
	}, func(get func(*frame) []bool) code {
		return mk(KBool, func(f *frame) bool { return get(f)[idx(f)] })
	}, func(get func(*frame) []int8) code {
		return mk(KI8, func(f *frame) int8 { return get(f)[idx(f)] })
	}, func(get func(*frame) []int16) code {
		return mk(KI16, func(f *frame) int16 { return get(f)[idx(f)] })
	}, func(get func(*frame) []uint8) code {
		return mk(KU8, func(f *frame) uint8 { return get(f)[idx(f)] })
	}, func(get func(*frame) []uint16) code {
		return mk(KU16, func(f *frame) uint16 { return get(f)[idx(f)] })
	})
}

// withBuf dispatches once on a buffer's element kind and hands the caller the
// typed accessor. It exists so that reading an element, writing one and
// measuring a buffer are each written once per operation rather than once per
// operation per kind.
func withBuf(v *Var,
	f32 func(func(*frame) []float32) code,
	f64 func(func(*frame) []float64) code,
	i32 func(func(*frame) []int32) code,
	i64 func(func(*frame) []int64) code,
	u32 func(func(*frame) []uint32) code,
	u64 func(func(*frame) []uint64) code,
	b func(func(*frame) []bool) code,
	i8 func(func(*frame) []int8) code,
	i16 func(func(*frame) []int16) code,
	u8 func(func(*frame) []uint8) code,
	u16 func(func(*frame) []uint16) code,
) code {
	g := bufOf(v)
	switch v.Kind {
	case KF32:
		return f32(g.(func(*frame) []float32))
	case KF64:
		return f64(g.(func(*frame) []float64))
	case KI32:
		return i32(g.(func(*frame) []int32))
	case KI64:
		return i64(g.(func(*frame) []int64))
	case KU32:
		return u32(g.(func(*frame) []uint32))
	case KU64:
		return u64(g.(func(*frame) []uint64))
	case KBool:
		return b(g.(func(*frame) []bool))
	case KI8:
		return i8(g.(func(*frame) []int8))
	case KI16:
		return i16(g.(func(*frame) []int16))
	case KU8:
		return u8(g.(func(*frame) []uint8))
	case KU16:
		return u16(g.(func(*frame) []uint16))
	}
	panic("fuzz: no buffer of kind " + v.Kind.goName())
}

func (c *compiler) length(v *Var) code {
	if v.Shape != nil {
		n, shape := v.slot, v.Shape
		return mk(KInt, func(f *frame) int { return shape.length(f.bStruct[n]) })
	}
	return withBuf(v, func(get func(*frame) []float32) code {
		return mk(KInt, func(f *frame) int { return len(get(f)) })
	}, func(get func(*frame) []float64) code {
		return mk(KInt, func(f *frame) int { return len(get(f)) })
	}, func(get func(*frame) []int32) code {
		return mk(KInt, func(f *frame) int { return len(get(f)) })
	}, func(get func(*frame) []int64) code {
		return mk(KInt, func(f *frame) int { return len(get(f)) })
	}, func(get func(*frame) []uint32) code {
		return mk(KInt, func(f *frame) int { return len(get(f)) })
	}, func(get func(*frame) []uint64) code {
		return mk(KInt, func(f *frame) int { return len(get(f)) })
	}, func(get func(*frame) []bool) code {
		return mk(KInt, func(f *frame) int { return len(get(f)) })
	}, func(get func(*frame) []int8) code {
		return mk(KInt, func(f *frame) int { return len(get(f)) })
	}, func(get func(*frame) []int16) code {
		return mk(KInt, func(f *frame) int { return len(get(f)) })
	}, func(get func(*frame) []uint8) code {
		return mk(KInt, func(f *frame) int { return len(get(f)) })
	}, func(get func(*frame) []uint16) code {
		return mk(KInt, func(f *frame) int { return len(get(f)) })
	})
}

// numeric is every kind an arithmetic operator or a conversion may produce.
type numeric interface {
	~float32 | ~float64 | ~int32 | ~int64 | ~uint32 | ~uint64 | ~int
}

// integral is numeric without the floats, which is where %, the shifts and the
// bitwise operators live.
type integral interface {
	~int32 | ~int64 | ~uint32 | ~uint64 | ~int
}

func (c *compiler) binary(e *Binary) code {
	switch e.Op {
	case token.LAND:
		x, y := as[bool](c.expr(e.X)), as[bool](c.expr(e.Y))
		// Short-circuit, as Go and C both do. A generated program can put a
		// warp primitive on the right of one of these, which SPEC.md §6 names
		// as a limit of the divergence rules rather than something they catch.
		return mk(KBool, func(f *frame) bool { return x(f) && y(f) })
	case token.LOR:
		x, y := as[bool](c.expr(e.X)), as[bool](c.expr(e.Y))
		return mk(KBool, func(f *frame) bool { return x(f) || y(f) })
	}
	x, y := c.expr(e.X), c.expr(e.Y)
	if e.K == KBool && x.k != KBool {
		return c.compare(e.Op, x, y)
	}
	switch e.K {
	case KBool: // == and != on bools
		return c.compare(e.Op, x, y)
	case KF32:
		return mk(KF32, arith(e.Op, as[float32](x), as[float32](y)))
	case KF64:
		return mk(KF64, arith(e.Op, as[float64](x), as[float64](y)))
	case KI32:
		return mk(KI32, arithInt(e.Op, as[int32](x), as[int32](y)))
	case KI64:
		return mk(KI64, arithInt(e.Op, as[int64](x), as[int64](y)))
	case KU32:
		return mk(KU32, arithInt(e.Op, as[uint32](x), as[uint32](y)))
	case KU64:
		return mk(KU64, arithInt(e.Op, as[uint64](x), as[uint64](y)))
	case KInt:
		return mk(KInt, arithInt(e.Op, as[int](x), as[int](y)))
	}
	panic("fuzz: no binary operator producing " + e.K.goName())
}

// opFn is the operator set both floats and integers share, as a function of
// two values. It is spelled at the value level rather than over closures
// because a compound assignment needs the same operator applied to a slot it
// has already read.
func opFn[T numeric](op token.Token) func(T, T) T {
	switch op {
	case token.ADD:
		return func(a, b T) T { return a + b }
	case token.SUB:
		return func(a, b T) T { return a - b }
	case token.MUL:
		return func(a, b T) T { return a * b }
	case token.QUO:
		return func(a, b T) T { return a / b }
	}
	panic("fuzz: " + op.String() + " is not an arithmetic operator")
}

// opFnInt adds the operators only an integer has. The divisor of a / and a %
// has already been forced into [1,128] by the generator, and a shift count
// masked to the C width, so nothing here can reach the cases where Go defines
// an answer and C does not.
func opFnInt[T integral](op token.Token) func(T, T) T {
	switch op {
	case token.REM:
		return func(a, b T) T { return a % b }
	case token.AND:
		return func(a, b T) T { return a & b }
	case token.OR:
		return func(a, b T) T { return a | b }
	case token.XOR:
		return func(a, b T) T { return a ^ b }
	case token.AND_NOT:
		return func(a, b T) T { return a &^ b }
	case token.SHL:
		return func(a, b T) T { return a << uint(b) }
	case token.SHR:
		return func(a, b T) T { return a >> uint(b) }
	}
	return opFn[T](op)
}

// arith and arithInt lift an operator over two compiled operands.
func arith[T numeric](op token.Token, x, y func(*frame) T) func(*frame) T {
	g := opFn[T](op)
	return func(f *frame) T { return g(x(f), y(f)) }
}

func arithInt[T integral](op token.Token, x, y func(*frame) T) func(*frame) T {
	g := opFnInt[T](op)
	return func(f *frame) T { return g(x(f), y(f)) }
}

func (c *compiler) compare(op token.Token, x, y code) code {
	switch x.k {
	case KF32:
		return mk(KBool, cmp(op, as[float32](x), as[float32](y)))
	case KF64:
		return mk(KBool, cmp(op, as[float64](x), as[float64](y)))
	case KI32:
		return mk(KBool, cmp(op, as[int32](x), as[int32](y)))
	case KI64:
		return mk(KBool, cmp(op, as[int64](x), as[int64](y)))
	case KU32:
		return mk(KBool, cmp(op, as[uint32](x), as[uint32](y)))
	case KU64:
		return mk(KBool, cmp(op, as[uint64](x), as[uint64](y)))
	case KInt:
		return mk(KBool, cmp(op, as[int](x), as[int](y)))
	case KBool:
		a, b := as[bool](x), as[bool](y)
		if op == token.EQL {
			return mk(KBool, func(f *frame) bool { return a(f) == b(f) })
		}
		return mk(KBool, func(f *frame) bool { return a(f) != b(f) })
	}
	panic("fuzz: no comparison on " + x.k.goName())
}

// ordered is every kind the relational operators accept. A NaN operand makes
// every one of them false, on both backends, which is why the float kinds are
// here rather than special-cased.
type ordered interface {
	~float32 | ~float64 | ~int32 | ~int64 | ~uint32 | ~uint64 | ~int
}

func cmp[T ordered](op token.Token, x, y func(*frame) T) func(*frame) bool {
	switch op {
	case token.EQL:
		return func(f *frame) bool { return x(f) == y(f) }
	case token.NEQ:
		return func(f *frame) bool { return x(f) != y(f) }
	case token.LSS:
		return func(f *frame) bool { return x(f) < y(f) }
	case token.LEQ:
		return func(f *frame) bool { return x(f) <= y(f) }
	case token.GTR:
		return func(f *frame) bool { return x(f) > y(f) }
	case token.GEQ:
		return func(f *frame) bool { return x(f) >= y(f) }
	}
	panic("fuzz: " + op.String() + " is not a comparison")
}

func (c *compiler) unary(e *Unary) code {
	x := c.expr(e.X)
	if e.Op == token.NOT {
		b := as[bool](x)
		return mk(KBool, func(f *frame) bool { return !b(f) })
	}
	switch x.k {
	case KF32:
		return mk(KF32, neg(e.Op, as[float32](x)))
	case KF64:
		return mk(KF64, neg(e.Op, as[float64](x)))
	case KI32:
		return mk(KI32, negInt(e.Op, as[int32](x)))
	case KI64:
		return mk(KI64, negInt(e.Op, as[int64](x)))
	case KU32:
		return mk(KU32, negInt(e.Op, as[uint32](x)))
	case KU64:
		return mk(KU64, negInt(e.Op, as[uint64](x)))
	case KInt:
		return mk(KInt, negInt(e.Op, as[int](x)))
	}
	panic("fuzz: no unary operator on " + x.k.goName())
}

func neg[T numeric](op token.Token, x func(*frame) T) func(*frame) T {
	if op == token.SUB {
		return func(f *frame) T { return -x(f) }
	}
	return x // unary +, which Go defines as the identity
}

func negInt[T integral](op token.Token, x func(*frame) T) func(*frame) T {
	if op == token.XOR {
		return func(f *frame) T { return ^x(f) }
	}
	return neg(op, x)
}

func (c *compiler) conv(e *Conv) code {
	x := c.expr(e.X)
	if x.k == e.To {
		return x
	}
	switch e.To {
	case KF32:
		return mk(KF32, convTo[float32](x))
	case KF64:
		return mk(KF64, convTo[float64](x))
	case KI32:
		return mk(KI32, convTo[int32](x))
	case KI64:
		return mk(KI64, convTo[int64](x))
	case KU32:
		return mk(KU32, convTo[uint32](x))
	case KU64:
		return mk(KU64, convTo[uint64](x))
	case KInt:
		return mk(KInt, convTo[int](x))
	}
	panic("fuzz: no conversion to " + e.To.goName())
}

// convTo is the source half of a conversion. The float-to-integer cases are
// reachable only through a clamp the generator inserts: Go leaves a conversion
// of a NaN or an out-of-range float implementation-defined and C leaves it
// undefined, so an unclamped one would be a mismatch about nothing.
func convTo[To numeric](x code) func(*frame) To {
	switch x.k {
	case KF32:
		s := as[float32](x)
		return func(f *frame) To { return To(s(f)) }
	case KF64:
		s := as[float64](x)
		return func(f *frame) To { return To(s(f)) }
	case KI32:
		s := as[int32](x)
		return func(f *frame) To { return To(s(f)) }
	case KI64:
		s := as[int64](x)
		return func(f *frame) To { return To(s(f)) }
	case KU32:
		s := as[uint32](x)
		return func(f *frame) To { return To(s(f)) }
	case KU64:
		s := as[uint64](x)
		return func(f *frame) To { return To(s(f)) }
	case KInt:
		s := as[int](x)
		return func(f *frame) To { return To(s(f)) }
	case KI8:
		s := as[int8](x)
		return func(f *frame) To { return To(s(f)) }
	case KI16:
		s := as[int16](x)
		return func(f *frame) To { return To(s(f)) }
	case KU8:
		s := as[uint8](x)
		return func(f *frame) To { return To(s(f)) }
	case KU16:
		s := as[uint16](x)
		return func(f *frame) To { return To(s(f)) }
	}
	panic("fuzz: no conversion from " + x.k.goName())
}

// posFn is gpu.Ctx's position vocabulary. Each one is a method call on the
// real Ctx rather than arithmetic repeated here, so the emulator's answer is
// the emulator's.
func posFn(name string) func(*frame) int {
	switch name {
	case "ThreadIdx":
		return func(f *frame) int { return f.ctx.ThreadIdx() }
	case "ThreadIdxY":
		return func(f *frame) int { return f.ctx.ThreadIdxY() }
	case "ThreadIdxZ":
		return func(f *frame) int { return f.ctx.ThreadIdxZ() }
	case "BlockIdx":
		return func(f *frame) int { return f.ctx.BlockIdx() }
	case "BlockIdxY":
		return func(f *frame) int { return f.ctx.BlockIdxY() }
	case "BlockIdxZ":
		return func(f *frame) int { return f.ctx.BlockIdxZ() }
	case "BlockDim":
		return func(f *frame) int { return f.ctx.BlockDim() }
	case "BlockDimY":
		return func(f *frame) int { return f.ctx.BlockDimY() }
	case "BlockDimZ":
		return func(f *frame) int { return f.ctx.BlockDimZ() }
	case "GridDim":
		return func(f *frame) int { return f.ctx.GridDim() }
	case "GridDimY":
		return func(f *frame) int { return f.ctx.GridDimY() }
	case "GridDimZ":
		return func(f *frame) int { return f.ctx.GridDimZ() }
	case "GlobalID":
		return func(f *frame) int { return f.ctx.GlobalID() }
	case "GlobalIDX":
		return func(f *frame) int { return f.ctx.GlobalIDX() }
	case "GlobalIDY":
		return func(f *frame) int { return f.ctx.GlobalIDY() }
	case "GlobalIDZ":
		return func(f *frame) int { return f.ctx.GlobalIDZ() }
	case "LaneID":
		return func(f *frame) int { return f.ctx.LaneID() }
	}
	panic("fuzz: no position built-in " + name)
}

func (c *compiler) math(e *MathCall) code {
	if e.K == KF64 {
		args := make([]func(*frame) float64, len(e.Args))
		for i, a := range e.Args {
			args[i] = as[float64](c.expr(a))
		}
		return mk(KF64, math64(e.Fn, args))
	}
	args := make([]func(*frame) float32, len(e.Args))
	for i, a := range e.Args {
		args[i] = as[float32](c.expr(a))
	}
	return mk(KF32, math32(e.Fn, args))
}

func math32(fn string, a []func(*frame) float32) func(*frame) float32 {
	switch fn {
	case "Sqrt":
		return func(f *frame) float32 { return gpu.Sqrt(a[0](f)) }
	case "Abs":
		return func(f *frame) float32 { return gpu.Abs(a[0](f)) }
	case "Sin":
		return func(f *frame) float32 { return gpu.Sin(a[0](f)) }
	case "Cos":
		return func(f *frame) float32 { return gpu.Cos(a[0](f)) }
	case "Exp":
		return func(f *frame) float32 { return gpu.Exp(a[0](f)) }
	case "Log":
		return func(f *frame) float32 { return gpu.Log(a[0](f)) }
	case "Hypot":
		return func(f *frame) float32 { return gpu.Hypot(a[0](f), a[1](f)) }
	case "Fmin":
		return func(f *frame) float32 { return gpu.Fmin(a[0](f), a[1](f)) }
	case "Fmax":
		return func(f *frame) float32 { return gpu.Fmax(a[0](f), a[1](f)) }
	}
	panic("fuzz: no math function gpu." + fn)
}

func math64(fn string, a []func(*frame) float64) func(*frame) float64 {
	switch fn {
	case "Sqrt64":
		return func(f *frame) float64 { return gpu.Sqrt64(a[0](f)) }
	case "Abs64":
		return func(f *frame) float64 { return gpu.Abs64(a[0](f)) }
	case "Sin64":
		return func(f *frame) float64 { return gpu.Sin64(a[0](f)) }
	case "Cos64":
		return func(f *frame) float64 { return gpu.Cos64(a[0](f)) }
	case "Exp64":
		return func(f *frame) float64 { return gpu.Exp64(a[0](f)) }
	case "Log64":
		return func(f *frame) float64 { return gpu.Log64(a[0](f)) }
	case "Hypot64":
		return func(f *frame) float64 { return gpu.Hypot64(a[0](f), a[1](f)) }
	case "Fmin64":
		return func(f *frame) float64 { return gpu.Fmin64(a[0](f), a[1](f)) }
	case "Fmax64":
		return func(f *frame) float64 { return gpu.Fmax64(a[0](f), a[1](f)) }
	}
	panic("fuzz: no math function gpu." + fn)
}

func (c *compiler) minmax(e *MinMax) code {
	x, y := c.expr(e.Args[0]), c.expr(e.Args[1])
	hi := e.Fn == "max"
	switch e.K {
	case KF32:
		return mk(KF32, goMinMax(hi, as[float32](x), as[float32](y)))
	case KF64:
		return mk(KF64, goMinMax(hi, as[float64](x), as[float64](y)))
	case KI32:
		return mk(KI32, goMinMax(hi, as[int32](x), as[int32](y)))
	case KI64:
		return mk(KI64, goMinMax(hi, as[int64](x), as[int64](y)))
	case KU32:
		return mk(KU32, goMinMax(hi, as[uint32](x), as[uint32](y)))
	case KU64:
		return mk(KU64, goMinMax(hi, as[uint64](x), as[uint64](y)))
	case KInt:
		return mk(KInt, goMinMax(hi, as[int](x), as[int](y)))
	}
	panic("fuzz: no min/max on " + e.K.goName())
}

// goMinMax is Go's builtin, called as Go's builtin. It is deliberately not
// gpu.Fmin: PLAN.md records that the two disagree about a NaN operand, and the
// oracle has to compute what the generated Go source says rather than what the
// device is expected to answer.
func goMinMax[T ordered](hi bool, x, y func(*frame) T) func(*frame) T {
	if hi {
		return func(f *frame) T { return max(x(f), y(f)) }
	}
	return func(f *frame) T { return min(x(f), y(f)) }
}

func (c *compiler) warp(e *WarpCall) code {
	switch e.Method {
	case "LaneID":
		return mk(KInt, posFn("LaneID"))
	case "ActiveMask":
		return mk(KU32, func(f *frame) uint32 { return f.ctx.ActiveMask() })
	case "Ballot":
		p := as[bool](c.expr(e.Args[0]))
		return mk(KU32, func(f *frame) uint32 { return f.ctx.Ballot(p(f)) })
	case "Any":
		p := as[bool](c.expr(e.Args[0]))
		return mk(KBool, func(f *frame) bool { return f.ctx.Any(p(f)) })
	case "All":
		p := as[bool](c.expr(e.Args[0]))
		return mk(KBool, func(f *frame) bool { return f.ctx.All(p(f)) })
	}
	lane := as[int](c.expr(e.Args[1]))
	if e.K == KF32 {
		v := as[float32](c.expr(e.Args[0]))
		switch e.Method {
		case "ShuffleF32":
			return mk(KF32, func(f *frame) float32 { return f.ctx.ShuffleF32(v(f), lane(f)) })
		case "ShuffleXorF32":
			return mk(KF32, func(f *frame) float32 { return f.ctx.ShuffleXorF32(v(f), lane(f)) })
		case "ShuffleUpF32":
			return mk(KF32, func(f *frame) float32 { return f.ctx.ShuffleUpF32(v(f), lane(f)) })
		case "ShuffleDownF32":
			return mk(KF32, func(f *frame) float32 { return f.ctx.ShuffleDownF32(v(f), lane(f)) })
		}
	}
	v := as[int32](c.expr(e.Args[0]))
	switch e.Method {
	case "ShuffleI32":
		return mk(KI32, func(f *frame) int32 { return f.ctx.ShuffleI32(v(f), lane(f)) })
	case "ShuffleXorI32":
		return mk(KI32, func(f *frame) int32 { return f.ctx.ShuffleXorI32(v(f), lane(f)) })
	case "ShuffleUpI32":
		return mk(KI32, func(f *frame) int32 { return f.ctx.ShuffleUpI32(v(f), lane(f)) })
	case "ShuffleDownI32":
		return mk(KI32, func(f *frame) int32 { return f.ctx.ShuffleDownI32(v(f), lane(f)) })
	}
	panic("fuzz: no warp primitive ctx." + e.Method)
}

func (c *compiler) atomic(e *AtomicCall) code {
	idx := as[int](c.expr(e.Idx))
	if e.Fn == "AtomicAddF32" {
		buf := bufOf(e.Buf).(func(*frame) []float32)
		v := as[float32](c.expr(e.Args[0]))
		return mk(KF32, func(f *frame) float32 { return gpu.AtomicAddF32(buf(f), idx(f), v(f)) })
	}
	buf := bufOf(e.Buf).(func(*frame) []int32)
	v := as[int32](c.expr(e.Args[0]))
	switch e.Fn {
	case "AtomicAddI32":
		return mk(KI32, func(f *frame) int32 { return gpu.AtomicAddI32(buf(f), idx(f), v(f)) })
	case "AtomicMinI32":
		return mk(KI32, func(f *frame) int32 { return gpu.AtomicMinI32(buf(f), idx(f), v(f)) })
	case "AtomicMaxI32":
		return mk(KI32, func(f *frame) int32 { return gpu.AtomicMaxI32(buf(f), idx(f), v(f)) })
	case "AtomicExchI32":
		return mk(KI32, func(f *frame) int32 { return gpu.AtomicExchI32(buf(f), idx(f), v(f)) })
	case "AtomicCASI32":
		w := as[int32](c.expr(e.Args[1]))
		return mk(KI32, func(f *frame) int32 { return gpu.AtomicCASI32(buf(f), idx(f), v(f), w(f)) })
	}
	panic("fuzz: no atomic gpu." + e.Fn)
}

// call compiles a device call. The callee gets its own frame, which is what
// makes its locals its own and its parameters copies, exactly as in Go.
func (c *compiler) call(fn *Func, args []Expr) code {
	binders := make([]func(*frame, *frame), len(args))
	for i, a := range args {
		binders[i] = c.argBinder(fn.Params[i], a)
	}
	body := c.body(fn)
	layout := fn.frame
	invoke := func(f *frame) *frame {
		g := newFrame(layout, f.ctx)
		for _, bind := range binders {
			bind(f, g)
		}
		body(g)
		return g
	}
	switch fn.Result {
	case KInvalid:
		return mk(KInvalid, func(f *frame) struct{} { invoke(f); return struct{}{} })
	case KF32:
		return mk(KF32, func(f *frame) float32 { return invoke(f).rF32 })
	case KF64:
		return mk(KF64, func(f *frame) float64 { return invoke(f).rF64 })
	case KI32:
		return mk(KI32, func(f *frame) int32 { return invoke(f).rI32 })
	case KI64:
		return mk(KI64, func(f *frame) int64 { return invoke(f).rI64 })
	case KU32:
		return mk(KU32, func(f *frame) uint32 { return invoke(f).rU32 })
	case KU64:
		return mk(KU64, func(f *frame) uint64 { return invoke(f).rU64 })
	case KBool:
		return mk(KBool, func(f *frame) bool { return invoke(f).rB })
	case KInt:
		return mk(KInt, func(f *frame) int { return invoke(f).rInt })
	}
	panic("fuzz: no result of kind " + fn.Result.goName())
}

// argBinder evaluates one argument in the caller's frame and stores it in the
// callee's. A slice argument is passed by reference, as Go passes one.
func (c *compiler) argBinder(p *Var, a Expr) func(*frame, *frame) {
	if p.Slice {
		src, ok := a.(*Ref)
		if !ok {
			panic("fuzz: a slice argument must be a plain variable")
		}
		n := p.slot
		return withBufBinder(src.V, n)
	}
	x := c.expr(a)
	n := p.slot
	switch p.Kind {
	case KF32:
		s := as[float32](x)
		return func(f, g *frame) { g.sF32[n] = s(f) }
	case KF64:
		s := as[float64](x)
		return func(f, g *frame) { g.sF64[n] = s(f) }
	case KI32:
		s := as[int32](x)
		return func(f, g *frame) { g.sI32[n] = s(f) }
	case KI64:
		s := as[int64](x)
		return func(f, g *frame) { g.sI64[n] = s(f) }
	case KU32:
		s := as[uint32](x)
		return func(f, g *frame) { g.sU32[n] = s(f) }
	case KU64:
		s := as[uint64](x)
		return func(f, g *frame) { g.sU64[n] = s(f) }
	case KBool:
		s := as[bool](x)
		return func(f, g *frame) { g.sB[n] = s(f) }
	case KInt:
		s := as[int](x)
		return func(f, g *frame) { g.sInt[n] = s(f) }
	}
	panic("fuzz: no parameter of kind " + p.Kind.goName())
}

// withBufBinder moves a buffer from the caller's frame into slot n of the
// callee's.
func withBufBinder(src *Var, n int) func(*frame, *frame) {
	g := bufOf(src)
	switch src.Kind {
	case KF32:
		get := g.(func(*frame) []float32)
		return func(f, h *frame) { h.bF32[n] = get(f) }
	case KF64:
		get := g.(func(*frame) []float64)
		return func(f, h *frame) { h.bF64[n] = get(f) }
	case KI32:
		get := g.(func(*frame) []int32)
		return func(f, h *frame) { h.bI32[n] = get(f) }
	case KI64:
		get := g.(func(*frame) []int64)
		return func(f, h *frame) { h.bI64[n] = get(f) }
	case KU32:
		get := g.(func(*frame) []uint32)
		return func(f, h *frame) { h.bU32[n] = get(f) }
	case KU64:
		get := g.(func(*frame) []uint64)
		return func(f, h *frame) { h.bU64[n] = get(f) }
	case KBool:
		get := g.(func(*frame) []bool)
		return func(f, h *frame) { h.bB[n] = get(f) }
	case KI8:
		get := g.(func(*frame) []int8)
		return func(f, h *frame) { h.bI8[n] = get(f) }
	case KI16:
		get := g.(func(*frame) []int16)
		return func(f, h *frame) { h.bI16[n] = get(f) }
	case KU8:
		get := g.(func(*frame) []uint8)
		return func(f, h *frame) { h.bU8[n] = get(f) }
	case KU16:
		get := g.(func(*frame) []uint16)
		return func(f, h *frame) { h.bU16[n] = get(f) }
	}
	panic("fuzz: no buffer of kind " + src.Kind.goName())
}

// bindArg binds one of the kernel's arguments. The assertion happens here,
// once per launch, rather than in the closure every thread runs.
func bindArg(v *Var, val any) func(*frame) {
	n := v.slot
	if v.Slice && v.Shape != nil {
		// Held as an any, and never unwrapped here: which shape it is was
		// fixed when the program was generated, and the accessors that reach
		// into it know.
		return func(f *frame) { f.bStruct[n] = val }
	}
	if v.Slice {
		switch v.Kind {
		case KF32:
			s := val.([]float32)
			return func(f *frame) { f.bF32[n] = s }
		case KF64:
			s := val.([]float64)
			return func(f *frame) { f.bF64[n] = s }
		case KI32:
			s := val.([]int32)
			return func(f *frame) { f.bI32[n] = s }
		case KI64:
			s := val.([]int64)
			return func(f *frame) { f.bI64[n] = s }
		case KU32:
			s := val.([]uint32)
			return func(f *frame) { f.bU32[n] = s }
		case KU64:
			s := val.([]uint64)
			return func(f *frame) { f.bU64[n] = s }
		case KBool:
			s := val.([]bool)
			return func(f *frame) { f.bB[n] = s }
		case KI8:
			s := val.([]int8)
			return func(f *frame) { f.bI8[n] = s }
		case KI16:
			s := val.([]int16)
			return func(f *frame) { f.bI16[n] = s }
		case KU8:
			s := val.([]uint8)
			return func(f *frame) { f.bU8[n] = s }
		case KU16:
			s := val.([]uint16)
			return func(f *frame) { f.bU16[n] = s }
		}
		panic("fuzz: no slice parameter of kind " + v.Kind.goName())
	}
	switch v.Kind {
	case KF32:
		s := val.(float32)
		return func(f *frame) { f.sF32[n] = s }
	case KF64:
		s := val.(float64)
		return func(f *frame) { f.sF64[n] = s }
	case KI32:
		s := val.(int32)
		return func(f *frame) { f.sI32[n] = s }
	case KI64:
		s := val.(int64)
		return func(f *frame) { f.sI64[n] = s }
	case KU32:
		s := val.(uint32)
		return func(f *frame) { f.sU32[n] = s }
	case KU64:
		s := val.(uint64)
		return func(f *frame) { f.sU64[n] = s }
	case KBool:
		s := val.(bool)
		return func(f *frame) { f.sB[n] = s }
	case KInt:
		s := val.(int)
		return func(f *frame) { f.sInt[n] = s }
	}
	panic("fuzz: no scalar parameter of kind " + v.Kind.goName())
}
