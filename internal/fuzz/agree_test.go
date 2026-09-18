package fuzz

import (
	"go/token"
	"math"
	"testing"

	"github.com/CWBudde/gocuda/gpu"
)

// TestRenderersAgree is the foundation the whole differential test rests on.
//
// Source() and Closure() render the same IR, and if they disagree about what
// it means then every fuzz result is worthless: a mismatch would be the
// generator's and not the emitter's, and there would be nothing in the output
// to tell the two apart. So a handful of programs whose answers are known by
// hand are run both ways -- the Go text through the compiler, the closure
// through the emulator -- and checked against those answers as well as against
// each other.
//
// The cases are chosen where the two renderings could most easily part
// company: an expression whose grouping Go and C read differently, a statement
// that evaluates everything before storing anything, a switch that falls
// through, a branch that leaves two loops, a counter that keeps the ranged
// type, and a builtin whose answer to a NaN is the disagreement NUMERICS.md
// records.
func TestRenderersAgree(t *testing.T) {
	cases := []struct {
		name string
		make func() (*Program, *Args)
		want []string
	}{
		{"shift binds tighter than plus in Go", probeShift, []string{"y 17"}},
		{"bitwise and binds tighter than equality in Go", probeBitEq, []string{"y true"}},
		{"parallel assignment stores into the old index", probeParallel, []string{"y 0 9 0"}},
		// Thread 0 sets 1 and falls into the next clause, which adds 2; thread 1
		// enters that clause on its own from zero; thread 2 takes the default.
		{"a switch falls through only where it says so", probeFallthrough, []string{"y 3 2 3"}},
		{"a labelled break leaves both loops", probeLabelledBreak, []string{"y 1"}},
		{"a range over an int64 counts in int64", probeRangeInt64, []string{"y 10"}},
		{"a narrow element is read through a conversion", probeNarrow, []string{"y 200 -56", "src 200 200"}},
		// The NaN prints as a word rather than as bits: its sign is the
		// hardware's, not the program's. What is being pinned is that it
		// survived the builtin and did not survive gpu.Fmin.
		{"builtin min propagates a NaN where gpu.Fmin does not", probeMinNaN,
			[]string{"y NaN 40400000", "x NaN 40400000"}},
		{"a shared tile is filled, then read across the barrier", probeShared, []string{"y 3 2 1 0"}},
		{"a device function is called for its value", probeDevice, []string{"y 25"}},
	}

	progs := make([]*Program, len(cases))
	args := make([]*Args, len(cases))
	for i, tc := range cases {
		progs[i], args[i] = tc.make()
	}

	// The closure rendering runs here; the Go text is compiled and run in one
	// throwaway module.
	got := make([][]string, len(cases))
	for i := range cases {
		got[i] = runClosure(progs[i], progs[i].CloneArgs(args[i]))
	}
	fromSource := runSources(t, progs, args)

	for i, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if !equalLines(got[i], tc.want) {
				t.Errorf("the closure computed %q, want %q\n%s", got[i], tc.want, progs[i].Source())
			}
			if !equalLines(fromSource[i], tc.want) {
				t.Errorf("the source computed %q, want %q\n%s", fromSource[i], tc.want, progs[i].Source())
			}
		})
	}
}

// TestRenderersAgreeOnGenerated runs the same comparison over generated
// programs, where nobody knows the answer in advance.
//
// It is the half that scales: the hand-written cases say what the renderings
// mean, and this says they go on meaning it over shapes nobody chose. It runs
// few programs because the cost is a compilation each, not a run.
// agreeBase is where the seeds this test uses start. Moving it is how a wider
// sweep is run by hand; the committed value covers a sample rather than a
// corpus, because each program here costs a compilation.
const agreeBase = 1000

func TestRenderersAgreeOnGenerated(t *testing.T) {
	if testing.Short() {
		t.Skip("compiles a module per run")
	}
	const n = 30
	progs := make([]*Program, n)
	args := make([]*Args, n)
	got := make([][]string, n)
	for i := range n {
		progs[i] = Generate(int64(agreeBase + i))
		args[i] = progs[i].Inputs(int64(i))
		got[i] = runClosure(progs[i], progs[i].CloneArgs(args[i]))
	}
	fromSource := runSources(t, progs, args)
	for i := range n {
		if !equalLines(got[i], fromSource[i]) {
			t.Errorf("seed %d: the two renderings disagree\nclosure: %q\nsource:  %q\n%s",
				agreeBase+i, got[i], fromSource[i], progs[i].Source())
		}
	}
}

func equalLines(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// --- the builder -----------------------------------------------------------

// pb assembles a Program by hand. It exists so that a probe reads as the
// kernel it is rather than as a page of struct literals.
type pb struct {
	p    *Program
	fn   *Func
	args []any
	body []Stmt
}

func probe(grid, block int) *pb {
	fn := &Func{Name: "K", Kernel: true, Ctx: true}
	return &pb{
		p:  &Program{Pkg: "kernels", Main: fn, Grid: gpu.D1(grid), Block: gpu.D1(block)},
		fn: fn,
	}
}

func (b *pb) slice(name string, k Kind, val any, readOnly bool) *Var {
	v := &Var{Name: name, Kind: k, Slice: true, Len: sliceLen(val)}
	b.fn.Params = append(b.fn.Params, v)
	b.fn.frame.alloc(v)
	b.p.Params = append(b.p.Params, ParamSpec{Name: name, Kind: k, Slice: true, Len: v.Len, ReadOnly: readOnly, AliasOf: -1})
	b.args = append(b.args, val)
	return v
}

func (b *pb) scalar(name string, k Kind, val any) *Var {
	v := &Var{Name: name, Kind: k}
	b.fn.Params = append(b.fn.Params, v)
	b.fn.frame.alloc(v)
	b.p.Params = append(b.p.Params, ParamSpec{Name: name, Kind: k, ReadOnly: true, AliasOf: -1})
	b.args = append(b.args, val)
	return v
}

// local reserves a variable without declaring it; the caller emits whichever
// declaration form the probe is about.
func (b *pb) local(name string, k Kind, vary bool) *Var {
	v := &Var{Name: name, Kind: k, Vary: vary}
	b.fn.frame.alloc(v)
	return v
}

func (b *pb) tile(name string, k Kind, n int) *Var {
	v := &Var{Name: name, Kind: k, Slice: true, Len: n}
	b.fn.frame.alloc(v)
	return v
}

func (b *pb) add(s ...Stmt) { b.body = append(b.body, s...) }

func (b *pb) build() (*Program, *Args) {
	b.fn.Body = b.body
	b.p.Funcs = append(b.p.Funcs, b.fn)
	return b.p, &Args{Vals: b.args}
}

func sliceLen(v any) int {
	switch v := v.(type) {
	case []float32:
		return len(v)
	case []float64:
		return len(v)
	case []int32:
		return len(v)
	case []int64:
		return len(v)
	case []uint32:
		return len(v)
	case []uint64:
		return len(v)
	case []bool:
		return len(v)
	case []int8:
		return len(v)
	case []uint8:
		return len(v)
	case []int16:
		return len(v)
	case []uint16:
		return len(v)
	}
	panic("fuzz: not a slice")
}

func iLit(k Kind, v int64) Expr                  { return &Lit{K: k, I: v} }
func bin(op token.Token, x, y Expr, k Kind) Expr { return &Binary{Op: op, X: x, Y: y, K: k} }

// --- the probes ------------------------------------------------------------

// probeShift is `y[0] = a<<b + c`, which Go reads as (a<<b) + c and C as
// a << (b+c). With a=3, b=2, c=5 the two answers are 17 and 384.
func probeShift() (*Program, *Args) {
	b := probe(1, 1)
	y := b.slice("y", KI32, []int32{0}, false)
	a := b.scalar("a", KI32, int32(3))
	bb := b.scalar("b", KI32, int32(2))
	c := b.scalar("c", KI32, int32(5))
	b.add(&Assign{
		LHS: Lvalue{V: y, Idx: iLit(KInt, 0)},
		Op:  token.ASSIGN,
		RHS: bin(token.ADD, bin(token.SHL, &Ref{V: a}, &Ref{V: bb}, KI32), &Ref{V: c}, KI32),
	})
	return b.build()
}

// probeBitEq is `y[0] = a&b == c`, which Go reads as (a&b) == c and C as
// a & (b == c). With a=6, b=3, c=2 Go says true.
func probeBitEq() (*Program, *Args) {
	b := probe(1, 1)
	y := b.slice("y", KBool, []bool{false}, false)
	a := b.scalar("a", KI32, int32(6))
	bb := b.scalar("b", KI32, int32(3))
	c := b.scalar("c", KI32, int32(2))
	b.add(&Assign{
		LHS: Lvalue{V: y, Idx: iLit(KInt, 0)},
		Op:  token.ASSIGN,
		RHS: bin(token.EQL, bin(token.AND, &Ref{V: a}, &Ref{V: bb}, KI32), &Ref{V: c}, KBool),
	})
	return b.build()
}

// probeParallel is `i, y[i] = 2, 9` with i starting at 1: everything is
// evaluated first, so the store goes to the old i's element.
func probeParallel() (*Program, *Args) {
	b := probe(1, 1)
	y := b.slice("y", KI32, []int32{0, 0, 0}, false)
	i := b.local("i", KInt, false)
	ti := b.local("", KInt, false)
	tv := b.local("", KI32, false)
	idx := b.local("", KInt, false)
	b.add(&Decl{V: i, Init: iLit(KInt, 1), Form: DeclShort})
	b.add(&ParAssign{
		LHS:  []Lvalue{{V: i}, {V: y, Idx: &Ref{V: i}}},
		RHS:  []Expr{iLit(KInt, 2), iLit(KI32, 9)},
		Vals: []*Var{ti, tv},
		Idxs: []*Var{nil, idx},
	})
	return b.build()
}

// probeFallthrough is a switch whose first clause falls into the second and
// whose second does not fall into the third. Each of the three threads takes
// one clause and they all end up at 3.
func probeFallthrough() (*Program, *Args) {
	b := probe(1, 3)
	y := b.slice("y", KI32, []int32{0, 0, 0}, false)
	i := b.local("i", KInt, true)
	b.add(&Decl{V: i, Init: &Pos{Method: "GlobalID", Vary: true}, Form: DeclShort})
	set := func(v int64) Stmt {
		return &Assign{LHS: Lvalue{V: y, Idx: &Ref{V: i}}, Op: token.ASSIGN, RHS: iLit(KI32, v)}
	}
	add := func(v int64) Stmt {
		return &Assign{LHS: Lvalue{V: y, Idx: &Ref{V: i}}, Op: token.ADD_ASSIGN, RHS: iLit(KI32, v)}
	}
	b.add(&Switch{
		Tag:     &Ref{V: i},
		CSwitch: true,
		Cases: []Case{
			{Vals: []Expr{iLit(KInt, 0)}, Body: []Stmt{set(1)}, Fallthrough: true},
			{Vals: []Expr{iLit(KInt, 1)}, Body: []Stmt{add(2)}},
			{Body: []Stmt{set(3)}},
		},
	})
	return b.build()
}

// probeLabelledBreak leaves two loops at once, so the counter stops at 1
// rather than running the inner loop out three times.
func probeLabelledBreak() (*Program, *Args) {
	b := probe(1, 1)
	y := b.slice("y", KI32, []int32{0}, false)
	n := b.local("n", KI32, false)
	i := b.local("i", KInt, false)
	j := b.local("j", KInt, false)
	b.add(&Decl{V: n, Init: &Conv{To: KI32, X: iLit(KInt, 0)}, Form: DeclShort})
	inner := &For{
		Init: &Decl{V: j, Init: iLit(KInt, 0), Form: DeclShort},
		Cond: bin(token.LSS, &Ref{V: j}, iLit(KInt, 3), KBool),
		Post: &IncDec{LHS: Lvalue{V: j}, Op: token.INC},
		Body: []Stmt{
			&Assign{LHS: Lvalue{V: n}, Op: token.ADD_ASSIGN, RHS: iLit(KI32, 1)},
			&Branch{Tok: token.BREAK, Label: "L1"},
		},
	}
	b.add(&For{
		Label: "L1",
		Init:  &Decl{V: i, Init: iLit(KInt, 0), Form: DeclShort},
		Cond:  bin(token.LSS, &Ref{V: i}, iLit(KInt, 3), KBool),
		Post:  &IncDec{LHS: Lvalue{V: i}, Op: token.INC},
		Body:  []Stmt{inner},
	})
	b.add(&Assign{LHS: Lvalue{V: y, Idx: iLit(KInt, 0)}, Op: token.ASSIGN, RHS: &Ref{V: n}})
	return b.build()
}

// probeRangeInt64 ranges over an int64 and sums the counter, which is an
// int64 too: 0+1+2+3+4 is 10.
func probeRangeInt64() (*Program, *Args) {
	b := probe(1, 1)
	y := b.slice("y", KI64, []int64{0}, false)
	acc := b.local("acc", KI64, false)
	k := b.local("k", KI64, false)
	b.add(&Decl{V: acc, Form: DeclVarZero})
	b.add(&Range{
		Key:  k,
		Over: RangeOver{N: &Conv{To: KI64, X: iLit(KInt, 5)}},
		Body: []Stmt{&Assign{LHS: Lvalue{V: acc}, Op: token.ADD_ASSIGN, RHS: &Ref{V: k}}},
	})
	b.add(&Assign{LHS: Lvalue{V: y, Idx: iLit(KInt, 0)}, Op: token.ASSIGN, RHS: &Ref{V: acc}})
	return b.build()
}

// probeNarrow reads a []uint8 through the conversion that is the only thing
// accepting one, and writes the same byte back two ways: as an int32, where
// 200 stays 200, and as an int8, where it is -56.
func probeNarrow() (*Program, *Args) {
	b := probe(1, 1)
	y := b.slice("y", KI32, []int32{0, 0}, false)
	src := b.slice("src", KU8, []uint8{200, 200}, true)
	b.add(&Assign{
		LHS: Lvalue{V: y, Idx: iLit(KInt, 0)},
		Op:  token.ASSIGN,
		RHS: &Conv{To: KI32, X: &Index{Base: src, Idx: iLit(KInt, 0)}},
	})
	b.add(&Assign{
		LHS: Lvalue{V: y, Idx: iLit(KInt, 1)},
		Op:  token.ASSIGN,
		RHS: bin(token.SUB, &Conv{To: KI32, X: &Index{Base: src, Idx: iLit(KInt, 1)}}, iLit(KI32, 256), KI32),
	})
	return b.build()
}

// probeMinNaN puts a NaN through Go's builtin min, which propagates it, and
// gpu.Fmin, which does not. NUMERICS.md records that CUDA's min agrees with
// fminf and so disagrees with Go's builtin, and that nothing tests it -- this
// pins what the oracle believes, which is Go's answer.
func probeMinNaN() (*Program, *Args) {
	b := probe(1, 1)
	y := b.slice("y", KF32, []float32{0, 0}, false)
	x := b.slice("x", KF32, []float32{float32(math.NaN()), 3}, true)
	load := func(i int64) Expr { return &Index{Base: x, Idx: iLit(KInt, i)} }
	b.add(&Assign{
		LHS: Lvalue{V: y, Idx: iLit(KInt, 0)},
		Op:  token.ASSIGN,
		RHS: &MinMax{Fn: "min", K: KF32, Args: []Expr{load(0), load(1)}},
	})
	b.add(&Assign{
		LHS: Lvalue{V: y, Idx: iLit(KInt, 1)},
		Op:  token.ASSIGN,
		RHS: &MathCall{Fn: "Fmin", K: KF32, Args: []Expr{load(0), load(1)}},
	})
	return b.build()
}

// probeShared fills a tile at each thread's own slot and reads it reversed
// across the barrier, so the output is the thread numbers backwards.
func probeShared() (*Program, *Args) {
	b := probe(1, 4)
	y := b.slice("y", KI32, []int32{0, 0, 0, 0}, false)
	tile := b.tile("s", KI32, 4)
	t := b.local("t", KInt, true)
	b.add(&SharedDecl{V: tile, N: 4})
	b.add(&Decl{V: t, Init: &Pos{Method: "ThreadIdx", Vary: true}, Form: DeclShort})
	b.add(&Assign{LHS: Lvalue{V: tile, Idx: &Ref{V: t}}, Op: token.ASSIGN, RHS: &Conv{To: KI32, X: &Ref{V: t}}})
	b.add(&Barrier{})
	b.add(&Assign{
		LHS: Lvalue{V: y, Idx: &Ref{V: t}},
		Op:  token.ASSIGN,
		RHS: &Index{Base: tile, Idx: bin(token.SUB, iLit(KInt, 3), &Ref{V: t}, KInt)},
	})
	return b.build()
}

// probeDevice calls a helper, which is emitted as a __device__ function in the
// same translation unit. 5*5 is 25.
func probeDevice() (*Program, *Args) {
	b := probe(1, 1)
	y := b.slice("y", KI32, []int32{0}, false)
	arg := &Var{Name: "v", Kind: KI32, Vary: true}
	sq := &Func{Name: "sq", Params: []*Var{arg}, Result: KI32}
	sq.frame.alloc(arg)
	sq.Body = []Stmt{&Return{X: bin(token.MUL, &Ref{V: arg}, &Ref{V: arg}, KI32)}}
	b.p.Funcs = append(b.p.Funcs, sq)
	b.add(&Assign{
		LHS: Lvalue{V: y, Idx: iLit(KInt, 0)},
		Op:  token.ASSIGN,
		RHS: &CallExpr{Fn: sq, Args: []Expr{iLit(KI32, 5)}},
	})
	return b.build()
}
