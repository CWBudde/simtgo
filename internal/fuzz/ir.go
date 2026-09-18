// Package fuzz generates random Go programs that lie inside the subset
// internal/lower accepts, and renders each one twice: as Go source for
// simt.Transpile, and as a Go closure for gpu.RunCPU.
//
// It is one half of a differential fuzzer. The other half runs the generated
// CUDA on a device and compares the buffers the two backends produced, so
// everything here exists to make that comparison mean something.
//
// Three properties are what the design is for.
//
// Yield. An unbiased random AST is outside the subset essentially always, and
// a generator that only produced refusals would be testing the refusal path.
// So nothing is generated and then filtered: the generator is type-directed
// and context-aware, asking for "an expression of kind int32 in a value
// position" and building only what is legal there. SPEC.md is the grammar and
// gen.go follows it rule by rule; the yield is measured by TestYield rather
// than hoped for.
//
// One IR, rendered twice. The Go text and the closure come from the same tree,
// so the oracle is not a second implementation of the language that somebody
// would have to trust. Closure is not an interpreter of Go semantics either --
// it compiles each node to a Go closure performing that very operation on that
// very type, so int32 addition in the IR is int32 addition in the oracle. What
// the two renderings can still disagree about is which tree they are
// rendering, and TestRenderersAgree is what pins that.
//
// No undefined behaviour. The CPU emulator has defined semantics where the
// device has none, so a program that reached UB would manufacture a mismatch
// that says nothing about the emitter. Every shape where the two languages
// part company is closed by construction rather than by luck, and each one is
// commented where it is closed: indices are wrapped into range, integer
// divisors are forced into [1,128] -- C leaves both division by zero and
// INT_MIN/-1 undefined where Go defines them -- shift counts are masked to the
// C width, since Go defines an over-wide shift as zero and C does not, and no
// buffer is bound to two parameters where either is written.
//
// What is deliberately left open, because it is the transpiler's business
// rather than the generator's: signed integer overflow, which Go wraps and C
// leaves undefined. Integer inputs are drawn small enough that a few
// multiplications stay inside int32, but nothing here proves it, and a program
// that overflows is one whose mismatch has to be read with that in mind.
package fuzz

import (
	"go/token"
)

// Kind is a Go type of the subset, as far as the generator models one.
//
// The split SPEC.md §3 draws between the by-value map and the layout map is
// what the last four entries are: they may be a slice element and nothing
// else, and no operator accepts one. KInt is the mirror image -- legal by
// value, where it narrows to C's 32-bit int, and refused in a layout slot,
// where it would be a different stride.
type Kind uint8

const (
	KInvalid Kind = iota
	KF32
	KF64
	KI32
	KI64
	KU32
	KU64
	KBool
	KInt
	// KStruct is a buffer of one of the catalogue's shapes (structs.go). It is
	// a buffer kind and never a value one: nothing in the subset assigns a
	// struct whole, and a Field reached through one has the *field's* kind, so
	// no expression is ever of this kind. It sits before the narrow kinds so
	// that Kind.narrow's range still means what it says.
	KStruct
	KI8
	KI16
	KU8
	KU16
	numKinds
)

// goName is the Go spelling of a kind, which is also how a conversion to it is
// written.
func (k Kind) goName() string {
	switch k {
	case KF32:
		return "float32"
	case KF64:
		return "float64"
	case KI32:
		return "int32"
	case KI64:
		return "int64"
	case KU32:
		return "uint32"
	case KU64:
		return "uint64"
	case KBool:
		return "bool"
	case KInt:
		return "int"
	case KStruct:
		// Named by the shape rather than the kind, which is why a Var carrying
		// this one always carries a Shape as well.
		return "struct"
	case KI8:
		return "int8"
	case KI16:
		return "int16"
	case KU8:
		return "uint8"
	case KU16:
		return "uint16"
	}
	return "<invalid>"
}

// GoName is goName for callers outside the package: the tests that render a
// shape as Go source need to spell a field's type, and the catalogue is
// exported for them already.
func (k Kind) GoName() string { return k.goName() }

// narrow reports whether k is storage only: legal as a slice element and
// accepted by no operator, because Go's 8- and 16-bit arithmetic wraps where
// C's promotes to int.
func (k Kind) narrow() bool { return k >= KI8 && k < numKinds }

// float reports whether k is one of the two floating-point kinds.
func (k Kind) float() bool { return k == KF32 || k == KF64 }

// integer reports whether k is an integer kind an operator may be applied to.
func (k Kind) integer() bool {
	return k == KI32 || k == KI64 || k == KU32 || k == KU64 || k == KInt
}

// signed reports whether k is a signed integer kind.
func (k Kind) signed() bool { return k == KI32 || k == KI64 || k == KInt }

// cWidth is how many bits the kind has in the generated CUDA C, which is what
// a shift count has to be masked against. It is 32 rather than 64 for KInt:
// Go's int is 64 bits and C's is 32, and that narrowing is the subset's one
// deliberate infidelity, so a shift the device can define is a shift under 32.
func (k Kind) cWidth() int {
	switch k {
	case KI64, KU64:
		return 64
	default:
		return 32
	}
}

// A Var is one declared name: a parameter, a local, a shared tile or a range
// variable.
//
// Vary is the thread-varying bit SPEC.md §6 is written in terms of, and it is
// fixed when the variable is declared rather than inferred afterwards. That
// mirrors what internal/lower/diverge.go computes -- its varying set is a
// fixpoint over the whole function, so a variable assigned a varying value
// anywhere is varying everywhere -- and it lets the generator stay inside the
// rule by construction: a uniform variable is never assigned a varying value
// and never assigned under a varying condition.
type Var struct {
	Name  string
	Kind  Kind
	Slice bool // a slice parameter or a shared tile
	Array int  // >0: a local array of this many elements
	// Shape is the struct type of a KStruct buffer, and nil otherwise. It is
	// on the variable rather than on the kind because the kind says only that
	// this is a struct buffer, not which one.
	Shape *StructShape
	Vary  bool
	Len   int // elements, for a buffer whose length the generator knows
	// Fixed says nothing may assign to this variable after it is declared.
	// It is what makes every generated loop terminate: a counter the body
	// could reset, or a bound the body could raise, is a loop whose trip count
	// is somebody else's business, and a fuzz run that does not return reports
	// nothing at all.
	Fixed bool

	slot int // index into the frame's storage for this kind
}

// buffer reports whether the variable addresses memory rather than holding a
// value, which is what decides both how it is indexed and which half of the
// frame it lives in.
func (v *Var) buffer() bool { return v.Slice || v.Array > 0 }

// An Expr is one expression node. Every node knows its Go kind and whether it
// may hold different values in two threads of the same block; the renderers
// are type switches over this set, in source.go and closure.go, so the two
// cannot drift apart without one of them failing to compile.
type Expr interface {
	kind() Kind
	varies() bool
}

// Lit is a typed constant. Only finite, exactly representable values appear
// here: Go has no NaN or infinity literal, so the special values a numerics
// fuzzer wants arrive through input buffers instead (values.go).
type Lit struct {
	K Kind
	F float64 // KF32, KF64
	I int64   // signed kinds
	U uint64  // unsigned kinds
	B bool
}

func (l *Lit) kind() Kind   { return l.K }
func (l *Lit) varies() bool { return false }

// Ref reads a scalar variable.
type Ref struct{ V *Var }

func (r *Ref) kind() Kind   { return r.V.Kind }
func (r *Ref) varies() bool { return r.V.Vary }

// Index reads one element of a slice, a shared tile or a local array.
//
// Idx is always already in range: the generator wraps every index it does not
// otherwise know to be safe (gen.go, wrapIndex), because an out-of-range read
// panics in the emulator and is undefined on the device, and neither says
// anything about the emitter.
type Index struct {
	Base *Var
	Idx  Expr
}

func (x *Index) kind() Kind { return x.Base.Kind }

// varies follows diverge.go: a slice is one piece of memory the whole block
// addresses, so what may differ between threads is the index. A local array is
// the opposite -- it is per-thread storage -- so it varies if the variable
// does.
func (x *Index) varies() bool {
	if x.Base.Slice {
		return x.Idx.varies()
	}
	return x.Base.Vary || x.Idx.varies()
}

// Field reads one field of one element of a struct buffer: p[i].A.
//
// Its kind is the *field's*, not the struct's, and that is the whole reason
// structs fit the generator at all. Every per-kind compiler in closure.go --
// there are some two hundred case arms of them -- then handles the result
// unchanged, and nothing had to learn that a struct exists except the four
// places that reach one.
type Field struct {
	Base *Var
	Idx  Expr
	F    int
}

func (x *Field) kind() Kind { return x.Base.Shape.Fields[x.F].Kind }

// varies follows Index: the buffer is one piece of memory the whole block
// addresses, so what may differ between threads is the index.
func (x *Field) varies() bool { return x.Idx.varies() }

// Len is len(x) on a slice parameter, a shared tile or a local array. It is
// block-uniform, which is what makes it legal to branch a barrier on.
type Len struct{ Base *Var }

func (l *Len) kind() Kind   { return KInt }
func (l *Len) varies() bool { return false }

// Binary is a binary operator. K is the result kind, which differs from the
// operands' for a comparison, and for a shift Go takes the left operand's.
type Binary struct {
	Op   token.Token
	X, Y Expr
	K    Kind
}

func (b *Binary) kind() Kind   { return b.K }
func (b *Binary) varies() bool { return b.X.varies() || b.Y.varies() }

// Unary is one of the four prefix operators the subset keeps: -, +, ! and ^.
type Unary struct {
	Op token.Token
	X  Expr
}

func (u *Unary) kind() Kind   { return u.X.kind() }
func (u *Unary) varies() bool { return u.X.varies() }

// Conv is a conversion T(x). It is the only construct that accepts a narrow
// kind, which is how a []uint8 is read at all.
type Conv struct {
	To Kind
	X  Expr
}

func (c *Conv) kind() Kind   { return c.To }
func (c *Conv) varies() bool { return c.X.varies() }

// Pos is one of gpu.Ctx's position built-ins. Vary is not derived from the
// method name here but recorded when the node is built, from the one list
// diverge.go keeps: the block's own coordinates and the grid's shape are
// uniform and everything else is not.
type Pos struct {
	Method string
	Vary   bool
}

func (p *Pos) kind() Kind   { return KInt }
func (p *Pos) varies() bool { return p.Vary }

// MathCall is one of package gpu's math functions. Fn is the Go spelling; K is
// float32 for the unsuffixed half and float64 for the 64 half, whose use is
// what makes a program need //gocuda:float64.
type MathCall struct {
	Fn   string
	Args []Expr
	K    Kind
}

func (m *MathCall) kind() Kind   { return m.K }
func (m *MathCall) varies() bool { return anyVaries(m.Args) }

// MinMax is Go's builtin min or max.
//
// It is generated on purpose even though NUMERICS.md records that the two backends
// disagree about it -- Go's builtin propagates a NaN operand and CUDA's
// min/max return the other one. A generator that avoided the shape would be
// hiding a known defect from the fuzzer it feeds.
type MinMax struct {
	Fn   string // "min" or "max"
	Args []Expr
	K    Kind
}

func (m *MinMax) kind() Kind   { return m.K }
func (m *MinMax) varies() bool { return anyVaries(m.Args) }

// WarpCall is one of the warp-level primitives. Mask says whether the emitter
// writes 0xffffffff into it, which is the promise that every lane of the warp
// is present and therefore the thing the divergence rules police; LaneID and
// ActiveMask do not carry one.
type WarpCall struct {
	Method string
	Args   []Expr
	K      Kind
	Mask   bool
}

func (w *WarpCall) kind() Kind { return w.K }

// varies is unconditionally true: every warp primitive answers a question
// about this lane, which is what makes it thread-varying in diverge.go's
// lattice.
func (w *WarpCall) varies() bool { return true }

// AtomicCall is one of package gpu's atomics. Buf is a plain variable rather
// than an expression because the vocabulary takes a buffer and an index: the
// subset has no address-of, so the emitter is the only thing that can write
// &s[i], and it needs to see which buffer.
type AtomicCall struct {
	Fn   string
	Buf  *Var
	Idx  Expr
	Args []Expr
	K    Kind
}

func (a *AtomicCall) kind() Kind { return a.K }

// varies is true because the value an atomic returns is what the element held
// before this thread's turn, which depends on the order the threads arrived.
func (a *AtomicCall) varies() bool { return true }

// CallExpr calls a device function for its result.
type CallExpr struct {
	Fn   *Func
	Args []Expr
}

func (c *CallExpr) kind() Kind { return c.Fn.Result }

// varies is unconditionally true, and not because a call of uniform arguments
// really does vary.
//
// internal/lower/diverge.go's varyingCall takes every call into the kernel
// package to vary whatever it was handed: summarising what a device function's
// result depends on would be a fourth answer per declaration, and assuming it
// varies costs only a refusal. The generator has to agree with the rule that
// is enforced rather than with the one SPEC.md §6 describes, which lists what
// is block-uniform and does not mention calls at all.
func (c *CallExpr) varies() bool { return true }

// anyVaries is the varying rule for an argument list: a call is thread-varying
// when anything it is handed is.
func anyVaries(args []Expr) bool {
	for _, a := range args {
		if a.varies() {
			return true
		}
	}
	return false
}

// A Stmt is one statement node.
type Stmt interface{ stmt() }

// DeclForm is which of the three spellings a declaration uses. All three are
// generated because they are not interchangeable in the transpiler: the
// explicit var form is a different AST node, and it has escaped a refusal the
// other two hit.
type DeclForm uint8

const (
	// DeclShort is `x := e`.
	DeclShort DeclForm = iota
	// DeclVarTyped is `var x T = e`.
	DeclVarTyped
	// DeclVarZero is `var x T`, with no initialiser.
	DeclVarZero
)

// Decl declares a local. Init is nil exactly when Form is DeclVarZero.
type Decl struct {
	V    *Var
	Init Expr
	Form DeclForm
}

func (*Decl) stmt() {}

// ArrayDecl declares a local array, `var a [N]T`. An array is storage rather
// than a value -- it cannot be a parameter, a result, assigned whole or
// compared -- so this is the only way one comes into existence.
type ArrayDecl struct{ V *Var }

func (*ArrayDecl) stmt() {}

// An Lvalue is somewhere a statement can store: a scalar variable, or one
// element of a buffer.
type Lvalue struct {
	V   *Var // the scalar when Idx is nil, otherwise the buffer
	Idx Expr
	// F selects a field when V is a struct buffer. It is read only when
	// V.Shape is non-nil, which is what makes the zero value safe for every
	// other Lvalue -- and there are many, all of them written before structs
	// existed.
	F int
}

func (l Lvalue) kind() Kind {
	if l.V.Shape != nil {
		return l.V.Shape.Fields[l.F].Kind
	}
	return l.V.Kind
}

// Assign is `lhs op= rhs`, with Op == token.ASSIGN for a plain store.
type Assign struct {
	LHS Lvalue
	Op  token.Token
	RHS Expr
}

func (*Assign) stmt() {}

// ParAssign is a parallel assignment, `a, b = b, a`.
//
// It is here for the shape SPEC.md §2 calls out: everything is evaluated
// first, so `i, y[i] = 2, 7` stores into the old i's element. The generator
// keeps every index in one of these free of calls, so that evaluating it once
// and evaluating it twice are the same thing and the two renderings cannot
// disagree about how many times it happened.
// The temporaries are part of the node rather than of its compilation: they
// are frame slots, so they are per-call, and a closure-level Go variable would
// be shared by every thread of the launch.
type ParAssign struct {
	LHS []Lvalue
	RHS []Expr
	// Vals holds one temporary per right-hand side, of that side's kind.
	Vals []*Var
	// Idxs holds one temporary per left-hand side that indexes a buffer, of
	// kind int, and nil where the target is a scalar.
	Idxs []*Var
}

func (*ParAssign) stmt() {}

// IncDec is `x++` or `x--`.
type IncDec struct {
	LHS Lvalue
	Op  token.Token
}

func (*IncDec) stmt() {}

// If is an if statement. Else is nil, or the statements of an else block --
// an else-if is one *If in that list.
type If struct {
	Cond Expr
	Then []Stmt
	Else []Stmt
}

func (*If) stmt() {}

// For is the three-clause loop. Init and Post are nil or a simple statement;
// Label is "" unless a branch inside names it.
type For struct {
	Init  Stmt
	Cond  Expr
	Post  Stmt
	Body  []Stmt
	Label string
}

func (*For) stmt() {}

// RangeOver is what a range statement ranges over: a buffer, or an integer.
type RangeOver struct {
	Buf *Var // non-nil: range over a slice, tile or array
	N   Expr // non-nil: range over an integer expression
}

// Range is `for i := range x` and `for i, v := range x`. Val is nil in the
// first form, and is never a narrow kind: a narrow value would be a local of a
// storage-only type, which the subset refuses.
type Range struct {
	Key   *Var
	Val   *Var
	Over  RangeOver
	Body  []Stmt
	Label string
}

func (*Range) stmt() {}

// Case is one clause of a switch. Vals is empty for the default clause.
type Case struct {
	Vals        []Expr
	Body        []Stmt
	Fallthrough bool
}

// Switch is a switch statement. Tag is nil for the tagless form.
//
// Which of the two lowerings it gets is decided by the transpiler, not here:
// an integral tag whose every case folds to a constant becomes a C switch, and
// everything else becomes the if/else chain Go's semantics describe. The
// generator cares because a bare break and a fallthrough are legal only in the
// first, which is what CSwitch records.
type Switch struct {
	Tag     Expr
	Cases   []Case
	CSwitch bool
}

func (*Switch) stmt() {}

// Branch is break, continue or fallthrough. Label is "" for the bare forms.
type Branch struct {
	Tok   token.Token
	Label string
}

func (*Branch) stmt() {}

// Return leaves a function. X is nil for a bare return, which is the only form
// a kernel or a resultless device function has.
type Return struct{ X Expr }

func (*Return) stmt() {}

// Barrier is ctx.SyncThreads(). It is a statement rather than an expression
// because that is what it is in Go -- SyncThreads returns nothing -- and
// diverge.go relies on it: a barrier can only ever be an ExprStmt, so a
// structured walk over the statement tree sees every one.
type Barrier struct{}

func (*Barrier) stmt() {}

// SyncWarp is ctx.SyncWarp(), which carries the same full-warp promise every
// other _sync built-in does.
type SyncWarp struct{}

func (*SyncWarp) stmt() {}

// SharedDecl declares a shared tile. Dyn is the one whose length the launch
// supplies; a kernel gets at most one, because CUDA has a single dynamic
// __shared__ block and a second declaration would silently alias it.
type SharedDecl struct {
	V   *Var
	N   int
	Dyn bool
}

func (*SharedDecl) stmt() {}

// Assume is ctx.AssumeBlockDim(n). It emits no code; it records the block size
// the kernel was written for, which both Kernel.Launch and RunCPU enforce, so
// the generator only ever emits the block size it has already chosen.
type Assume struct{ N int }

func (*Assume) stmt() {}

// CallStmt calls a device function for its effects.
type CallStmt struct {
	Fn   *Func
	Args []Expr
}

func (*CallStmt) stmt() {}

// Discard is a call evaluated as a statement, its result dropped.
//
// It is how the atomics are generated. What one returns is the value the
// element held before this thread's turn, which is a function of the order the
// threads arrived in -- so the device and the emulator may legitimately give
// different answers, and a differential test can only use an atomic whose
// result nothing reads.
type Discard struct{ X Expr }

func (*Discard) stmt() {}

// A Func is one generated function: the kernel, or a helper it reaches.
//
// Ctx says the function takes a gpu.Ctx, which is what makes a function a
// kernel -- so a helper that takes one must carry //gocuda:device, and that is
// exactly the pair Device records.
type Func struct {
	Name   string
	Kernel bool
	Ctx    bool
	Device bool // emit //gocuda:device
	Params []*Var
	Result Kind // KInvalid when the function returns nothing
	Body   []Stmt

	frame frameLayout
}

// frameLayout is how many storage slots of each kind one call of a function
// needs. It is filled while the function is generated, so compiling the
// closure allocates exactly what the body uses and no more.
type frameLayout struct {
	scalars [numKinds]int
	buffers [numKinds]int
}

// alloc reserves a slot for v and records it on the variable.
func (f *frameLayout) alloc(v *Var) {
	if v.buffer() {
		v.slot = f.buffers[v.Kind]
		f.buffers[v.Kind]++
		return
	}
	v.slot = f.scalars[v.Kind]
	f.scalars[v.Kind]++
}
