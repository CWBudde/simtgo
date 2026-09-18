package fuzz

import (
	"fmt"
	"go/token"
	"math/rand/v2"

	"github.com/CWBudde/gocuda/gpu"
)

// Generate builds one program from a seed.
//
// It is deterministic: the same seed produces an identical Program, source
// text and closure, on any machine, because every choice comes from the
// seeded generator and nothing here reads a map in its iteration order. That
// is what makes a fuzz failure reproducible from one number, and TestDeterminism
// is what holds it to it.
func Generate(seed int64) *Program {
	g := &gen{r: rand.New(rand.NewPCG(uint64(seed), 0x2545f4914f6cdd1d))}
	return g.program(seed)
}

// gen carries the state of one program's generation.
//
// The four flags are what keep the yield high, and each answers a question
// before a node is built rather than after: uniform, whether the statement
// being generated is on the block's common path, which is what a barrier and a
// masked warp primitive require; uni, whether the expression must be
// block-uniform, which is what a condition guarding one requires; pure,
// whether it must be free of calls, which a switch tag needs so that the two
// lowerings of a switch mean the same thing; and noWarp, whether a masked warp
// primitive would land somewhere the divergence rules do not reach.
type gen struct {
	r *rand.Rand
	p *Program

	fn     *Func
	scopes [][]*Var
	depth  map[string]int // innermost scope depth each name is declared at

	uniform bool
	uni     bool
	pure    bool
	// noWarp closes the three expression positions a masked warp primitive
	// must not appear in. Two of them the divergence walk reads against the
	// construct's own site rather than its enclosing one -- a case expression
	// of a tagless switch, and a loop's condition, both re-evaluated by
	// whichever threads are still there. The third it does not read at all:
	// the right operand of && or ||, which short-circuits, so the call is made
	// by whichever lanes got that far with a mask naming all of them. SPEC.md
	// §6 states that last one as a limit of the rules rather than something
	// they catch.
	noWarp bool

	sync   bool // this kernel may contain barriers and masked warp primitives
	warpOK bool // the block is a whole number of warps
	budget int

	// pending holds statements a form needs emitted just before itself -- a
	// loop hoisting its bound into a variable of its own is the only one.
	pending []Stmt

	loops    []*loopInfo // the enclosing loops, innermost last
	inSwitch bool        // inside a switch clause, where a bare break means the switch
	inTail   bool        // inside the guard that makes this thread's own slot exist

	helpers []*Func
	pureFns []*Func // helpers usable inside an expression

	inBufs  []*Var // read-only parameters, indexable anywhere
	outBufs []*Var // written parameters, touched only at this thread's own slot
	// structIn is a read-only struct buffer and structOut a written one, or
	// nil. They are held apart from inBufs and outBufs because a struct is
	// never read or written whole -- only one field of one element -- so the
	// places that index a buffer must not reach for one.
	structIn  *Var
	structOut *Var
	// atomBuf is the buffer only the atomics touch, and atomFn the single
	// operation they use on it. One operation, because two of them do not
	// commute with each other: an add and a min over one element answer
	// differently depending on which arrived first, and the order threads
	// arrive in is exactly what neither backend promises.
	atomBuf *Var
	atomFn  string
	tiles   []*Var // shared tiles, readable after the fill barrier

	gid *Var
	tid *Var

	// labelN numbers the labels of the whole program.
	//
	// A label's scope is one function body, so numbering per function would be
	// correct Go -- and internal/lower refuses it. Its label map is keyed by
	// name and shared across the translation unit, and a device function
	// lowered from inside a labelled loop deletes the caller's entry on the
	// way out, after which the caller's own `continue L1` is refused with "L1
	// does not label a for loop". The generator numbers across the program to
	// stay out of it; the refusal is a finding, not a rule.
	labelN int

	// noShadow are the names nothing may be declared over. Parameters and
	// buffers are in it because the generator keeps its own lists of those and
	// picks from them by variable, while the source resolves the name -- so a
	// local shadowing a buffer would turn len(x) into a length of something
	// else. Shadowing a local is left alone, which is where it is worth
	// generating anyway.
	noShadow map[string]bool

	nameN int
}

// --- program ---------------------------------------------------------------

func (g *gen) program(seed int64) *Program {
	p := &Program{Seed: seed, Pkg: "kernels"}
	g.p = p
	g.geometry()
	p.Float64 = g.r.IntN(4) == 0
	g.sync = g.r.IntN(2) == 0
	g.warpOK = p.Threads()%gpu.WarpSize == 0 && p.Threads() >= gpu.WarpSize

	k := &Func{Name: "K", Kernel: true, Ctx: true}
	p.Main = k
	g.fn = k
	g.depth = map[string]int{}
	g.noShadow = map[string]bool{}
	g.push()
	g.params(k)
	g.genHelpers()
	g.fn = k
	g.budget = 14 + g.r.IntN(14)
	k.Body = g.kernelBody()
	g.pop()
	fixUnused(k)

	if g.r.IntN(3) == 0 {
		// The kernel first, helpers after. Declaration order must not matter:
		// the emitter writes prototypes ahead of the definitions precisely so
		// that it does not, and that is worth generating rather than assuming.
		p.Funcs = append([]*Func{k}, g.helpers...)
	} else {
		p.Funcs = append(append([]*Func{}, g.helpers...), k)
	}
	return p
}

// geometry chooses the launch, independently of what the kernel will turn out
// to be.
//
// Ragged on purpose: a launch whose thread count divides the buffer length
// exactly never takes the guard a kernel writes around its tail, and a real
// divergence bug in this repository hid behind one. The grid is one
// dimensional and the block at most two, because every global store the
// generator emits goes to the thread's own slot and that argument needs a flat
// thread number it can write down.
func (g *gen) geometry() {
	blocks := []int{1, 2, 3, 5, 7}
	g.p.Grid = gpu.D1(blocks[g.r.IntN(len(blocks))])
	if g.r.IntN(4) == 0 {
		x := []int{8, 16, 32}[g.r.IntN(3)]
		y := 2 + g.r.IntN(3)
		g.p.Block = gpu.D2(x, y)
		return
	}
	// The whole-warp sizes are over-represented, because the warp primitives
	// are only generated for a block that is one: the emitter writes
	// 0xffffffff into every _sync built-in, and in a short warp that mask
	// names lanes which do not exist. The odd sizes are still here, and are
	// what a kernel's tail guard is exercised by.
	//
	// Ten of sixteen rather than seven of thirteen: this is the one lever on
	// warp coverage that costs nothing elsewhere. The alternative -- raising
	// g.sync above its coin flip -- would have moved more, and was rejected
	// because a program with a barrier or a warp primitive is out of the host
	// oracle's scope, so buying warp coverage that way is paid for directly
	// out of the differential's reach.
	widths := []int{1, 2, 3, 7, 31, 32, 32, 32, 33, 64, 64, 64, 96, 96, 128, 128}
	g.p.Block = gpu.D1(widths[g.r.IntN(len(widths))])
}

// valueKinds are the kinds a parameter, a local or an expression may have.
func (g *gen) valueKinds() []Kind {
	k := []Kind{KF32, KI32, KI64, KU32, KU64, KInt, KBool}
	if g.p.Float64 {
		k = append(k, KF64)
	}
	return k
}

// elemKinds are the kinds a slice element may have. It is not the same list:
// int is refused as an element because it is a different stride there, and the
// four narrow kinds are legal only here.
func (g *gen) elemKinds() []Kind {
	k := []Kind{KF32, KI32, KI64, KU32, KU64, KBool, KI8, KI16, KU8, KU16}
	if g.p.Float64 {
		k = append(k, KF64)
	}
	return k
}

// storeKinds are the kinds an output buffer may have: the narrow ones are left
// out because no operator accepts one, so a kernel could only ever store a
// conversion of something into it, which says less than storing a computed
// value.
func (g *gen) storeKinds() []Kind {
	k := []Kind{KF32, KI32, KI64, KU32, KU64, KBool}
	if g.p.Float64 {
		k = append(k, KF64)
	}
	return k
}

func (g *gen) params(k *Func) {
	threads := g.p.Threads()
	outLen := g.raggedLen(threads)

	nOut := 1 + g.r.IntN(2)
	for range nOut {
		v := &Var{Name: g.paramName(), Kind: pick(g.r, g.storeKinds()), Slice: true, Len: outLen}
		g.addParam(k, v, false)
		g.outBufs = append(g.outBufs, v)
	}
	if g.r.IntN(3) == 0 {
		kind := KI32
		if g.r.IntN(2) == 0 {
			kind = KF32
		}
		v := &Var{Name: g.paramName(), Kind: kind, Slice: true, Len: 1 + g.r.IntN(8)}
		g.addParam(k, v, false)
		g.atomBuf = v
		g.atomFn = "AtomicAddF32"
		if kind == KI32 {
			g.atomFn = pick(g.r, []string{"AtomicAddI32", "AtomicMinI32", "AtomicMaxI32"})
		}
	}
	// The inputs are shaped before any of them is given a frame slot, because
	// the aliasing below retypes one and a slot belongs to a kind.
	nIn := 1 + g.r.IntN(3)
	ins := make([]*Var, nIn)
	for i := range ins {
		ins[i] = &Var{Name: g.paramName(), Kind: pick(g.r, g.elemKinds()), Slice: true, Len: g.raggedLen(threads)}
	}
	alias := g.alias(ins)
	for _, v := range ins {
		g.addParam(k, v, true)
		g.inBufs = append(g.inBufs, v)
	}
	if alias >= 0 {
		g.p.Params[len(g.p.Params)-nIn+alias+1].AliasOf = len(g.p.Params) - nIn + alias
	}
	for range g.r.IntN(3) {
		v := &Var{Name: g.paramName(), Kind: pick(g.r, g.valueKinds())}
		g.addParam(k, v, true)
	}
	g.structParams(k, threads, outLen)
}

// structParams adds a struct buffer to read from and, less often, one to write
// to.
//
// They are worth generating for what the emitter does with the *type* rather
// than with the values: every hole becomes a gocuda_padN member and the whole
// struct gets a sizeof assertion, which is what stands in for the field offsets
// NVRTC has no offsetof to check. So the test is mostly that NVRTC accepts the
// declaration at all, and that is why a struct earns its place even in a
// program that only reads one field.
//
// Not every program gets one. A struct in every kernel would crowd out the
// shapes the generator already covers, and the point is to add a seam, not to
// replace the others with it.
func (g *gen) structParams(k *Func, threads, outLen int) {
	if g.r.IntN(3) != 0 {
		return
	}
	shape := pick(g.r, Shapes)
	g.p.Shapes = append(g.p.Shapes, shape)

	in := &Var{Name: g.paramName(), Kind: KStruct, Shape: shape, Slice: true, Len: g.raggedLen(threads)}
	g.addParam(k, in, true)
	g.structIn = in

	// A written one only where the shape has a field worth writing: a narrow
	// field takes no value the subset can produce except a conversion, which
	// would say nothing the slice case does not.
	if g.r.IntN(2) != 0 || !shapeHasWritableField(shape) {
		return
	}
	// The same length as the other outputs, because the tail that writes them
	// is guarded by `gid < len(outBufs[0])` and this is written in it. A buffer
	// of its own length would be indexed past its end on the first thread the
	// guard let through and the other did not.
	out := &Var{Name: g.paramName(), Kind: KStruct, Shape: shape, Slice: true, Len: outLen}
	g.addParam(k, out, false)
	g.structOut = out
}

// shapeHasWritableField reports whether anything in the shape can be stored to.
func shapeHasWritableField(shape *StructShape) bool {
	for _, f := range shape.Fields {
		if f.writable() {
			return true
		}
	}
	return false
}

// raggedLen is a buffer length that is deliberately not a multiple of the
// launch: shorter than the grid, a little longer, or an awkward multiple.
func (g *gen) raggedLen(threads int) int {
	switch g.r.IntN(5) {
	case 0:
		return max(1, threads-1-g.r.IntN(threads))
	case 1:
		return threads + 1 + g.r.IntN(7)
	case 2:
		return threads
	case 3:
		return threads*2 - 1
	}
	return max(1, threads/2+g.r.IntN(5))
}

func (g *gen) addParam(k *Func, v *Var, readOnly bool) {
	k.Params = append(k.Params, v)
	k.frame.alloc(v)
	g.declare(v)
	g.p.Params = append(g.p.Params, ParamSpec{
		Name: v.Name, Kind: v.Kind, Slice: v.Slice, Len: v.Len,
		ReadOnly: readOnly, AliasOf: -1, Shape: v.Shape,
	})
}

// alias binds two read-only parameters of the same kind and length to one
// buffer.
//
// It is generated because the boundary is subtle and worth exercising from the
// host side: two read-only pointers may share, and what __restrict__ forbids
// is reaching a modified object through another, so dot(x, x) is sound. A
// written parameter is never shared -- the emulator cannot see aliasing at
// all, so that program would manufacture a mismatch about nothing.
// alias makes ins[i] and ins[i+1] one buffer and returns i, or -1 for none.
//
// The pair is made rather than looked for: two inputs that happen to share an
// element kind and a length are rare among random parameters, and waiting for
// one would leave this shape almost never generated.
func (g *gen) alias(ins []*Var) int {
	if g.r.IntN(3) != 0 || len(ins) < 2 {
		return -1
	}
	i := g.r.IntN(len(ins) - 1)
	ins[i+1].Kind, ins[i+1].Len = ins[i].Kind, ins[i].Len
	return i
}

// --- kernel body -----------------------------------------------------------

func (g *gen) kernelBody() []Stmt {
	var body []Stmt
	threads := g.p.Threads()

	if g.r.IntN(2) == 0 {
		// The block size is the one already chosen, so the contract the source
		// records and the launch that will run it cannot disagree.
		body = append(body, &Assume{N: threads})
	}

	// The flat thread number within the block, and the flat thread number
	// within the grid. Every global store goes to the second of these, which
	// is what makes two threads never write one element.
	// Both are thread-varying: they are what a thread's own position is. The
	// divergence rules turn on exactly this, so getting the bit wrong here
	// would have the generator writing barriers under conditions it believes
	// uniform and the transpiler refusing every one of them.
	g.tid = g.local(KInt, false)
	g.noShadow[g.tid.Name] = true
	body = append(body, &Decl{V: g.tid, Init: g.flatTid(), Form: DeclShort})
	g.declare(g.tid)
	g.gid = g.local(KInt, false)
	// Neither may be shadowed: every global store is written at the slot the
	// grid number names, and a local of another type wearing that name would
	// silently make it somebody else's index.
	g.noShadow[g.gid.Name] = true
	body = append(body, &Decl{V: g.gid, Init: g.flatGid(), Form: DeclShort})
	g.declare(g.gid)

	body = append(body, g.sharedTiles()...)

	g.uniform = true
	body = append(body, g.stmts(2+g.r.IntN(4))...)

	// The tail of the kernel: everything that writes, under the one guard that
	// makes the thread's own slot exist.
	g.uniform = false
	g.inTail = true
	g.push()
	tail := g.stores()
	tail = append(tail, g.stmts(2+g.r.IntN(4))...)
	tail = append(tail, g.stores()...)
	g.pop()
	g.inTail = false
	body = append(body, &If{Cond: &Binary{Op: token.LSS, X: &Ref{V: g.gid}, Y: &Len{Base: g.outBufs[0]}, K: KBool}, Then: tail})
	return body
}

// flatTid is the thread's number within its block, counted across the axes the
// launch uses. A one-dimensional block is ctx.ThreadIdx() and nothing more.
func (g *gen) flatTid() Expr {
	if g.p.Block.Y == 1 {
		return &Pos{Method: "ThreadIdx"}
	}
	return &Binary{Op: token.ADD, K: KInt,
		X: &Binary{Op: token.MUL, K: KInt, X: &Pos{Method: "ThreadIdxY"}, Y: &Pos{Method: "BlockDim", Vary: false}},
		Y: &Pos{Method: "ThreadIdx"}}
}

// flatGid is the thread's number within the grid. For a one-dimensional block
// that is exactly gpu.GlobalID, which is worth spelling as itself.
func (g *gen) flatGid() Expr {
	if g.p.Block.Y == 1 {
		return &Pos{Method: "GlobalID"}
	}
	perBlock := &Binary{Op: token.MUL, K: KInt, X: &Pos{Method: "BlockDim"}, Y: &Pos{Method: "BlockDimY"}}
	return &Binary{Op: token.ADD, K: KInt,
		X: &Binary{Op: token.MUL, K: KInt, X: &Pos{Method: "BlockIdx"}, Y: perBlock},
		Y: &Ref{V: g.tid}}
}

// sharedTiles declares up to two tiles, fills each at this thread's own slot
// and closes the fill with one barrier.
//
// Filling at tid and nowhere else is what keeps the fill free of races, and
// the barrier is what makes every later read of the tile see it. Both are
// properties of the generator rather than of the subset, which would accept a
// racy fill and hand the difference to the comparison.
func (g *gen) sharedTiles() []Stmt {
	if !g.sync || g.r.IntN(3) == 0 {
		return nil
	}
	threads := g.p.Threads()
	var decls []*Var
	var out []Stmt
	dynUsed := false
	for range 1 + g.r.IntN(2) {
		kind := pick(g.r, []Kind{KF32, KI32, KI64, KU32})
		if g.p.Float64 && g.r.IntN(4) == 0 {
			kind = KF64
		}
		v := &Var{Name: g.paramName(), Kind: kind, Slice: true, Len: threads + g.r.IntN(4)}
		dyn := !dynUsed && g.r.IntN(3) == 0
		if dyn {
			// CUDA has one dynamic __shared__ block, so a kernel gets one
			// tile; a second declaration would silently alias the first.
			dynUsed = true
			g.p.HasDyn = true
			g.p.DynLen = v.Len
		}
		g.fn.frame.alloc(v)
		out = append(out, &SharedDecl{V: v, N: v.Len, Dyn: dyn})
		g.declare(v)
		decls = append(decls, v)
	}
	// The fill happens before any tile is readable, so its values come from
	// the inputs rather than from a tile somebody else is still writing. It
	// writes this thread's own slot and no other, which is what keeps it free
	// of races; the barrier is what makes every later read see all of it.
	for _, v := range decls {
		out = append(out, &Assign{LHS: Lvalue{V: v, Idx: &Ref{V: g.tid}}, Op: token.ASSIGN, RHS: g.expr(v.Kind, 2)})
	}
	out = append(out, &Barrier{})
	g.tiles = append(g.tiles, decls...)
	return out
}

// stores writes the thread's own slot of each output buffer. A kernel that
// wrote nothing would make the whole comparison vacuous, so at least the first
// output is always written.
func (g *gen) stores() []Stmt {
	var out []Stmt
	for i, b := range g.outBufs {
		if i > 0 && g.r.IntN(2) == 0 {
			continue
		}
		lhs := Lvalue{V: b, Idx: &Ref{V: g.gid}}
		op := token.ASSIGN
		if b.Kind != KBool && g.r.IntN(3) == 0 {
			op = pick(g.r, []token.Token{token.ADD_ASSIGN, token.SUB_ASSIGN, token.MUL_ASSIGN})
		}
		out = append(out, &Assign{LHS: lhs, Op: op, RHS: g.expr(b.Kind, 3)})
	}
	// And this thread's own element of the struct output, one writable field of
	// it. A plain store only: a compound assignment would read and write
	// through the same pair of accessors and be testing those rather than the
	// translation.
	if g.structOut != nil {
		var writable []int
		for i, f := range g.structOut.Shape.Fields {
			if f.writable() {
				writable = append(writable, i)
			}
		}
		f := pick(g.r, writable)
		k := g.structOut.Shape.Fields[f].Kind
		out = append(out, &Assign{
			LHS: Lvalue{V: g.structOut, Idx: &Ref{V: g.gid}, F: f},
			Op:  token.ASSIGN,
			RHS: g.expr(k, 3),
		})

	}
	if g.atomBuf != nil {
		out = append(out, g.atomic())
	}
	return out
}

// atomic is one atomic, as a statement whose result is dropped.
//
// Only the order-independent ones are generated, and the float add only with
// an addend of one: float32 addition is not associative, and the order the
// emulator's goroutines take its lock is not the order the device's threads
// arrive in, so any other addend would make the two backends legitimately
// disagree. Every addend being one keeps each partial sum an exact integer
// below 2^24, which is the same argument the repository's own atomic parity
// tests rest on.
func (g *gen) atomic() Stmt {
	b := g.atomBuf
	idx := g.wrapIndex(g.expr(KInt, 1), b)
	if b.Kind == KF32 {
		return &Discard{X: &AtomicCall{Fn: g.atomFn, Buf: b, Idx: idx, Args: []Expr{&Lit{K: KF32, F: 1}}, K: KF32}}
	}
	return &Discard{X: &AtomicCall{Fn: g.atomFn, Buf: b, Idx: idx, Args: []Expr{g.expr(KI32, 1)}, K: KI32}}
}

// --- helpers ---------------------------------------------------------------

func (g *gen) genHelpers() {
	n := g.r.IntN(4)
	for i := range n {
		fn := &Func{Name: fmt.Sprintf("h%d", i)}
		if g.sync && g.r.IntN(3) == 0 {
			// A device function taking a gpu.Ctx is a kernel unless it says
			// otherwise, which is what the marker is for.
			fn.Ctx, fn.Device = true, true
		}
		fn.Result = pick(g.r, g.valueKinds())
		g.genHelperBody(fn)
		g.helpers = append(g.helpers, fn)
		if !fn.Ctx {
			g.pureFns = append(g.pureFns, fn)
		}
	}
}

func (g *gen) genHelperBody(fn *Func) {
	prevFn, prevScopes, prevDepth := g.fn, g.scopes, g.depth
	prevIn, prevTiles, prevOut, prevAtom := g.inBufs, g.tiles, g.outBufs, g.atomBuf
	prevSIn, prevSOut := g.structIn, g.structOut
	prevUniform, prevBudget, prevPure := g.uniform, g.budget, g.pureFns
	prevNoShadow := g.noShadow
	g.fn, g.scopes, g.depth, g.noShadow = fn, nil, map[string]int{}, map[string]bool{}
	// A helper reaches none of the kernel's buffers except through its own
	// parameters, and it never writes one: a buffer passed twice to a helper
	// that writes is refused at lowering, and passing one that the launch
	// aliased would be unsound even where lowering accepts it.
	g.outBufs, g.atomBuf, g.tiles = nil, nil, nil
	g.structIn, g.structOut = nil, nil
	g.budget = 3 + g.r.IntN(4)
	// A helper without a gpu.Ctx holds no barrier and can say nothing
	// block-uniform in the first place: its scalar parameters come from its
	// caller and the analysis assumes they vary, so there is no uniform value
	// for a condition to be built from.
	g.uniform = fn.Ctx
	g.pureFns = nil
	for _, h := range g.helpers {
		if !h.Ctx {
			g.pureFns = append(g.pureFns, h)
		}
	}

	g.push()
	var ins []*Var
	// The first parameter is an int, which guarantees every helper has one
	// non-constant value to build an expression around; see anchorInt.
	kinds := []Kind{KInt}
	for range g.r.IntN(2) {
		kinds = append(kinds, pick(g.r, g.valueKinds()))
	}
	for _, kind := range kinds {
		v := &Var{Name: g.paramName(), Kind: kind, Vary: true}
		fn.Params = append(fn.Params, v)
		fn.frame.alloc(v)
		g.declare(v)
	}
	if g.r.IntN(2) == 0 {
		v := &Var{Name: g.paramName(), Kind: pick(g.r, g.elemKinds()), Slice: true, Len: 1}
		fn.Params = append(fn.Params, v)
		fn.frame.alloc(v)
		g.declare(v)
		ins = append(ins, v)
	}
	g.inBufs = ins

	var body []Stmt
	if fn.Ctx {
		// A barrier at the top of the helper, which is a barrier at every call
		// site: that the analysis follows calls is the thing worth generating.
		if tile := g.helperTile(fn); tile != nil {
			body = append(body, tile...)
		} else {
			body = append(body, &Barrier{})
		}
	}
	body = append(body, g.stmts(1+g.r.IntN(3))...)
	body = append(body, &Return{X: g.expr(fn.Result, 3)})
	fn.Body = body
	fixUnused(fn)
	g.pop()

	g.fn, g.scopes, g.depth = prevFn, prevScopes, prevDepth
	g.inBufs, g.tiles, g.outBufs, g.atomBuf = prevIn, prevTiles, prevOut, prevAtom
	g.structIn, g.structOut = prevSIn, prevSOut
	g.uniform, g.budget, g.pureFns = prevUniform, prevBudget, prevPure
	g.noShadow = prevNoShadow
}

// helperTile gives a ctx-taking helper a static shared tile.
//
// A device function may declare one -- that is block-scoped storage CUDA
// allocates once per function -- and may not declare a dynamic one, whose
// length is a launch parameter. The fill is at this thread's own slot and is
// closed by the same barrier, for the same reason the kernel's is.
func (g *gen) helperTile(fn *Func) []Stmt {
	if g.r.IntN(2) != 0 {
		return nil
	}
	kind := pick(g.r, []Kind{KF32, KI32, KU32})
	v := &Var{Name: g.paramName(), Kind: kind, Vary: false}
	v.Slice = true
	v.Len = g.p.Threads() + g.r.IntN(3)
	fn.frame.alloc(v)
	g.declare(v)
	tid := &Pos{Method: "ThreadIdx"}
	fill := &Assign{LHS: Lvalue{V: v, Idx: g.wrapIndex(tid, v)}, Op: token.ASSIGN, RHS: g.expr(kind, 2)}
	out := []Stmt{&SharedDecl{V: v, N: v.Len}, fill, &Barrier{}}
	g.tiles = append(g.tiles, v)
	return out
}

// --- names and scopes ------------------------------------------------------

// namePool are ordinary identifiers.
var namePool = []string{
	"a", "b", "c", "d", "e", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
	"acc", "sum", "tmp", "idx", "lo", "hi", "off", "cnt", "val", "buf", "src", "dst", "step",
}

// keywordPool are valid Go identifiers that are C++ keywords, which the
// emitter escapes with a trailing underscore.
//
// Three of the emitter's keywords are absent. const is one of Go's keywords
// too and is not an identifier at all; int and bool are Go's own type names,
// and a local called int would shadow the type the very next declaration
// needs. Nothing here is ever spelled with a trailing underscore, so a local
// can never collide with the escaped form of another name in the same
// function -- which is a refusal rather than a mistranslation, and so belongs
// in errors_test.go rather than here.
var keywordPool = []string{
	"auto", "class", "delete", "double", "extern", "float", "friend", "inline",
	"long", "new", "operator", "private", "protected", "public", "register", "short",
	"signed", "sizeof", "static", "template", "this", "throw", "try", "typedef", "union",
	"unsigned", "virtual", "void", "volatile", "namespace", "using", "char",
}

// reserved are the names the generated source already means something by.
var reserved = map[string]bool{
	"ctx": true, "gpu": true, "K": true, "len": true, "min": true, "max": true,
	// The emitter reserves this one for the temporary an if/else switch
	// evaluates its tag into.
	"switch_tag": true,
}

// paramName draws a name for a parameter or a buffer, and closes it to
// shadowing for the rest of the function.
func (g *gen) paramName() string {
	n := g.freshName(false)
	if g.noShadow == nil {
		g.noShadow = map[string]bool{}
	}
	g.noShadow[n] = true
	return n
}

// freshName draws an identifier that is free in the current scope.
//
// shadow allows one already declared further out, which is legal in Go and in
// C++ alike and is worth generating: a shadowed name is one of the ways the
// emitter's own generated names have collided with the author's.
func (g *gen) freshName(shadow bool) string {
	pool := namePool
	if g.r.IntN(5) == 0 {
		pool = keywordPool
	}
	for range 24 {
		n := pool[g.r.IntN(len(pool))]
		if reserved[n] {
			continue
		}
		if g.noShadow[n] {
			continue
		}
		d, taken := g.depth[n]
		if !taken || (shadow && d < len(g.scopes)-1) {
			return n
		}
	}
	g.nameN++
	return fmt.Sprintf("v%d", g.nameN)
}

func (g *gen) push() { g.scopes = append(g.scopes, nil) }

func (g *gen) pop() {
	last := g.scopes[len(g.scopes)-1]
	g.scopes = g.scopes[:len(g.scopes)-1]
	for _, v := range last {
		delete(g.depth, v.Name)
	}
	// A name shadowed by this scope is visible again; recompute what is left.
	for d, sc := range g.scopes {
		for _, v := range sc {
			g.depth[v.Name] = d
		}
	}
}

func (g *gen) declare(v *Var) {
	g.scopes[len(g.scopes)-1] = append(g.scopes[len(g.scopes)-1], v)
	if g.depth == nil {
		g.depth = map[string]int{}
	}
	g.depth[v.Name] = len(g.scopes) - 1
}

// local makes an undeclared scalar variable with a slot of its own. uniform
// says it may only ever hold a block-uniform value, which is a property the
// generator fixes here rather than inferring later.
func (g *gen) local(k Kind, uniform bool) *Var {
	v := &Var{Name: g.freshName(g.r.IntN(4) == 0), Kind: k, Vary: !uniform}
	g.fn.frame.alloc(v)
	return v
}

// visible walks the scopes from the innermost out, so a shadowing declaration
// is what a name resolves to -- which is what the Go source says too.
func (g *gen) visible(keep func(*Var) bool) []*Var {
	seen := map[string]bool{}
	var out []*Var
	for i := len(g.scopes) - 1; i >= 0; i-- {
		for j := len(g.scopes[i]) - 1; j >= 0; j-- {
			v := g.scopes[i][j]
			if seen[v.Name] {
				continue
			}
			seen[v.Name] = true
			if keep(v) {
				out = append(out, v)
			}
		}
	}
	return out
}

func pick[T any](r *rand.Rand, xs []T) T { return xs[r.IntN(len(xs))] }
