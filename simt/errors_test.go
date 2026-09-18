package simt_test

import (
	"strings"
	"testing"
	"testing/fstest"

	"github.com/CWBudde/gocuda/simt"
)

// The transpiler covers a subset of Go. These tests pin the edge of that
// subset: every construct outside it must be refused with a message that says
// where and why, never mistranslated.
func TestUnsupported(t *testing.T) {
	const prelude = "package kernels\n\nimport \"github.com/CWBudde/gocuda/gpu\"\n\n"
	cases := []struct {
		name, body, want string
	}{{
		name: "no context parameter",
		body: "func K(a []float32) { a[0] = gpu.Sqrt(a[1]) }",
		want: "first parameter must be gpu.Ctx",
	}, {
		name: "returns a value",
		body: "func K(ctx gpu.Ctx, a []float32) float32 { return a[0] }",
		want: "must not return values",
	}, {
		name: "float64 without the directive",
		body: "func K(ctx gpu.Ctx, a []float64) { a[0] = 1 }",
		want: "float64 needs //gocuda:float64 on kernel K",
	}, {
		name: "//gocuda:float64 on a device function",
		body: "//gocuda:float64\nfunc half(x float32) float32 { return x / 2 }\n\nfunc K(ctx gpu.Ctx, a []float32) { a[0] = half(a[1]) }",
		want: "belongs on the kernel, not on device function half",
	}, {
		// The four narrow integers are storage and nothing else. They cross as
		// slice elements, array elements and struct fields, where the widths
		// agree and no arithmetic happens; every position that exists in order
		// to be computed with is refused, and so is every operator. What the
		// two languages disagree about is an operator whose result feeds
		// another one -- `a, b := int8(100), int8(3); a*b/2` is 22 in Go and
		// -106 in C -- and the rest of these are refused with it rather than
		// listed as exceptions somebody would have to trust.
		name: "a narrow local",
		body: "func K(ctx gpu.Ctx, y []uint8) { v := y[0]; y[1] = v }",
		want: "not a variable, a parameter or a result",
	}, {
		name: "a narrow by-value parameter",
		body: "func K(ctx gpu.Ctx, y []uint8, k uint8) { y[0] = k }",
		want: "not a variable, a parameter or a result",
	}, {
		name: "a narrow result",
		body: "func clip(x int32) uint8 { return uint8(x) }\n\n" +
			"func K(ctx gpu.Ctx, y []uint8) { y[0] = clip(1) }",
		want: "not a variable, a parameter or a result",
	}, {
		// `for i, v := range narrow` declares v, so it is the local rule
		// reached by another spelling. `for i := range narrow` is fine and
		// TestNarrowStorage pins that.
		name: "a range value over a narrow slice",
		body: "func K(ctx gpu.Ctx, y []uint8) { for i, v := range y { y[i] = v } }",
		want: "not a variable, a parameter or a result",
	}, {
		name: "arithmetic on narrow elements",
		body: "func K(ctx gpu.Ctx, y, a, b []int8) { y[0] = a[0] * b[0] }",
		want: "so `*` on one can give a different answer",
	}, {
		name: "unary minus on a narrow element",
		body: "func K(ctx gpu.Ctx, y, a []int8) { y[0] = -a[0] }",
		want: "so `-` on one can give a different answer",
	}, {
		// The two languages do agree about this one. It goes anyway, because
		// the rule is every operator rather than a list of the safe ones, and
		// because relaxing a refusal later costs a line while retracting an
		// acceptance costs somebody a kernel that worked.
		name: "comparing narrow elements",
		body: "func K(ctx gpu.Ctx, y []float32, a, b []uint16) { if a[0] < b[0] { y[0] = 1 } }",
		want: "so `<` on one can give a different answer",
	}, {
		name: "compound assignment into a narrow slot",
		body: "func K(ctx gpu.Ctx, y []uint8) { y[0] += 1 }",
		want: "`+=` is refused anyway",
	}, {
		name: "incrementing a narrow slot",
		body: "func K(ctx gpu.Ctx, y []uint8) { y[0]++ }",
		want: "`++` is refused anyway",
	}, {
		// CUDA's min and max are overloaded for int and float, not for a
		// narrow type, so this would resolve to the int overload and hand back
		// an int -- an answer about generated code either way.
		name: "min on narrow elements",
		body: "func K(ctx gpu.Ctx, y, a, b []uint8) { y[0] = min(a[0], b[0]) }",
		want: "min has no overload for it",
	}, {
		// A switch is not an operator, and the constant form lowers (see
		// TestNarrowSwitch); this one becomes an if/else chain against a
		// declared tag, and the tag is a local, which is the rule that stops
		// it.
		name: "a non-constant switch on a narrow value",
		body: "func K(ctx gpu.Ctx, y []float32, a, b []uint8) { switch a[0] {\ncase b[0]:\n\ty[0] = 1\n} }",
		want: "not a variable, a parameter or a result",
	}, {
		name: "a shift by a narrow count",
		body: "func K(ctx gpu.Ctx, y []int32, n []uint8) { y[0] = y[1] << n[0] }",
		want: "so `<<` on one can give a different answer",
	}, {
		name: "uint",
		body: "func K(ctx gpu.Ctx, n uint, a []float32) { a[0] = float32(n) }",
		want: "use uint32 or uint64",
	}, {
		// The one that was lowering cleanly and returning wrong numbers: Go's
		// int is 8 bytes, the emitted C int is 4, and cuda.Upload copies the
		// Go layout, so the kernel strode half the buffer.
		name: "[]int",
		body: "func K(ctx gpu.Ctx, a []int) { a[0] = 1 }",
		want: "cannot cross to the device",
	}, {
		name: "an array parameter",
		body: "func K(ctx gpu.Ctx, y []float32, taps [4]float32) { y[0] = taps[0] }",
		want: "Go passes an array by value and C would pass a pointer to it",
	}, {
		name: "an array result",
		body: "//gocuda:ignore\nfunc pair() [2]float32 { var a [2]float32; return a }\n\nfunc K(ctx gpu.Ctx, y []float32) { y[0] = pair()[0] }",
		want: "C cannot return an array",
	}, {
		name: "whole-array assignment",
		body: "func K(ctx gpu.Ctx, y []float32) { var a, b [2]float32; a = b; y[0] = a[0] }",
		want: "cannot be assigned",
	}, {
		// The same move spelled as a declaration. `:=` and `=` both refused it;
		// an explicit `var` with an initialiser went straight through and
		// emitted `float a[2] = b;`, which C++ will not initialise either.
		name: "whole-array var initialiser",
		body: "func K(ctx gpu.Ctx, y []float32) { var b [2]float32; var a [2]float32 = b; y[0] = a[0] }",
		want: "cannot be assigned",
	}, {
		name: "struct equality",
		body: "type P struct{ X, Y float32 }\n\nfunc K(ctx gpu.Ctx, y []float32, a, b []P) { if a[0] == b[0] { y[0] = 1 } }",
		want: "field by field in Go, which C cannot do",
	}, {
		name: "an int field in a struct",
		body: "type P struct{ N int }\n\nfunc K(ctx gpu.Ctx, y []float32, ps []P) { y[0] = float32(ps[0].N) }",
		want: "cannot cross to the device",
	}, {
		// An array field lowers now; what cannot is a field spelled like the
		// padding the emitter adds to pin the offsets, because two members
		// with one name is an NVRTC error about code nobody wrote.
		name: "a field named like the emitted padding",
		body: "type P struct{ gocuda_pad0 int32 }\n\nfunc K(ctx gpu.Ctx, y []float32, ps []P) { y[0] = float32(ps[0].gocuda_pad0) }",
		want: "spelled like the padding gocuda emits",
	}, {
		name: "an embedded field",
		body: "type Inner struct{ X float32 }\ntype Outer struct{ Inner }\n\nfunc K(ctx gpu.Ctx, y []float32, os []Outer) { y[0] = os[0].X }",
		want: "embedded field",
	}, {
		// Two blank fields are one Go name and two C++ ones: both were emitted
		// as `int _;` and NVRTC refused the redeclaration.
		name: "a blank struct field",
		body: "type P struct {\n\t_, _ int32\n\tX float32\n}\n\nfunc K(ctx gpu.Ctx, y []float32, ps []P) { y[0] = ps[0].X }",
		want: "has a blank field",
	}, {
		name: "an anonymous struct type",
		body: "func K(ctx gpu.Ctx, y []float32) { p := struct{ X float32 }{1}; y[0] = p.X }",
		want: "unsupported type struct{X float32} on the device",
	}, {
		name: "a method call",
		body: "type P struct{ X float32 }\n\nfunc (p P) Twice() float32 { return p.X * 2 }\n\nfunc K(ctx gpu.Ctx, y []float32, ps []P) { y[0] = ps[0].Twice() }",
		want: "methods are not supported in kernels",
	}, {
		name: "a switch over a struct",
		body: "type P struct{ X, Y float32 }\n\nfunc K(ctx gpu.Ctx, y []float32, ps []P) {\n\tswitch ps[0] {\n\tcase P{X: 1}:\n\t\ty[0] = 1\n\t}\n}",
		want: "cannot switch on kernels.P",
	}, {
		name: "a zero-length array",
		body: "func K(ctx gpu.Ctx, y []float32) { var a [0]float32; y[0] = float32(len(a)) }",
		want: "has no elements",
	}, {
		name: "int(x) from int64",
		body: "func K(ctx gpu.Ctx, a []float32, n int64) { a[int(n)] = 1 }",
		want: "truncates on the device",
	}, {
		name: "non-constant shared memory",
		body: "func K(ctx gpu.Ctx, a []float32) { s := ctx.SharedF32(len(a)); s[0] = 1 }",
		want: "constant size",
	}, {
		name: "goroutine",
		body: "func K(ctx gpu.Ctx, a []float32) { go func() { a[0] = 1 }() }",
		want: "unsupported statement",
	}, {
		// Device functions are emitted, but not ones that call themselves:
		// there is no stack depth on the device to spend on it.
		name: "a recursive device function",
		body: "func down(x float32) float32 { if x > 0 { return down(x - 1) }\nreturn x }\n\n" +
			"func K(ctx gpu.Ctx, a []float32) { a[0] = down(a[1]) }",
		want: "down calls itself",
	}, {
		name: "mutually recursive device functions",
		body: "func even(x float32) float32 { return odd(x) }\n\nfunc odd(x float32) float32 { return even(x) }\n\n" +
			"func K(ctx gpu.Ctx, a []float32) { a[0] = even(a[1]) }",
		want: "even, which is already being lowered",
	}, {
		name: "calling another kernel",
		body: "func Other(ctx gpu.Ctx, a []float32) { a[0] = 1 }\n\nfunc K(ctx gpu.Ctx, a []float32) { Other(ctx, a) }",
		want: "Other is a kernel",
	}, {
		name: "a device function returning two values",
		body: "func two(x float32) (float32, float32) { return x, x }\n\nfunc K(ctx gpu.Ctx, a []float32) { two(a[0]); a[0] = 1 }",
		want: "must return at most one value",
	}, {
		// (a, b float32) is one result field holding two values, so counting
		// fields rather than the checked signature would let it through.
		name: "a device function returning two named values",
		body: "func two(x float32) (a, b float32) { a = x\nb = x\nreturn }\n\nfunc K(ctx gpu.Ctx, a []float32) { two(a[0]); a[0] = 1 }",
		want: "must return at most one value",
	}, {
		// A named result is a local the body assigns to and a bare return
		// that carries it; neither is emitted, so it is refused instead.
		name: "a device function naming its result",
		body: "func one(x float32) (r float32) { r = x\nreturn }\n\nfunc K(ctx gpu.Ctx, a []float32) { a[0] = one(a[1]) }",
		want: "must not name its result",
	}, {
		name: "a variadic device function",
		body: "func any(xs ...float32) float32 { return xs[0] }\n\nfunc K(ctx gpu.Ctx, a []float32) { a[0] = any(a[1]) }",
		want: "must not be variadic",
	}, {
		// A statically sized tile in a device function is accepted -- it is
		// block-scoped storage, which a __device__ function may declare -- but
		// the dynamic one cannot be, because its length is a parameter of the
		// kernel and a helper has no way to reach it.
		name: "a dynamically sized shared tile inside a device function",
		body: "//gocuda:ignore\nfunc stage(ctx gpu.Ctx) float32 { s := ctx.SharedDynF32()\nreturn s[0] }\n\n" +
			"func K(ctx gpu.Ctx, a []float32) { a[0] = stage(ctx) }",
		want: "may only be declared in a kernel: its length is a launch parameter",
	}, {
		name: "AssumeBlockDim inside a device function",
		body: "//gocuda:ignore\nfunc stage(ctx gpu.Ctx) float32 { ctx.AssumeBlockDim(256)\nreturn 1 }\n\n" +
			"func K(ctx gpu.Ctx, a []float32) { a[0] = stage(ctx) }",
		want: "AssumeBlockDim may only be called in a kernel",
	}, {
		// CUDA has one dynamic __shared__ block per launch, and NVRTC accepts
		// a second `extern __shared__` declaration without a word -- of a
		// different element type as readily as of the same one. The two names
		// would then be two views of one buffer.
		name: "two dynamically sized shared tiles",
		body: "func K(ctx gpu.Ctx, y []float32) { a := ctx.SharedDynF32(); b := ctx.SharedDynI32(); y[0] = a[0] + float32(b[0]) }",
		want: "at most one dynamically sized shared tile, and a is already one",
	}, {
		name: "a float64 tile without the directive",
		body: "func K(ctx gpu.Ctx, y []float32) { s := ctx.SharedF64(4); y[0] = float32(s[0]) }",
		want: "float64 needs //gocuda:float64 on kernel K",
	}, {
		name: "a dynamically sized float64 tile without the directive",
		body: "func K(ctx gpu.Ctx, y []float32) { s := ctx.SharedDynF64(); y[0] = float32(s[0]) }",
		want: "float64 needs //gocuda:float64 on kernel K",
	}, {
		// The refusal names the launch-sized constructor of the same element
		// type, because that is what the author wanted and could not spell.
		name: "a runtime size on a typed tile",
		body: "func K(ctx gpu.Ctx, a []int32) { s := ctx.SharedI32(len(a)); s[0] = 1 }",
		want: "use SharedDynI32, whose length the launch gives",
	}, {
		// A dynamic tile generates an `int s_len` parameter exactly as a slice
		// parameter does, so it can collide with either kind of name.
		name: "a dynamic tile's length collides with a parameter",
		body: "func K(ctx gpu.Ctx, y []float32, s_len int32) { s := ctx.SharedDynF32(); y[0] = s[0] + float32(s_len) }",
		want: "collides with parameter s_len",
	}, {
		// The other half of that check, and the only way to reach it: a tile
		// cannot be named after a parameter, since both are in the kernel's
		// scope, so its generated length can only meet a slice's through a
		// C++ keyword. The parameter "float" is declared as "float_", whose
		// generated length is "float__len" -- which is also what a tile called
		// "float_" would generate.
		name: "a dynamic tile's length collides with a slice's generated length",
		body: "func K(ctx gpu.Ctx, float []float32) { float_ := ctx.SharedDynF32(); float[0] = float_[0] }",
		want: "collides with the length generated for slice parameter float",
	}, {
		// The top level of the enclosing function, now that a device function
		// may have a tile too: a tile declared inside a conditional is one the
		// threads of a block could disagree about.
		name: "a shared tile inside a conditional",
		body: "func K(ctx gpu.Ctx, y []float32) { if ctx.ThreadIdx() == 0 { s := ctx.SharedF32(4); y[0] = s[0] } }",
		want: "top level of the function",
	}, {
		// Without a marker the helper is a kernel in its own right, and
		// calling a kernel is what the previous case refuses.
		name: "a gpu.Ctx helper that did not say it was a device function",
		body: "func where(ctx gpu.Ctx) int { return ctx.GlobalID() }\n\n" +
			"func K(ctx gpu.Ctx, a []float32) { a[0] = float32(where(ctx)) }",
		want: "where is a kernel",
	}, {
		// And the refusal points at the marker that says what the helper is,
		// not at the one that says what it is not.
		name: "the refusal names the device marker",
		body: "func where(ctx gpu.Ctx) int { return ctx.GlobalID() }\n\n" +
			"func K(ctx gpu.Ctx, a []float32) { a[0] = float32(where(ctx)) }",
		want: "Mark it //gocuda:device",
	}, {
		// The marker only means anything where the signature rule would
		// otherwise make a kernel. Anywhere else it is decoration, and a
		// marker that is decoration half the time is read as decoration.
		name: "//gocuda:device on a function that takes no gpu.Ctx",
		body: "//gocuda:device\nfunc half(x float32) float32 { return x / 2 }\n\n" +
			"func K(ctx gpu.Ctx, a []float32) { a[0] = half(a[1]) }",
		want: "//gocuda:device does nothing on half, which takes no gpu.Ctx",
	}, {
		// Refused even though nothing reaches it: the mistake is easiest to
		// make on a function no kernel calls, so a check that only ran along
		// the lowering paths would pass over exactly that case.
		name: "//gocuda:device on a function nothing calls",
		body: "//gocuda:device\nfunc unused(x float32) float32 { return x }\n\n" +
			"func K(ctx gpu.Ctx, a []float32) { a[0] = 1 }",
		want: "//gocuda:device does nothing on unused",
	}, {
		// A static tile in a helper is fine -- it is block-scoped storage the
		// compiler allocates once per function, and its bytes are accounted
		// onto the kernel that reaches it. The dynamic tile is the one that
		// cannot go there, because its length arrives as a parameter of the
		// kernel, which a helper has no way to be handed. The marker changes
		// neither answer: what a helper lacks is a launch, not a name.
		name: "a dynamic shared tile inside a marked device function",
		body: "//gocuda:device\nfunc stage(ctx gpu.Ctx) float32 { s := ctx.SharedDynF32()\nreturn s[0] }\n\n" +
			"func K(ctx gpu.Ctx, a []float32) { a[0] = stage(ctx) }",
		want: "device function",
	}, {
		// The refusal that lives in ctype cannot see this one: no float64 is
		// written down anywhere. The argument is an untyped constant and the
		// result is converted away, so without a check on the call itself the
		// kernel would have run a double it never named.
		name: "a float64 helper whose type never surfaces",
		body: "func K(ctx gpu.Ctx, y []float32) { y[0] = float32(gpu.Sqrt64(2)) }",
		want: "gpu.Sqrt64 is double precision and needs //gocuda:float64 on kernel K",
	}, {
		name: "a float64 helper on a float64 kernel without the directive",
		body: "func K(ctx gpu.Ctx, y []float64) { y[0] = gpu.Hypot64(y[1], y[2]) }",
		want: "needs //gocuda:float64 on kernel K",
	}, {
		// An atomic names a buffer and an index rather than a pointer, so the
		// emitter is what writes the "&". A slice expression has no address to
		// take, and saying so is more use than letting NVRTC complain about
		// code the author never wrote.
		name: "an atomic on a slice expression rather than a buffer",
		body: "func K(ctx gpu.Ctx, h []int32) { gpu.AtomicAddI32(h[0:2], 0, 1) }",
		want: "needs the buffer itself as its first argument",
	}, {
		name: "an atomic on a shared buffer built inline",
		body: "func K(ctx gpu.Ctx, y []float32) { gpu.AtomicAddF32(ctx.SharedF32(4), 0, 1); y[0] = 1 }",
		want: "needs the buffer itself as its first argument",
	}, {
		name: "an atomic on a package-level buffer",
		body: "var g []int32\n\nfunc K(ctx gpu.Ctx, y []float32) { gpu.AtomicAddI32(g, 0, 1); y[0] = 1 }",
		want: "declared outside the kernel",
	}, {
		// The half of multiple assignment that is an ABI question rather than
		// a statement one: C returns one value, and a pair would have to come
		// back through out-parameters the subset cannot spell.
		name: "assigning from a two-valued call",
		body: "func two(x float32) (float32, float32) { return x, x }\n\n" +
			"func K(ctx gpu.Ctx, a []float32) { p, q := two(a[0]); a[0] = p + q }",
		want: "assigning 2 values from one expression is not supported",
	}, {
		// A for clause is one C expression and the temporaries the two phases
		// need are declarations, so the parallel form is refused there and
		// supported everywhere else.
		name: "a parallel assignment in a for clause",
		body: "func K(ctx gpu.Ctx, a []float32, n int32) { for i, j := 0, int(n); i < j; i, j = i+1, j-1 { a[i] = a[j] } }",
		want: "multiple assignment is not supported in a for clause",
	}, {
		name: "the blank identifier in a parallel assignment",
		body: "func K(ctx gpu.Ctx, a []float32) { v, _ := a[0], a[1]; a[0] = v }",
		want: "the blank identifier is not supported in kernels",
	}, {
		// The whole-array refusal has to survive the new path: it is stated
		// against the target's type there, because the left-hand side of an
		// assignment has no recorded expression type to read.
		name: "two whole arrays swapped",
		body: "func K(ctx gpu.Ctx, y []float32) { var p, q [2]float32; p, q = q, p; y[0] = p[0] }",
		want: "cannot be assigned",
	}, {
		// Every pointer in the generated C is __restrict__, and a launch
		// checks that against the buffers it was given. A call inside the
		// kernel has no launch to check it, so it is refused where it is
		// lowered -- one Go call with nothing wrong with it, and two aliased
		// restrict pointers in C.
		name: "one buffer passed as two parameters of a helper that writes",
		body: "func blend(out, a []float32) { out[0] = a[0] * 2 }\n\n" +
			"func K(ctx gpu.Ctx, y []float32) { blend(y, y) }",
		want: "blend is passed the same buffer as both out and a",
	}, {
		name: "map",
		body: "func K(ctx gpu.Ctx, a []float32) { m := map[int]int{}; a[0] = float32(m[1]) }",
		want: "unsupported type",
	}, {
		// A slice parameter lowers to a pointer plus a generated length, so a
		// parameter spelled like that length would reach NVRTC as a duplicate.
		name: "parameter collides with a generated length",
		body: "func K(ctx gpu.Ctx, x []float32, x_len int32) { x[0] = float32(x_len) }",
		want: "x_len is the length generated for slice parameter x",
	}, {
		// The same collision one scope in, which is the worse half: a duplicate
		// parameter is at least an NVRTC error, while a local merely shadows
		// the length, so len(x) quietly reads 3 from here on. It compiled and
		// returned wrong numbers until the check stopped looking only at the
		// other parameters.
		name: "local shadows a generated length",
		body: "func K(ctx gpu.Ctx, x []float32) { x_len := int32(3)\nif ctx.GlobalID() < len(x) { x[0] = float32(x_len) } }",
		want: "x_len is the length generated for slice parameter x",
	}, {
		// cname spells a C++ keyword with a trailing underscore, so a variable
		// already spelled that way becomes the same C identifier as the
		// keyword-named one. Both then read whichever was declared last.
		name: "a variable spelled like an escaped keyword",
		body: "func K(ctx gpu.Ctx, y []float32, int int32) { int_ := int32(99)\ny[0] = float32(int_) + float32(int) }",
		want: "int is a C++ keyword and is emitted as int_",
	}, {
		name: "non-constant block size",
		body: "func K(ctx gpu.Ctx, a []float32) { ctx.AssumeBlockDim(len(a)); a[0] = 1 }",
		want: "constant block size",
	}, {
		name: "two different block sizes",
		body: "func K(ctx gpu.Ctx, a []float32) { ctx.AssumeBlockDim(128); ctx.AssumeBlockDim(256); a[0] = 1 }",
		want: "already declared a block size of 128",
	}, {
		name: "conditional block size",
		body: "func K(ctx gpu.Ctx, a []float32) { if len(a) > 0 { ctx.AssumeBlockDim(128) } }",
		want: "top level of the kernel body",
	}, {
		name: "type switch",
		body: "func K(ctx gpu.Ctx, a []float32) { var x any = 1; switch x.(type) { case int: a[0] = 1 } }",
		want: "type switches are not supported",
	}, {
		// A switch with a non-constant case lowers to an if/else chain, where
		// a C break would leave the enclosing loop instead of the switch.
		name: "break inside a switch that lowered to an if/else chain",
		body: "func K(ctx gpu.Ctx, a []float32, n int32) { for i := 0; i < int(n); i++ { switch { case i > 1: break } }; a[0] = 1 }",
		want: "cannot break out of a switch",
	}, {
		name: "a label on something other than a loop",
		body: "func K(ctx gpu.Ctx, a []float32, n int32) { here: switch n { case 1: break here }; a[0] = 1 }",
		want: "a label may only be placed on a for loop",
	}, {
		name: "range with a value, assigned rather than declared",
		body: "func K(ctx gpu.Ctx, y, x []float32) { var i int; var v float32; for i, v = range x { y[i] = v } }",
		want: "only `for i := range x` and `for i, v := range x` are supported",
	}, {
		// CUDA reads every lane argument as unsigned, so -1 is not the lane
		// below but lane 4294967295, which the built-in then wraps or clamps.
		// The CPU emulator has a definite answer for it -- the caller's own
		// value -- so it is a spelling that means one thing where it is read
		// and another where it runs.
		name: "a negative source lane",
		body: "func K(ctx gpu.Ctx, y []float32) { y[0] = ctx.ShuffleF32(y[1], -1) }",
		want: "takes a lane offset and -1 is negative",
	}, {
		name: "a negative shuffle delta",
		body: "func K(ctx gpu.Ctx, y []float32) { y[0] = ctx.ShuffleDownF32(y[1], -2) }",
		want: "takes a lane offset and -2 is negative",
	}}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fsys := fstest.MapFS{"k.go": &fstest.MapFile{Data: []byte(prelude + tc.body + "\n")}}
			_, err := simt.Transpile(fsys, "K")
			if err == nil {
				t.Fatalf("expected an error, got none")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %q does not mention %q", err, tc.want)
			}
		})
	}
}

// Kernels run on a device with no runtime and no standard library, so imports
// other than package gpu have to be refused outright.
func TestImportRefused(t *testing.T) {
	src := "package kernels\n\nimport (\n\t\"math\"\n\n\t\"github.com/CWBudde/gocuda/gpu\"\n)\n\n" +
		"func K(ctx gpu.Ctx, a []float32) { a[0] = float32(math.Pi) }\n"
	fsys := fstest.MapFS{"k.go": &fstest.MapFile{Data: []byte(src)}}
	_, err := simt.Transpile(fsys, "K")
	if err == nil || !strings.Contains(err.Error(), "may not import") {
		t.Fatalf("got %v, want an import refusal", err)
	}
}
