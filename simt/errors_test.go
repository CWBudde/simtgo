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
		name: "int8",
		body: "func K(ctx gpu.Ctx, a []int8) { a[0] = 1 }",
		want: "C promotes it to int, so the two would disagree",
	}, {
		name: "uint16",
		body: "func K(ctx gpu.Ctx, a []uint16) { a[0] = 1 }",
		want: "C promotes it to int, so the two would disagree",
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
		name: "shared memory inside a device function",
		body: "//gocuda:ignore\nfunc stage(ctx gpu.Ctx) float32 { s := ctx.SharedF32(4)\nreturn s[0] }\n\n" +
			"func K(ctx gpu.Ctx, a []float32) { a[0] = stage(ctx) }",
		want: "shared memory may only be declared in a kernel",
	}, {
		// Without the opt-out the helper is a kernel in its own right, and
		// calling a kernel is what the previous case refuses.
		name: "a gpu.Ctx helper that did not opt out of being a kernel",
		body: "func where(ctx gpu.Ctx) int { return ctx.GlobalID() }\n\n" +
			"func K(ctx gpu.Ctx, a []float32) { a[0] = float32(where(ctx)) }",
		want: "where is a kernel",
	}, {
		name: "multiple assignment",
		body: "func K(ctx gpu.Ctx, a []float32) { i, j := 0, 1; a[i] = a[j] }",
		want: "multiple assignment",
	}, {
		name: "map",
		body: "func K(ctx gpu.Ctx, a []float32) { m := map[int]int{}; a[0] = float32(m[1]) }",
		want: "unsupported type",
	}, {
		// A slice parameter lowers to a pointer plus a generated length, so a
		// parameter spelled like that length would reach NVRTC as a duplicate.
		name: "parameter collides with a generated length",
		body: "func K(ctx gpu.Ctx, x []float32, x_len int32) { x[0] = float32(x_len) }",
		want: "collides with the length generated for slice parameter x",
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
