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
		name: "float64",
		body: "func K(ctx gpu.Ctx, a []float64) { a[0] = 1 }",
		want: "float32/int32 only",
	}, {
		name: "non-constant shared memory",
		body: "func K(ctx gpu.Ctx, a []float32) { s := ctx.SharedF32(len(a)); s[0] = 1 }",
		want: "constant size",
	}, {
		name: "goroutine",
		body: "func K(ctx gpu.Ctx, a []float32) { go func() { a[0] = 1 }() }",
		want: "unsupported statement",
	}, {
		name: "calling a Go function",
		body: "func helper(x float32) float32 { return x }\n\nfunc K(ctx gpu.Ctx, a []float32) { a[0] = helper(a[1]) }",
		want: "device functions are not implemented",
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
