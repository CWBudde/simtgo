package fuzz_test

import (
	"fmt"
	"strings"
	"testing"
	"testing/fstest"

	"github.com/CWBudde/simtgo/internal/fuzz"
	"github.com/CWBudde/simtgo/simt"
)

// TestShapesLowerWithTheirPadding is what the struct catalogue is for.
//
// The emitter turns every hole in a Go struct into a simtgo_padN member and
// asserts the whole struct's size and alignment, because NVRTC has no offsetof
// to check the field positions with -- so the size assertion is what stands in
// for them, and it is only as good as the padding it is computed from. A shape
// with no hole would exercise none of that, which is why the catalogue is
// chosen for its holes rather than for variety.
//
// Each case names the hole it is there for. If one of these stops appearing,
// either Go's layout rules have changed or the emitter has stopped seeing
// them, and both are worth a failing test.
func TestShapesLowerWithTheirPadding(t *testing.T) {
	want := map[string]struct {
		pad  string // the padding member the hole produces
		size int
		why  string
	}{
		"SPair": {pad: "", size: 8, why: "two four-byte fields, so no hole at all: the control"},
		"SHole": {pad: "simtgo_pad0[3]", size: 8, why: "three bytes after an int8, the commonest hole"},
		"SWide": {pad: "simtgo_pad0[4]", size: 16, why: "four bytes to reach the int64's alignment of eight"},
		"STail": {pad: "simtgo_pad0[6]", size: 16, why: "six bytes at the end, which move no field and change sizeof"},
	}

	// Both directions, because the loop below only checks the shapes that are
	// there: a catalogue that lost STail would run three passing subtests and
	// say nothing about the trailing hole nothing else can see.
	have := make(map[string]bool, len(fuzz.Shapes))
	for _, shape := range fuzz.Shapes {
		have[shape.Name] = true
	}
	for name, w := range want {
		if !have[name] {
			t.Errorf("the catalogue no longer has %s (%s)", name, w.why)
		}
	}

	for _, shape := range fuzz.Shapes {
		t.Run(shape.Name, func(t *testing.T) {
			w, ok := want[shape.Name]
			if !ok {
				t.Fatalf("the catalogue has a shape this test does not know about: %s", shape.Name)
			}
			src := shapeKernel(shape)
			u, err := simt.Transpile(fstest.MapFS{"k.go": &fstest.MapFile{Data: []byte(src)}}, "K")
			if err != nil {
				t.Fatalf("Transpile: %v\n%s", err, src)
			}
			sizeAssert := fmt.Sprintf("static_assert(sizeof(%s) == %d", shape.Name, w.size)
			if !strings.Contains(u.Source, sizeAssert) {
				t.Errorf("no %q (%s):\n%s", sizeAssert, w.why, u.Source)
			}
			switch {
			case w.pad == "" && strings.Contains(u.Source, "simtgo_pad"):
				t.Errorf("%s has no hole (%s) but the emitter padded it:\n%s", shape.Name, w.why, u.Source)
			case w.pad != "" && !strings.Contains(u.Source, w.pad):
				t.Errorf("no %q (%s):\n%s", w.pad, w.why, u.Source)
			}
		})
	}
}

// shapeKernel is a kernel that declares the shape and reads its first field,
// which is the least a kernel can do and still make the emitter lay the struct
// out.
func shapeKernel(shape *fuzz.StructShape) string {
	var b strings.Builder
	b.WriteString("package kernels\n\nimport \"github.com/CWBudde/simtgo/gpu\"\n\n")
	b.WriteString("type " + shape.Name + " struct {\n")
	for _, f := range shape.Fields {
		fmt.Fprintf(&b, "\t%s %s\n", f.Name, f.Kind.GoName())
	}
	b.WriteString("}\n\n")
	// The first field is read through a conversion, which is the only way a
	// narrow one can be read at all -- and two of these shapes start with one.
	fmt.Fprintf(&b, "func K(ctx gpu.Ctx, y []int64, p []%s) {\n", shape.Name)
	fmt.Fprintf(&b, "\ti := ctx.GlobalID()\n\tif i < len(p) {\n\t\ty[0] = int64(p[i].%s)\n\t}\n}\n", shape.Fields[0].Name)
	return b.String()
}
