package fuzz

import (
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// This file runs a Program's Source() for real.
//
// The closure rendering can be executed in this process; the Go text cannot,
// so it is written into a throwaway module that replaces this one by path and
// handed to `go run`. That is the only way the two renderings can be compared
// on anything but a constant: whatever Source() means is what the Go compiler
// says it means, and nothing short of compiling it asks the Go compiler.
//
// Several programs go into one module and one `go run`, because the cost here
// is the compilation rather than the run.

// encode is the canonical form of one run's output buffers.
//
// Floats are printed as their bits. A NaN, a negative zero and a subnormal all
// have to survive the comparison, and %v collapses the first two of those with
// values that are not equal to them.
func encode(p *Program, a *Args) []string {
	var out []string
	for i, spec := range p.Params {
		if !spec.Slice {
			continue
		}
		if spec.Shape != nil {
			out = append(out, spec.Name+" "+encodeStructSlice(spec.Shape, a.Vals[i]))
			continue
		}
		out = append(out, spec.Name+" "+encodeSlice(a.Vals[i]))
	}
	return out
}

// encodeStructSlice prints a struct buffer field by field, in declaration
// order, in exactly the form the probe's own emit prints it. It goes through
// the catalogue's getters because nothing outside structs.go names the type.
func encodeStructSlice(shape *StructShape, buf any) string {
	var parts []string
	n := shape.length(buf)
	for i := range n {
		for _, f := range shape.Fields {
			parts = append(parts, fieldText(f, buf, i))
		}
	}
	return strings.Join(parts, " ")
}

// fieldText is one field of one element, printed as its kind is printed
// everywhere else here: a float by its bits, so that a NaN, a negative zero
// and a subnormal all survive the comparison.
func fieldText(f StructField, buf any, i int) string {
	switch f.Kind {
	case KF32:
		return f32text(f.get.(func(any, int) float32)(buf, i))
	case KF64:
		return f64text(f.get.(func(any, int) float64)(buf, i))
	case KI32:
		return fmt.Sprintf("%d", f.get.(func(any, int) int32)(buf, i))
	case KI64:
		return fmt.Sprintf("%d", f.get.(func(any, int) int64)(buf, i))
	case KU32:
		return fmt.Sprintf("%d", f.get.(func(any, int) uint32)(buf, i))
	case KU64:
		return fmt.Sprintf("%d", f.get.(func(any, int) uint64)(buf, i))
	case KBool:
		return fmt.Sprintf("%t", f.get.(func(any, int) bool)(buf, i))
	case KI8:
		return fmt.Sprintf("%d", f.get.(func(any, int) int8)(buf, i))
	case KI16:
		return fmt.Sprintf("%d", f.get.(func(any, int) int16)(buf, i))
	case KU8:
		return fmt.Sprintf("%d", f.get.(func(any, int) uint8)(buf, i))
	case KU16:
		return fmt.Sprintf("%d", f.get.(func(any, int) uint16)(buf, i))
	}
	panic("fuzz: cannot encode a struct field of kind " + f.Kind.goName())
}

// structLiteral writes the source-side argument: a composite literal of the
// kernel package's own struct type, field by field, so that both renderings
// start from identical bytes.
func structLiteral(pkg string, shape *StructShape, buf any) string {
	n := shape.length(buf)
	elems := make([]string, n)
	for i := range n {
		fields := make([]string, len(shape.Fields))
		for j, f := range shape.Fields {
			fields[j] = f.Name + ": " + fieldLiteral(f, buf, i)
		}
		elems[i] = "{" + strings.Join(fields, ", ") + "}"
	}
	return fmt.Sprintf("[]%s.%s{%s}", pkg, shape.Name, strings.Join(elems, ", "))
}

// fieldLiteral is one field as Go source. A float goes through Frombits for
// the same reason goLiteral does it: a literal cannot spell a NaN, and %v
// would round a subnormal.
func fieldLiteral(f StructField, buf any, i int) string {
	switch f.Kind {
	case KF32:
		return fmt.Sprintf("math.Float32frombits(%#x)", math.Float32bits(f.get.(func(any, int) float32)(buf, i)))
	case KF64:
		return fmt.Sprintf("math.Float64frombits(%#x)", math.Float64bits(f.get.(func(any, int) float64)(buf, i)))
	case KBool:
		return fmt.Sprintf("%t", f.get.(func(any, int) bool)(buf, i))
	}
	return fieldText(f, buf, i)
}

func encodeSlice(v any) string {
	var parts []string
	switch v := v.(type) {
	case []float32:
		for _, x := range v {
			parts = append(parts, f32text(x))
		}
	case []float64:
		for _, x := range v {
			parts = append(parts, f64text(x))
		}
	case []bool:
		for _, x := range v {
			parts = append(parts, fmt.Sprintf("%t", x))
		}
	case []int32:
		for _, x := range v {
			parts = append(parts, fmt.Sprintf("%d", x))
		}
	case []int64:
		for _, x := range v {
			parts = append(parts, fmt.Sprintf("%d", x))
		}
	case []uint32:
		for _, x := range v {
			parts = append(parts, fmt.Sprintf("%d", x))
		}
	case []uint64:
		for _, x := range v {
			parts = append(parts, fmt.Sprintf("%d", x))
		}
	case []int8:
		for _, x := range v {
			parts = append(parts, fmt.Sprintf("%d", x))
		}
	case []int16:
		for _, x := range v {
			parts = append(parts, fmt.Sprintf("%d", x))
		}
	case []uint8:
		for _, x := range v {
			parts = append(parts, fmt.Sprintf("%d", x))
		}
	case []uint16:
		for _, x := range v {
			parts = append(parts, fmt.Sprintf("%d", x))
		}
	default:
		panic(fmt.Sprintf("fuzz: cannot encode %T", v))
	}
	return strings.Join(parts, " ")
}

// f32text and f64text print a float's bits, except for a NaN, which prints as
// one word whatever its payload and sign.
//
// The sign of a NaN is not a property two compilations agree on, let alone two
// backends: -Inf/-Inf produces a NaN whose sign is the hardware's, and
// negating or adding NaNs carries a payload nobody specifies. NUMERICS.md
// already says a tolerance relates a NaN to nothing and a test that expects
// one asks IsNaN; this is the same rule, applied where the comparison is made.
// Everything else -- a negative zero, a subnormal, an infinity -- is compared
// bit for bit, because those the two sides really do have to agree about.
func f32text(x float32) string {
	if x != x {
		return "NaN"
	}
	return fmt.Sprintf("%08x", math.Float32bits(x))
}

func f64text(x float64) string {
	if x != x {
		return "NaN"
	}
	return fmt.Sprintf("%016x", math.Float64bits(x))
}

// goLiteral writes a value as Go source that reproduces it exactly. A float
// goes through its bits for the same reason encode prints them.
func goLiteral(v any) string {
	switch v := v.(type) {
	case []float32:
		parts := make([]string, len(v))
		for i, x := range v {
			parts[i] = fmt.Sprintf("math.Float32frombits(%#x)", math.Float32bits(x))
		}
		return "[]float32{" + strings.Join(parts, ", ") + "}"
	case []float64:
		parts := make([]string, len(v))
		for i, x := range v {
			parts[i] = fmt.Sprintf("math.Float64frombits(%#x)", math.Float64bits(x))
		}
		return "[]float64{" + strings.Join(parts, ", ") + "}"
	case float32:
		return fmt.Sprintf("math.Float32frombits(%#x)", math.Float32bits(v))
	case float64:
		return fmt.Sprintf("math.Float64frombits(%#x)", math.Float64bits(v))
	case []bool:
		return sliceLiteral("bool", len(v), func(i int) string { return fmt.Sprintf("%t", v[i]) })
	case []int32:
		return sliceLiteral("int32", len(v), func(i int) string { return fmt.Sprintf("%d", v[i]) })
	case []int64:
		return sliceLiteral("int64", len(v), func(i int) string { return fmt.Sprintf("%d", v[i]) })
	case []uint32:
		return sliceLiteral("uint32", len(v), func(i int) string { return fmt.Sprintf("%d", v[i]) })
	case []uint64:
		return sliceLiteral("uint64", len(v), func(i int) string { return fmt.Sprintf("%d", v[i]) })
	case []int8:
		return sliceLiteral("int8", len(v), func(i int) string { return fmt.Sprintf("%d", v[i]) })
	case []int16:
		return sliceLiteral("int16", len(v), func(i int) string { return fmt.Sprintf("%d", v[i]) })
	case []uint8:
		return sliceLiteral("uint8", len(v), func(i int) string { return fmt.Sprintf("%d", v[i]) })
	case []uint16:
		return sliceLiteral("uint16", len(v), func(i int) string { return fmt.Sprintf("%d", v[i]) })
	case bool:
		return fmt.Sprintf("%t", v)
	case int:
		return fmt.Sprintf("int(%d)", v)
	case int32:
		return fmt.Sprintf("int32(%d)", v)
	case int64:
		return fmt.Sprintf("int64(%d)", v)
	case uint32:
		return fmt.Sprintf("uint32(%d)", v)
	case uint64:
		return fmt.Sprintf("uint64(%d)", v)
	}
	panic(fmt.Sprintf("fuzz: cannot spell %T", v))
}

func sliceLiteral(elem string, n int, at func(int) string) string {
	parts := make([]string, n)
	for i := range parts {
		parts[i] = at(i)
	}
	return "[]" + elem + "{" + strings.Join(parts, ", ") + "}"
}

// runSources compiles every program's Source() and runs it, returning what
// each one left in its buffers, encoded as encode does.
func runSources(t *testing.T, progs []*Program, args []*Args) [][]string {
	t.Helper()
	dir := t.TempDir()
	var body, imports strings.Builder
	for i, p := range progs {
		pkg := fmt.Sprintf("p%d", i)
		if err := os.MkdirAll(filepath.Join(dir, pkg), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, pkg, "k.go"), []byte(p.Source()), 0o644); err != nil {
			t.Fatal(err)
		}
		fmt.Fprintf(&imports, "\tk%d %q\n", i, "fuzzprobe/"+pkg)
		writeRunner(&body, i, p, args[i])
	}

	mod := fmt.Sprintf("module fuzzprobe\n\ngo %s\n\nrequire github.com/CWBudde/gocuda v0.0.0\n\nreplace github.com/CWBudde/gocuda => %s\n",
		goModVersion(), repoRoot(t))
	write(t, filepath.Join(dir, "go.mod"), mod)
	write(t, filepath.Join(dir, "main.go"), mainSource(imports.String(), body.String()))

	cmd := exec.Command("go", "run", ".")
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("go run the generated source: %v\n%s", err, out)
	}
	return splitSections(t, string(out), len(progs))
}

func write(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

// writeRunner emits the function that runs one program with its arguments and
// prints what its buffers hold afterwards.
func writeRunner(w *strings.Builder, i int, p *Program, a *Args) {
	fmt.Fprintf(w, "\nfunc run%d() {\n", i)
	fmt.Fprintf(w, "\tfmt.Printf(\"--- %d\\n\")\n", i)
	names := make([]string, len(p.Params))
	for j, spec := range p.Params {
		names[j] = fmt.Sprintf("a%d", j)
		if spec.AliasOf >= 0 {
			// Two read-only parameters bound to one buffer on purpose, which
			// the source has to be run with if it is to compute what the
			// closure computed.
			fmt.Fprintf(w, "\t%s := %s\n", names[j], names[spec.AliasOf])
			continue
		}
		if spec.Shape != nil {
			fmt.Fprintf(w, "\t%s := %s\n", names[j], structLiteral(fmt.Sprintf("k%d", i), spec.Shape, a.Vals[j]))
			continue
		}
		fmt.Fprintf(w, "\t%s := %s\n", names[j], goLiteral(a.Vals[j]))
	}
	call := fmt.Sprintf("k%d.%s(ctx, %s)", i, p.Name(), strings.Join(names, ", "))
	launch := fmt.Sprintf("gpu.RunCPUDim(gpu.D3(%d, %d, %d), gpu.D3(%d, %d, %d), func(ctx gpu.Ctx) { %s })",
		p.Grid.X, p.Grid.Y, p.Grid.Z, p.Block.X, p.Block.Y, p.Block.Z, call)
	if p.HasDyn {
		launch = fmt.Sprintf("gpu.RunCPUSharedDim(gpu.D3(%d, %d, %d), gpu.D3(%d, %d, %d), %d, func(ctx gpu.Ctx) { %s })",
			p.Grid.X, p.Grid.Y, p.Grid.Z, p.Block.X, p.Block.Y, p.Block.Z, p.DynLen, call)
	}
	fmt.Fprintf(w, "\t%s\n", launch)
	for j, spec := range p.Params {
		if !spec.Slice {
			fmt.Fprintf(w, "\t_ = %s\n", names[j])
			continue
		}
		fmt.Fprintf(w, "\temit(%q, %s)\n", spec.Name, names[j])
	}
	fmt.Fprintf(w, "}\n")
}

// mainSource is the fixed half of the probe: the emitters, which print exactly
// what encodeSlice prints.
func mainSource(imports, body string) string {
	var calls strings.Builder
	for i := 0; strings.Contains(body, fmt.Sprintf("func run%d()", i)); i++ {
		fmt.Fprintf(&calls, "\trun%d()\n", i)
	}
	return `package main

import (
	"fmt"
	"math"
	"reflect"
	"strings"

	"github.com/CWBudde/gocuda/gpu"
` + imports + `)

func main() {
` + calls.String() + `}

func emit(name string, v any) {
	var parts []string
	switch v := v.(type) {
	case []float32:
		for _, x := range v {
			if x != x {
				parts = append(parts, "NaN")
			} else {
				parts = append(parts, fmt.Sprintf("%08x", math.Float32bits(x)))
			}
		}
	case []float64:
		for _, x := range v {
			if x != x {
				parts = append(parts, "NaN")
			} else {
				parts = append(parts, fmt.Sprintf("%016x", math.Float64bits(x)))
			}
		}
	case []bool:
		for _, x := range v {
			parts = append(parts, fmt.Sprintf("%t", x))
		}
	case []int32:
		for _, x := range v {
			parts = append(parts, fmt.Sprintf("%d", x))
		}
	case []int64:
		for _, x := range v {
			parts = append(parts, fmt.Sprintf("%d", x))
		}
	case []uint32:
		for _, x := range v {
			parts = append(parts, fmt.Sprintf("%d", x))
		}
	case []uint64:
		for _, x := range v {
			parts = append(parts, fmt.Sprintf("%d", x))
		}
	case []int8:
		for _, x := range v {
			parts = append(parts, fmt.Sprintf("%d", x))
		}
	case []int16:
		for _, x := range v {
			parts = append(parts, fmt.Sprintf("%d", x))
		}
	case []uint8:
		for _, x := range v {
			parts = append(parts, fmt.Sprintf("%d", x))
		}
	case []uint16:
		for _, x := range v {
			parts = append(parts, fmt.Sprintf("%d", x))
		}
	default:
		// A struct buffer. Its element type lives in the kernel package, which
		// this one imports under a generated alias, so it is reached by
		// reflection rather than by name -- and printed field by field, in
		// declaration order, in exactly the form encodeStructSlice uses.
		rv := reflect.ValueOf(v)
		if rv.Kind() != reflect.Slice || rv.Type().Elem().Kind() != reflect.Struct {
			panic("probe: cannot encode")
		}
		for i := 0; i < rv.Len(); i++ {
			e := rv.Index(i)
			for j := 0; j < e.NumField(); j++ {
				parts = append(parts, fieldWord(e.Field(j)))
			}
		}
	}
	fmt.Println(name + " " + strings.Join(parts, " "))
}

// fieldWord prints one struct field, matching encode's rules for its kind: a
// float by its bits, with a NaN collapsed to one word whatever its payload.
func fieldWord(v reflect.Value) string {
	switch v.Kind() {
	case reflect.Float32:
		x := float32(v.Float())
		if x != x {
			return "NaN"
		}
		return fmt.Sprintf("%08x", math.Float32bits(x))
	case reflect.Float64:
		x := v.Float()
		if x != x {
			return "NaN"
		}
		return fmt.Sprintf("%016x", math.Float64bits(x))
	case reflect.Bool:
		return fmt.Sprintf("%t", v.Bool())
	case reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return fmt.Sprintf("%d", v.Int())
	case reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return fmt.Sprintf("%d", v.Uint())
	}
	panic("probe: cannot encode a struct field")
}
` + body
}

// splitSections pulls the per-program output apart again.
func splitSections(t *testing.T, out string, n int) [][]string {
	t.Helper()
	sections := make([][]string, n)
	cur := -1
	for line := range strings.SplitSeq(strings.TrimRight(out, "\n"), "\n") {
		if after, ok := strings.CutPrefix(line, "--- "); ok {
			var i int
			if _, err := fmt.Sscanf(after, "%d", &i); err != nil || i < 0 || i >= n {
				t.Fatalf("probe printed an unexpected marker %q", line)
			}
			cur = i
			continue
		}
		if cur < 0 {
			t.Fatalf("probe printed %q before any marker", line)
		}
		sections[cur] = append(sections[cur], line)
	}
	return sections
}

// repoRoot is the module this test belongs to, which the probe module replaces
// by path so that it needs no network and no published version.
func repoRoot(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("cannot locate this test's source")
	}
	return filepath.Dir(filepath.Dir(filepath.Dir(file)))
}

// goModVersion is the language version the probe declares. Matching this
// module's keeps the probe from asking for a toolchain this one does not use.
func goModVersion() string {
	v := strings.TrimPrefix(runtime.Version(), "go")
	if i := strings.IndexAny(v, "-+ "); i >= 0 {
		v = v[:i]
	}
	if v == "" {
		return "1.26.0"
	}
	return v
}

// runClosure executes the closure rendering and encodes what it left behind.
func runClosure(p *Program, a *Args) []string {
	p.Run(a)
	return encode(p, a)
}
