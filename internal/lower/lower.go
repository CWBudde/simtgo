// Package lower holds the Go-to-CUDA-C lowering: the part of the SIMT track
// that decides what a kernel may contain and renders what it does.
//
// It lives below package simt, and takes an already type-checked declaration
// rather than doing its own parsing, so that two callers can share exactly one
// definition of the supported subset: simt.Transpile, which type-checks kernel
// sources at run time, and the gocuda vet analyzer, which is handed the real
// package by the go/analysis framework. A subset that was described twice
// would eventually be described differently.
package lower

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"sort"
	"strings"
)

// Unit is one transpiled kernel: the generated CUDA C together with what its
// source says about how it has to be launched.
//
// Those requirements are discovered while lowering -- a shared tile's extent
// is only known there -- so they travel with the source instead of having to
// be restated, and kept in step, at every launch site.
type Unit struct {
	Source        string // the generated CUDA C
	Name          string // the Go function's name, also the C entry point
	RequiredBlock int    // block size the kernel demands, 0 when it declares none
	SharedBytes   int    // total statically declared __shared__ bytes
	// SourceHash identifies Source, and is what an ahead-of-time artifact is
	// filed under. Hashing the generated CUDA C rather than the Go source
	// means a change to the emitter invalidates a prebuilt just as a change to
	// the kernel does, and that comments and formatting in the Go source,
	// which cannot affect the output, do not.
	SourceHash string
}

// SourceHash is the identity of a piece of generated CUDA C. It is the whole
// staleness story for ahead-of-time artifacts: a prebuilt is found by what it
// was built from, so one built from anything else is simply not found.
func SourceHash(src string) string {
	sum := sha256.Sum256([]byte(src))
	return hex.EncodeToString(sum[:])
}

// A Diagnostic is one reason a kernel cannot be lowered.
//
// Msg carries neither a position nor a package prefix: an analysis.Diagnostic
// supplies its own position and category, and simt supplies both when it
// renders a diagnostic as an error, so a message that included them would say
// them twice.
type Diagnostic struct {
	Pos token.Pos
	Msg string
}

// Kernel lowers the kernel function fd, which must already have been
// type-checked into info, to CUDA C.
//
// A Unit is returned only when nothing was refused; a kernel that produced any
// diagnostic yields (nil, diags), so half-lowered CUDA can never escape.
func Kernel(fset *token.FileSet, info *types.Info, fd *ast.FuncDecl) (*Unit, []Diagnostic) {
	t := &transpiler{fset: fset, info: info, lens: map[types.Object]string{}}
	t.kernel(fd)
	if len(t.diags) > 0 {
		return nil, tidy(t.diags)
	}
	src := t.buf.String()
	return &Unit{
		Source:        src,
		Name:          fd.Name.Name,
		RequiredBlock: t.requiredBlock,
		SharedBytes:   t.sharedBytes,
		SourceHash:    SourceHash(src),
	}, nil
}

// tidy puts diagnostics in source order and drops exact duplicates, which a
// node reachable by two paths can otherwise produce.
func tidy(diags []Diagnostic) []Diagnostic {
	sort.SliceStable(diags, func(i, j int) bool { return diags[i].Pos < diags[j].Pos })
	out := diags[:0]
	var last Diagnostic
	for i, d := range diags {
		if i > 0 && d == last {
			continue
		}
		out = append(out, d)
		last = d
	}
	return out
}

type transpiler struct {
	fset *token.FileSet
	info *types.Info
	buf  strings.Builder
	ind  int
	// lens maps a slice-valued object to the C expression giving its length.
	// The key is the checked object rather than its name, so a declaration
	// shadowing a slice parameter cannot be mistaken for it.
	lens map[types.Object]string
	// requiredBlock and sharedBytes accumulate what the kernel demands of its
	// launch; see Unit.
	requiredBlock int
	sharedBytes   int
	// diags collects every construct this kernel was refused for.
	diags []Diagnostic
	// mark is len(diags) when the current statement began. Refusals are
	// collected per statement: descending further into a statement that has
	// already been refused only produces noise about the wreckage, but the
	// next statement starts clean and can be refused on its own merits. That
	// is the difference between reporting one problem per run and reporting
	// what is actually wrong with the kernel.
	mark int
	// poisoned holds objects a diagnostic was already issued about, so that
	// every later mention of a variable whose declaration was refused stays
	// quiet instead of repeating the consequences of one mistake.
	poisoned map[types.Object]bool
}

func (t *transpiler) fail(pos token.Pos, format string, args ...any) {
	t.diags = append(t.diags, Diagnostic{Pos: pos, Msg: fmt.Sprintf(format, args...)})
}

// failed reports whether the statement being rendered has already been refused.
func (t *transpiler) failed() bool { return len(t.diags) > t.mark }

// poison records that obj has been reported on, so later uses stay silent.
func (t *transpiler) poison(obj types.Object) {
	if obj == nil {
		return
	}
	if t.poisoned == nil {
		t.poisoned = map[types.Object]bool{}
	}
	t.poisoned[obj] = true
}

func (t *transpiler) line(format string, args ...any) {
	t.buf.WriteString(strings.Repeat("\t", t.ind))
	fmt.Fprintf(&t.buf, format, args...)
	t.buf.WriteByte('\n')
}

// kernel emits the __global__ entry point for fd.
func (t *transpiler) kernel(fd *ast.FuncDecl) {
	if fd.Recv != nil {
		t.fail(fd.Pos(), "a kernel must be a plain function, not a method")
		return
	}
	if fd.Type.Results != nil {
		t.fail(fd.Pos(), "a kernel must not return values; write results through a slice parameter")
		return
	}
	if fd.Type.TypeParams != nil {
		// Reported here rather than left to ctype, which would otherwise
		// complain about the type parameter's interface underlying type and
		// tell the author nothing about why.
		t.fail(fd.Pos(), "a kernel must not be generic")
		return
	}
	params := t.params(fd.Type.Params)
	if len(params) == 0 || !IsCtx(params[0].typ) {
		t.fail(fd.Pos(), "a kernel's first parameter must be gpu.Ctx")
		return
	}
	// A name clash is a fact about the signature, not a reason to stop: the
	// body is lowered anyway so that its own problems are reported in the
	// same run. Nothing is emitted while any diagnostic stands.
	t.checkLengthNames(params[1:])

	var decls []string
	for _, p := range params[1:] {
		name := cname(p.name)
		switch typ := p.typ.(type) {
		case *types.Slice:
			elem := t.ctype(typ.Elem(), p.pos)
			// A Go slice carries its length; C does not, so every slice
			// parameter lowers to a pointer plus an explicit length.
			decls = append(decls, fmt.Sprintf("%s* %s", elem, name), fmt.Sprintf("int %s_len", name))
			t.lens[p.obj] = name + "_len"
		default:
			decls = append(decls, fmt.Sprintf("%s %s", t.ctype(p.typ, p.pos), name))
		}
	}

	t.line("// generated by github.com/CWBudde/gocuda/simt from %s", t.fset.Position(fd.Pos()).Filename)
	t.line("extern \"C\" __global__ void %s(%s)", fd.Name.Name, strings.Join(decls, ", "))
	t.block(fd.Body)
}

// checkLengthNames refuses a kernel whose own parameters clash with the length
// parameters synthesised for its slices.
//
// The readable x_len spelling is worth keeping, so the clash is reported here,
// against the Go source, rather than left to NVRTC -- which would complain
// about a duplicate parameter in generated code the author never wrote. The
// tile track spells lengths the same way but names its parameters p0, p1, ...
// itself, so it has nothing to check.
func (t *transpiler) checkLengthNames(params []param) {
	generated := make(map[string]string, len(params))
	for _, p := range params {
		if _, ok := p.typ.(*types.Slice); ok {
			generated[cname(p.name)+"_len"] = p.name
		}
	}
	for _, p := range params {
		if slice, ok := generated[cname(p.name)]; ok {
			t.fail(p.pos, "parameter %s collides with the length generated for slice parameter %s; rename it", p.name, slice)
		}
	}
}

type param struct {
	name string
	obj  types.Object
	typ  types.Type
	pos  token.Pos
}

// params flattens a parameter list, resolving each name's checked object. The
// object is kept, not just its type: it is what the symbol table is keyed on.
func (t *transpiler) params(fl *ast.FieldList) []param {
	var out []param
	if fl == nil {
		return out
	}
	for _, f := range fl.List {
		for _, n := range f.Names {
			obj := t.info.Defs[n]
			if obj == nil {
				t.fail(n.Pos(), "parameter %s has no resolved type", n.Name)
				return out
			}
			out = append(out, param{name: n.Name, obj: obj, typ: obj.Type(), pos: n.Pos()})
		}
	}
	return out
}

// IsCtx reports whether typ is gpu.Ctx, the marker that makes a function a
// kernel. It matches by package path, so it holds equally for the synthetic
// gpu package this repository type-checks against at run time and for the real
// one an analyzer is handed.
func IsCtx(typ types.Type) bool {
	named, ok := typ.(*types.Named)
	if !ok {
		return false
	}
	obj := named.Obj()
	return obj != nil && obj.Name() == "Ctx" && obj.Pkg() != nil && obj.Pkg().Path() == GPUPkgPath
}

// ctype maps a Go type to its device counterpart.
//
// Go's int is 64-bit while CUDA's int is 32-bit. Kernel indices are bounded by
// the grid, so the narrowing is safe here and keeps generated code idiomatic;
// it is the one deliberate infidelity in the mapping.
func (t *transpiler) ctype(typ types.Type, pos token.Pos) string {
	basic, ok := typ.Underlying().(*types.Basic)
	if !ok {
		t.fail(pos, "unsupported type %s on the device", typ)
		return "void"
	}
	if basic.Kind() == types.Invalid {
		// The type checker has already said something better about this.
		return "void"
	}
	switch basic.Kind() {
	case types.Float32, types.UntypedFloat:
		return "float"
	case types.Int, types.Int32, types.UntypedInt:
		return "int"
	case types.Uint32:
		return "unsigned int"
	case types.Bool, types.UntypedBool:
		return "bool"
	}
	t.fail(pos, "unsupported type %s on the device (kernels are float32/int32 only)", typ)
	return "void"
}
