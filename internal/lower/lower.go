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
// files is the whole package. A kernel may call other functions in it, and
// each one it reaches is emitted into the same translation unit, so the
// lowering needs the declarations rather than just the entry point's.
//
// A Unit is returned only when nothing was refused; a kernel that produced any
// diagnostic yields (nil, diags), so half-lowered CUDA can never escape.
func Kernel(fset *token.FileSet, info *types.Info, files []*ast.File, fd *ast.FuncDecl) (*Unit, []Diagnostic) {
	t := &transpiler{fset: fset, info: info, files: files, lens: map[types.Object]string{}}
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
	// names holds every identifier the function spells, so that a generated
	// name -- the index a `for _, v := range` still needs -- can be chosen
	// where it cannot collide with one the author wrote.
	names map[string]bool
	// labels holds the labelled loops currently open, keyed by their Go name.
	labels map[string]*labelState
	// pendingLabel is the label the next loop will carry. It is claimed by
	// that loop and cleared immediately, so a nested one cannot inherit it.
	pendingLabel *labelState
	// breakables is what a bare `break` would leave, innermost last.
	breakables []breakable
	// files is the package the kernel was declared in, which is where a call
	// to another Go function has to be resolved to a declaration.
	files []*ast.File
	// current is the function being lowered, and lowering holds every
	// function on the path to it. Together they are the cycle check: a device
	// function that reaches itself is refused rather than emitted, because
	// there is no stack depth on the device to spend on recursion.
	current  *types.Func
	lowering map[*types.Func]bool
	// deviceNames maps an emitted device function to its C name, so one
	// reached by two paths is defined once.
	deviceNames map[*types.Func]string
	// deviceProtos and deviceDefs accumulate the emitted device functions.
	// Prototypes are written before any definition, which is what makes the
	// order of the Go declarations irrelevant.
	deviceProtos []string
	deviceDefs   []string
	// inDevice is set while a device function is being lowered. Shared memory
	// and the launch contract belong to the kernel, so both are refused here.
	inDevice bool
	// result is the device function's return type, nil in a kernel and in one
	// returning nothing.
	result types.Type
}

// capture renders f into a buffer of its own and returns it, leaving the
// emitter's own output untouched. It is how the entry point can be lowered
// first -- which is what discovers the device functions -- and still be
// written last.
func (t *transpiler) capture(f func()) string {
	saved := t.buf
	t.buf = strings.Builder{}
	f()
	out := t.buf.String()
	t.buf = saved
	return out
}

// collectNames records every identifier spelled anywhere in fn.
//
// It is deliberately blunt: a generated name is rejected because the spelling
// occurs at all, not because it is in scope. Scope information is not
// available on both of the paths that share this lowering, and a name nobody
// used is no loss.
func collectNames(fn *ast.FuncDecl) map[string]bool {
	names := map[string]bool{}
	ast.Inspect(fn, func(n ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok {
			names[id.Name] = true
		}
		return true
	})
	return names
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
	t.names = collectNames(fd)
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

	// The entry point is rendered first, because that is what discovers the
	// device functions it calls, and written last, because C needs them
	// declared before it sees the call.
	entry := t.capture(func() {
		t.line("extern \"C\" __global__ void %s(%s)", fd.Name.Name, strings.Join(decls, ", "))
		t.block(fd.Body)
	})

	t.line("// generated by github.com/CWBudde/gocuda/simt from %s", t.fset.Position(fd.Pos()).Filename)
	for _, proto := range t.deviceProtos {
		t.line("%s;", proto)
	}
	if len(t.deviceProtos) > 0 {
		t.line("")
	}
	for _, def := range t.deviceDefs {
		t.buf.WriteString(def)
		t.line("")
	}
	t.buf.WriteString(entry)
}

// deviceFunc lowers the package-local function obj into the translation unit,
// once, and reports the C name a call to it is spelled with.
//
// Everything a kernel may contain, a device function may contain: it is the
// same statement and expression machinery, with the signature rules of a
// kernel minus the gpu.Ctx and plus a return value.
func (t *transpiler) deviceFunc(pos token.Pos, obj *types.Func) (string, bool) {
	if name, ok := t.deviceNames[obj]; ok {
		return name, true
	}
	if t.lowering[obj] {
		if obj == t.current {
			t.fail(pos, "%s calls itself; recursion is not supported in kernels", obj.Name())
		} else {
			t.fail(pos, "%s calls %s, which is already being lowered; recursion is not supported in kernels",
				t.current.Name(), obj.Name())
		}
		return "", false
	}
	fd := t.declOf(obj)
	if fd == nil {
		t.fail(pos, "%s is not declared in this package", obj.Name())
		return "", false
	}
	if t.isKernelDecl(fd) {
		t.fail(pos, "%s is a kernel; a kernel cannot be called from a kernel. "+
			"Mark it //gocuda:ignore to make it a device function instead", obj.Name())
		return "", false
	}
	if fd.Recv != nil {
		t.fail(pos, "%s is a method; a device function must be a plain function", obj.Name())
		return "", false
	}
	if fd.Type.TypeParams != nil {
		t.fail(pos, "%s is generic; a device function must not be generic", obj.Name())
		return "", false
	}
	if sig, ok := obj.Type().(*types.Signature); ok && sig.Variadic() {
		t.fail(pos, "%s must not be variadic", obj.Name())
		return "", false
	}
	if fd.Type.Results != nil && len(fd.Type.Results.List) > 1 {
		t.fail(pos, "%s must return at most one value", obj.Name())
		return "", false
	}

	name := cname(obj.Name())
	// Reserve the name before the body is lowered: a helper that reaches
	// itself has to find it taken, which is what the cycle check reads.
	if t.lowering == nil {
		t.lowering = map[*types.Func]bool{}
	}
	t.lowering[obj] = true
	defer delete(t.lowering, obj)

	savedCurrent, savedDevice, savedResult, savedNames := t.current, t.inDevice, t.result, t.names
	t.current, t.inDevice, t.names = obj, true, collectNames(fd)
	t.result = nil
	if fd.Type.Results != nil && len(fd.Type.Results.List) == 1 {
		t.result = t.typeOf(fd.Type.Results.List[0].Type)
	}

	ret := "void"
	if t.result != nil {
		ret = t.ctype(t.result, fd.Type.Results.Pos())
	}
	proto := fmt.Sprintf("__device__ %s %s(%s)", ret, name, t.signature(fd))
	// The discovery happens partway through another function's body, so the
	// indent has to start again: this definition is emitted at file scope.
	savedInd := t.ind
	t.ind = 0
	def := t.capture(func() {
		t.line("%s", proto)
		t.block(fd.Body)
	})
	t.ind = savedInd

	t.current, t.inDevice, t.result, t.names = savedCurrent, savedDevice, savedResult, savedNames

	if t.deviceNames == nil {
		t.deviceNames = map[*types.Func]string{}
	}
	t.deviceNames[obj] = name
	t.deviceProtos = append(t.deviceProtos, proto)
	t.deviceDefs = append(t.deviceDefs, def)
	return name, true
}

// declOf finds the declaration obj was checked from.
func (t *transpiler) declOf(obj *types.Func) *ast.FuncDecl {
	for _, f := range t.files {
		for _, d := range f.Decls {
			fd, ok := d.(*ast.FuncDecl)
			if ok && fd.Name != nil && t.info.Defs[fd.Name] == obj {
				return fd
			}
		}
	}
	return nil
}

// isKernelDecl reports whether fd would be generated as a kernel, which is
// IsKernelDecl plus the file-wide opt-out that Package.Names also honours.
func (t *transpiler) isKernelDecl(fd *ast.FuncDecl) bool {
	for _, f := range t.files {
		if fd.Pos() < f.Pos() || fd.Pos() > f.End() {
			continue
		}
		if Ignored(f.Doc) {
			return false
		}
		break
	}
	return IsKernelDecl(t.info, fd)
}

// signature renders a C parameter list: a leading gpu.Ctx is dropped, because
// it has no device representation, and every slice becomes a pointer plus the
// length C does not carry.
func (t *transpiler) signature(fd *ast.FuncDecl) string {
	params := t.deviceParams(fd)
	t.checkLengthNames(params)
	decls := make([]string, 0, len(params))
	for _, p := range params {
		name := cname(p.name)
		switch typ := p.typ.(type) {
		case *types.Slice:
			elem := t.ctype(typ.Elem(), p.pos)
			decls = append(decls, fmt.Sprintf("%s* %s", elem, name), fmt.Sprintf("int %s_len", name))
			t.lens[p.obj] = name + "_len"
		default:
			decls = append(decls, fmt.Sprintf("%s %s", t.ctype(p.typ, p.pos), name))
		}
	}
	return strings.Join(decls, ", ")
}

// deviceParams is fd's parameters without a leading gpu.Ctx.
func (t *transpiler) deviceParams(fd *ast.FuncDecl) []param {
	params := t.params(fd.Type.Params)
	if len(params) > 0 && IsCtx(params[0].typ) {
		return params[1:]
	}
	return params
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
		if len(f.Names) == 0 {
			// An unnamed parameter cannot be referred to, and silently
			// dropping it would shift every argument after it.
			t.fail(f.Pos(), "every parameter must be named")
			continue
		}
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
