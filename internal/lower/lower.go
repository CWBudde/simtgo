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
	"runtime"
	"sort"
	"strconv"
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
	// DynSharedWidth is the size in bytes of one element of the kernel's
	// dynamically sized __shared__ tile, and 0 when it declares none. One
	// field says both things because no element type is zero bytes wide, and
	// the host needs exactly this much: whether to size the dynamic block at
	// all, and what to multiply the caller's element count by.
	DynSharedWidth int
	// Params describes the kernel's parameters after the gpu.Ctx, in the order
	// a launch supplies them. It travels with the unit for the same reason
	// RequiredBlock does: what the source says about a parameter is discovered
	// here, and a launch is where it has to be honoured.
	Params []Param
	// SourceHash identifies Source, and is what an ahead-of-time artifact is
	// filed under. Hashing the generated CUDA C rather than the Go source
	// means a change to the emitter invalidates a prebuilt just as a change to
	// the kernel does, and that comments and formatting in the Go source,
	// which cannot affect the output, do not.
	SourceHash string
	// FastMath says the kernel carried //gocuda:fastmath, and is what tells a
	// caller to hand NVRTC --use_fast_math. It is a launch contract like the
	// fields above -- discovered while lowering, honoured by whoever compiles
	// -- but with a second job: Source already records the fact in a marker
	// comment, so SourceHash differs and no prebuilt built the ordinary way
	// can be mistaken for this one. The bool exists because reading the flag
	// back out of the source would be parsing our own output.
	FastMath bool
	// BoundsChecks says the unit was lowered with WithBoundsChecks, so every
	// subscript whose bound the emitter could name is range-checked and traps
	// on a violation. Like FastMath it is recorded in Source as well, so
	// SourceHash differs and a debug build cannot find a release artifact --
	// but unlike FastMath the checks themselves are usually already in those
	// bytes, and the marker is what covers the kernel that indexes nothing.
	BoundsChecks bool
}

// An Option is a build-time choice about what to emit.
//
// No Option refuses a program the default accepts. That is the rule the
// analyzer depends on: analysis/simtcheck passes none, so the subset it
// reports on is the subset a build accepts, by construction rather than by
// agreement. An Option may add code; it may never add a Diagnostic.
type Option func(*options)

type options struct{ boundsChecks bool }

// WithBoundsChecks emits a range check at every slice, array and shared-tile
// subscript whose bound the emitter can name, trapping on a violation.
//
// Why this is a build option rather than a source directive, and what the
// device can and cannot tell you afterwards:
// docs/decisions.md#bounds-checks-are-a-build-option.
func WithBoundsChecks() Option { return func(o *options) { o.boundsChecks = true } }

// A Param is one of a kernel's parameters, as the generated C declares it.
//
// ReadOnly is what the launch-time aliasing check reads: every pointer in the
// generated signature is __restrict__, which promises the parameters do not
// overlap, and that promise can only be broken by a buffer something writes.
// It is meaningful only for a slice -- a scalar is a copy.
type Param struct {
	Name     string
	Slice    bool
	ReadOnly bool
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
func Kernel(fset *token.FileSet, info *types.Info, files []*ast.File, fd *ast.FuncDecl, opts ...Option) (*Unit, []Diagnostic) {
	var o options
	for _, opt := range opts {
		opt(&o)
	}
	t := &transpiler{fset: fset, info: info, files: files, lens: map[types.Object]string{}, sizes: goSizes(), boundsChecks: o.boundsChecks}
	t.kernel(fd)
	if len(t.diags) > 0 {
		return nil, tidy(t.diags)
	}
	src := t.buf.String()
	return &Unit{
		Source:         src,
		Name:           fd.Name.Name,
		RequiredBlock:  t.requiredBlock,
		SharedBytes:    t.sharedBytes,
		DynSharedWidth: t.dynSharedWidth,
		Params:         t.paramInfo,
		SourceHash:     SourceHash(src),
		FastMath:       t.fastMath,
		BoundsChecks:   t.boundsChecks,
	}, nil
}

// goSizes is the layout the Go compiler uses on this machine.
//
// The generator and the analyzer both run on the host that builds the kernel,
// so both get the same answer; an artifact is keyed on the hash of the C it
// produced, which carries these numbers, so one generated elsewhere is simply
// not found rather than trusted.
func goSizes() types.Sizes {
	if s := types.SizesFor("gc", runtime.GOARCH); s != nil {
		return s
	}
	// Every host CUDA runs on is 64-bit with the same scalar layout, so this
	// only matters if Go gains an architecture gc has no entry for.
	return types.SizesFor("gc", "amd64")
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
	// sizes is what Go believes about layout: widths, alignments and field
	// offsets. It is asked rather than assumed so that a message about a type
	// too wide for the device can name the width, and so that the struct
	// definitions this emits can state Go's offsets for the C++ compiler to
	// check against its own.
	sizes types.Sizes
	// allowFloat64 is set from the kernel's //gocuda:float64 directive, and
	// kernelName is whose directive it would have been, for the message.
	allowFloat64 bool
	kernelName   string
	// fastMath is set from the kernel's //gocuda:fastmath directive. Unlike
	// allowFloat64 it gates nothing while lowering -- there is no construct to
	// refuse -- so its whole job is to reach Unit.FastMath and the marker line
	// in the generated source.
	fastMath bool
	// boundsChecks is set from WithBoundsChecks and gates every call to
	// checked. usedBounds records whether any of those calls actually emitted
	// one, which is what decides whether the helper is defined: a kernel that
	// indexes nothing must not carry a __device__ function nothing calls.
	boundsChecks bool
	usedBounds   bool
	// lens maps a slice-valued object to the C expression giving its length.
	// The key is the checked object rather than its name, so a declaration
	// shadowing a slice parameter cannot be mistaken for it.
	lens map[types.Object]string
	// requiredBlock, sharedBytes and paramInfo accumulate what the kernel
	// demands of its launch; see Unit.
	requiredBlock int
	sharedBytes   int
	// dynShared is the Go name of the kernel's dynamically sized tile, empty
	// until one is declared -- which is both the "at most one" check and what
	// names the first of the two in the refusal. dynSharedLen is the parameter
	// the launch fills with its length, and dynSharedWidth the element's size
	// in bytes, which is what turns the caller's element count into bytes.
	dynShared      string
	dynSharedLen   string
	dynSharedWidth int
	// sigNames is every C identifier the kernel's own signature spells, both
	// the parameters and the lengths generated for its slices. A dynamic
	// tile's generated length has to be checked against it, and unlike the
	// parameters it is only discovered while the body is lowered.
	sigNames  map[string]string
	paramInfo []Param
	// written memoises the read-only analysis, and analysing is its cycle
	// guard; see readonly.go.
	written   map[*ast.FuncDecl]map[types.Object]bool
	analysing map[*ast.FuncDecl]bool
	// uses memoises the barrier-divergence summaries, varying the uniformity
	// set each one was computed against, and diverging is that walk's cycle
	// guard; see diverge.go. They are separate from the three above because
	// the two analyses answer different questions about the same declarations
	// and neither needs the other's intermediate state.
	uses      map[*ast.FuncDecl]barrierUse
	varying   map[*ast.FuncDecl]map[types.Object]bool
	diverging map[*ast.FuncDecl]bool
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
	// renamed holds the locals whose C name had to differ from their Go one,
	// keyed on the object so that every use follows the declaration. It is
	// empty for almost every kernel: see shadowRename for the one case that
	// fills it.
	renamed map[types.Object]string
	// wrapSigned and wrapUnsigned are the open unsigned-arithmetic domain, or
	// "" when there is none. See wrapDomain in expr.go: Go's signed overflow
	// wraps and C's is undefined, so a region of + - * and << is computed in
	// the unsigned type of the same width and converted back once at its
	// boundary rather than once per operator.
	wrapSigned, wrapUnsigned string
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
	// structNames maps a struct type to its C name, and structDefs holds the
	// definitions in first-mention order. They are written ahead of the device
	// prototypes, because a prototype may name one.
	structNames map[*types.Named]string
	structDefs  []string
	// structLayouts records the padding members emitted for each struct, which
	// is what a positional literal has to step over.
	structLayouts map[*types.Named]structLayout
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

// collectDeclared records every variable fn declares, with the position of its
// first spelling, and nothing else.
//
// It is the precise counterpart to collectNames, which is deliberately blunt
// because it only has to keep a generated name from colliding with anything at
// all. Here bluntness would cost a refusal somebody could not act on: a struct
// field lives in its own C namespace and can share a spelling with a variable
// without either becoming the other, so counting one as a collision would
// refuse a kernel that is perfectly translatable. What becomes a C variable is
// a Go variable that is not a field, so that is what this collects.
func collectDeclared(info *types.Info, fn *ast.FuncDecl) map[string]token.Pos {
	out := map[string]token.Pos{}
	ast.Inspect(fn, func(n ast.Node) bool {
		id, ok := n.(*ast.Ident)
		if !ok {
			return true
		}
		v, ok := info.Defs[id].(*types.Var)
		if !ok || v.IsField() {
			return true
		}
		if _, seen := out[id.Name]; !seen {
			out[id.Name] = id.Pos()
		}
		return true
	})
	return out
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
	t.names, t.renamed = collectNames(fd), nil
	// The permission is the kernel's, and it covers every device function
	// lowered into the same translation unit. What is being opted into is the
	// cost of a launch, and the kernel is what gets launched -- the same
	// reasoning that keeps SharedF32 and AssumeBlockDim out of a device
	// function. A helper may therefore lower as float in one kernel and double
	// in another, which is fine: each kernel is its own translation unit.
	t.allowFloat64 = Float64Enabled(fd.Doc)
	t.fastMath = FastMathEnabled(fd.Doc)
	if f := t.fileOf(fd.Pos()); f != nil {
		if Float64Enabled(f.Doc) {
			t.allowFloat64 = true
		}
		if FastMathEnabled(f.Doc) {
			t.fastMath = true
		}
	}
	t.kernelName = fd.Name.Name
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
	t.checkGeneratedNames(fd, params[1:])

	// Divergence is a property of the whole call graph rather than of any one
	// statement, so it is asked once about the kernel here rather than at each
	// barrier as the body is rendered. It emits nothing; every answer it has
	// is a refusal.
	t.checkDivergence(fd)

	written := t.writtenParams(fd)
	var decls []string
	for _, p := range params[1:] {
		name := cname(p.name)
		switch typ := p.typ.(type) {
		case *types.Slice:
			elem := t.ctypeElem(typ.Elem(), p.pos)
			readOnly := !written[p.obj]
			// A Go slice carries its length; C does not, so every slice
			// parameter lowers to a pointer plus an explicit length.
			decls = append(decls, pointerDecl(elem, name, readOnly), fmt.Sprintf("int %s_len", name))
			t.lens[p.obj] = name + "_len"
			t.paramInfo = append(t.paramInfo, Param{Name: p.name, Slice: true, ReadOnly: readOnly})
		default:
			t.checkParamType(p.typ, p.pos)
			decls = append(decls, t.cdecl(p.typ, name, p.pos))
			t.paramInfo = append(t.paramInfo, Param{Name: p.name})
		}
	}

	// The body is rendered first, because that is what discovers the device
	// functions it calls -- and also the dynamically sized shared tile, whose
	// generated length is a parameter, so the signature cannot be written
	// until the body has been read. The definitions are written last for the
	// other half of the same reason: C needs them before it sees the call.
	body := t.capture(func() { t.block(fd.Body) })
	if t.dynSharedLen != "" {
		decls = append(decls, "int "+t.dynSharedLen)
	}
	entry := t.capture(func() {
		t.line("extern \"C\" __global__ void %s(%s)", fd.Name.Name, strings.Join(decls, ", "))
		t.buf.WriteString(body)
	})

	t.line("// generated by github.com/CWBudde/gocuda/simt from %s", t.fset.Position(fd.Pos()).Filename)
	if t.fastMath {
		// This line is the whole reason SourceHash keeps meaning "these exact
		// bytes". Fast math is a compiler flag and leaves no trace in the C,
		// so without the marker a prebuilt compiled the ordinary way would
		// hash identically and be found for a kernel that asked for the flag.
		// Writing it down instead of salting the hash also makes the committed
		// kernels/prebuilt/*.cu say how they were compiled, and gives
		// internal/jit's cacheKey -- which hashes the source -- the same
		// distinction for free.
		t.line("// gocuda: fastmath")
	}
	if t.boundsChecks {
		// Unlike the fast-math marker this one is redundant in every kernel
		// that indexes anything: the checks are themselves in the bytes the
		// hash covers. It earns its place on the degenerate case -- a kernel
		// with no subscript at all emits identical C either way, and without
		// the marker a debug build of it would find and load the release
		// prebuilt. That load would in fact be correct, the same bytes having
		// been compiled the same way, so this is not a correctness fix; it is
		// what lets "a debug build never finds a release artifact" be true
		// with no case split, and what lets someone reading a dumped .cu see
		// which mode produced it.
		t.line("// gocuda: bounds")
	}
	for _, def := range t.structDefs {
		t.buf.WriteString(def)
		t.line("")
	}
	if t.usedBounds {
		t.buf.WriteString(boundsHelperDef)
		t.line("")
	}
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
			"Mark it %s to make it a device function instead", obj.Name(), DeviceDirective)
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
	if FastMathEnabled(fd.Doc) {
		// Refused for the reason Float64Directive is, one line below, and with
		// a sharper edge: fast math is a property of the whole translation
		// unit -- NVRTC takes --use_fast_math for the compilation, not for a
		// function -- so a helper carrying it would be asking for something
		// the compiler cannot give one function without giving it every
		// function in the unit.
		t.fail(fd.Pos(), "%s belongs on the kernel, not on device function %s: it is a promise about how a launch computes", FastMathDirective, obj.Name())
		return "", false
	}
	if Float64Enabled(fd.Doc) {
		// The same reasoning that keeps SharedF32 and AssumeBlockDim out of a
		// device function: the directive is a statement about what a launch
		// costs, and a helper has no launch. Letting one carry its own opt-in
		// would also let a kernel without the directive inherit double
		// precision through a call, which is the one thing it exists to stop.
		t.fail(fd.Pos(), "%s belongs on the kernel, not on device function %s: it is a promise about what a launch costs", Float64Directive, obj.Name())
		return "", false
	}
	sig, ok := obj.Type().(*types.Signature)
	if !ok {
		t.fail(pos, "%s has no resolved signature", obj.Name())
		return "", false
	}
	if sig.Variadic() {
		t.fail(pos, "%s must not be variadic", obj.Name())
		return "", false
	}
	// The checked signature, not the AST: `(a, b float32)` is one result
	// field holding two values, so counting fields would let a two-result
	// function through and then render it as a scalar-returning C one.
	if sig.Results().Len() > 1 {
		t.fail(pos, "%s must return at most one value", obj.Name())
		return "", false
	}
	if namedResult(fd) {
		// A named result is a local the body may assign to and a bare return
		// that carries it. Neither is emitted, so the C would reference an
		// identifier that was never declared -- refused rather than written.
		t.fail(pos, "%s must not name its result", obj.Name())
		return "", false
	}
	if sig.Results().Len() == 1 {
		if res := sig.Results().At(0).Type(); isArray(res) {
			// C cannot return an array at all. Caught here rather than left to
			// whatever the caller does with the value, so the message names the
			// function that cannot be written this way.
			t.fail(fd.Pos(), "%s returns %s, and C cannot return an array; return a struct wrapping it, or write through a slice parameter", obj.Name(), res)
			return "", false
		}
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
	savedRenamed := t.renamed
	t.current, t.inDevice, t.names, t.renamed = obj, true, collectNames(fd), nil
	t.result = nil
	if sig.Results().Len() == 1 {
		t.result = sig.Results().At(0).Type()
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
	t.renamed = savedRenamed

	if t.deviceNames == nil {
		t.deviceNames = map[*types.Func]string{}
	}
	t.deviceNames[obj] = name
	t.deviceProtos = append(t.deviceProtos, proto)
	t.deviceDefs = append(t.deviceDefs, def)
	return name, true
}

// namedResult reports whether fd gives its result a name.
func namedResult(fd *ast.FuncDecl) bool {
	if fd.Type.Results == nil {
		return false
	}
	for _, f := range fd.Type.Results.List {
		if len(f.Names) > 0 {
			return true
		}
	}
	return false
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
	if f := t.fileOf(fd.Pos()); f != nil && Ignored(f.Doc) {
		return false
	}
	return IsKernelDecl(t.info, fd)
}

// fileOf finds the file a position falls in, which is where a file-wide
// directive is written.
func (t *transpiler) fileOf(pos token.Pos) *ast.File {
	for _, f := range t.files {
		if pos >= f.Pos() && pos <= f.End() {
			return f
		}
	}
	return nil
}

// signature renders a C parameter list: a leading gpu.Ctx is dropped, because
// it has no device representation, and every slice becomes a pointer plus the
// length C does not carry.
func (t *transpiler) signature(fd *ast.FuncDecl) string {
	params := t.deviceParams(fd)
	t.checkGeneratedNames(fd, params)
	written := t.writtenParams(fd)
	decls := make([]string, 0, len(params))
	for _, p := range params {
		name := cname(p.name)
		switch typ := p.typ.(type) {
		case *types.Slice:
			elem := t.ctypeElem(typ.Elem(), p.pos)
			decls = append(decls, pointerDecl(elem, name, !written[p.obj]), fmt.Sprintf("int %s_len", name))
			t.lens[p.obj] = name + "_len"
		default:
			t.checkParamType(p.typ, p.pos)
			decls = append(decls, t.cdecl(p.typ, name, p.pos))
		}
	}
	return strings.Join(decls, ", ")
}

// pointerDecl declares the pointer half of a slice parameter.
//
// Both qualifiers are claims about the whole kernel rather than decoration.
// const says this pointer is never written through, on any path the parameter
// reaches, which readonly.go proves conservatively. __restrict__ says no other
// pointer parameter reaches the same memory, which the Go source cannot
// promise at all -- VecAdd(ctx, c, a, b) may be handed one buffer three times
// -- so Kernel.Launch checks it against the buffers actually bound and refuses
// an overlap, and calls inside the translation unit are checked where they are
// lowered. An unchecked __restrict__ would be undefined behaviour dressed as a
// speed-up.
func pointerDecl(elem, name string, readOnly bool) string {
	if readOnly {
		return fmt.Sprintf("const %s* __restrict__ %s", elem, name)
	}
	return fmt.Sprintf("%s* __restrict__ %s", elem, name)
}

// deviceParams is fd's parameters without a leading gpu.Ctx.
func (t *transpiler) deviceParams(fd *ast.FuncDecl) []param {
	params := t.params(fd.Type.Params)
	if len(params) > 0 && IsCtx(params[0].typ) {
		return params[1:]
	}
	return params
}

// checkGeneratedNames refuses a function in which a name the emitter generates
// is already the name of one of the author's own variables.
//
// The emitter writes into the same C namespace the kernel does, and it writes
// two kinds of name there: the x_len carried alongside every slice, and the
// trailing underscore cname adds to a C++ keyword. Neither is visible in the
// Go source, and both were reachable, silently:
//
//	func K(ctx gpu.Ctx, y []float32) {    // y_len is generated
//	    y_len := int32(3)                 // and so is this, now
//	    if i < len(y) { ... }             // reads 3
//	}
//
//	func K(ctx gpu.Ctx, y []float32, int int32) {  // int is emitted as int_
//	    int_ := int32(99)                          // so is this
//	    y[i] = float32(int_) + float32(int)        // both read 99
//	}
//
// Each compiles under NVRTC and computes something the Go says nothing about,
// which is the failure this emitter exists to rule out. Only the first
// collision was checked before, and only against another parameter, so a local
// reached neither. Checking against every variable the function declares
// covers both, and covers a shared tile and a range variable with them, since
// the emitter names those from the source too.
//
// A device function is checked over its own body: its parameters and locals
// are what share a C scope with its generated lengths. The kernel additionally
// records sigNames, because a dynamically sized tile generates a length while
// the body is lowered, long after this runs.
func (t *transpiler) checkGeneratedNames(fd *ast.FuncDecl, params []param) {
	generated := make(map[string]string, len(params))
	for _, p := range params {
		if _, ok := p.typ.(*types.Slice); ok {
			generated[cname(p.name)+"_len"] = p.name
		}
	}
	declared := collectDeclared(t.info, fd)
	for g, slice := range generated {
		pos, taken := declared[g]
		if !taken {
			continue
		}
		t.fail(pos, "%s is the length generated for slice parameter %s, so in CUDA C the two would be one variable; rename it", g, slice)
	}
	// A C++ keyword is emitted with a trailing underscore, so a variable
	// already spelled that way becomes the same C identifier. Sorted, because
	// a map's order would shuffle the diagnostics between runs.
	spellings := make([]string, 0, len(declared))
	for name := range declared {
		spellings = append(spellings, name)
	}
	sort.Strings(spellings)
	for _, name := range spellings {
		escaped := cname(name)
		if escaped == name {
			continue
		}
		if pos, taken := declared[escaped]; taken {
			t.fail(pos, "%s is a C++ keyword and is emitted as %s, which is also declared here, so in CUDA C the two would be one variable; rename one of them", name, escaped)
		}
	}
	if t.inDevice {
		// A device function has its own parameters and no dynamic tile, so
		// recording them would only let the kernel's tile collide with a name
		// it never shares a scope with.
		return
	}
	t.sigNames = make(map[string]string, 2*len(params))
	for _, p := range params {
		t.sigNames[cname(p.name)] = "parameter " + p.name
	}
	for g, slice := range generated {
		t.sigNames[g] = "the length generated for slice parameter " + slice
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

// cdecl renders a C declaration of typ named name, without a terminator.
//
// It exists because C's declarator syntax is not "type, then name": an array's
// extent goes after the name, as `float taps[4]`, so no function returning a
// type alone can spell one. Every site that declares something -- parameters,
// locals, struct fields -- goes through here; ctype is what remains for the two
// places that genuinely want a bare type, a cast and a return type.
func (t *transpiler) cdecl(typ types.Type, name string, pos token.Pos) string {
	if arr, ok := typ.Underlying().(*types.Array); ok {
		if arr.Len() == 0 {
			// Go allows [0]T; C++ does not allow a zero-extent array, so
			// `float a[0]` is an NVRTC error about generated code. There is
			// nothing to lower it to that would still be an array, and nothing
			// an author could do with one anyway.
			t.fail(pos, "%s has no elements, and C++ has no zero-length array to lower it to", typ)
			return "void " + name
		}
		return fmt.Sprintf("%s %s[%d]", t.ctypeElem(arr.Elem(), pos), name, arr.Len())
	}
	return t.ctype(typ, pos) + " " + name
}

// checkParamType refuses an array parameter.
//
// Go passes an array by value; C decays an array parameter to a pointer, so the
// callee writes through the caller's storage. A helper that modifies its
// parameter would be a local change in Go and a caller-visible one in C -- the
// same code, two answers, with nothing to see at the call site. Wrapping it in
// a struct is one line and both languages then agree to copy.
func (t *transpiler) checkParamType(typ types.Type, pos token.Pos) {
	if _, ok := typ.Underlying().(*types.Array); ok {
		t.fail(pos, "%s cannot be a parameter: Go passes an array by value and C would pass a pointer to it, so a write inside the function would reach the caller's array; pass a slice, or wrap it in a struct", typ)
	}
}

// isArray reports whether typ is a fixed-size array, which is storage on the
// device and never a value that moves.
func isArray(typ types.Type) bool {
	_, ok := typ.Underlying().(*types.Array)
	return ok
}

// zeroValue is the C initialiser for a declaration Go wrote without one.
//
// Go's `var x T` is always the zero value; C's is whatever was in the storage.
// A scalar keeps the bare 0 the existing kernels are already generated with --
// changing it to 0.0f would rewrite a golden for no gain and bury the real
// diff -- while an array or a struct takes the empty braces C++ value-
// initialises from, because `float taps[4] = 0;` is not a thing.
func (t *transpiler) zeroValue(typ types.Type) string {
	switch typ.Underlying().(type) {
	case *types.Array, *types.Struct:
		return "{}"
	}
	return "0"
}

// ctype maps a Go type to its device counterpart.
//
// Go's int is 64-bit while CUDA's int is 32-bit. Kernel indices are bounded by
// the grid, so the narrowing is safe for a value passed by itself, and it keeps
// generated code idiomatic; it is the one deliberate infidelity in the mapping.
// It is only safe there, though -- see ctypeElem, which is what every position
// that has a memory layout to agree about goes through instead.
//
// The 64-bit types are spelled "long long" and never "long", because C's long
// is 8 bytes on Linux and 4 on Windows. Windows support is still open in the
// plan, and a type whose width depends on a host nobody has tried yet is a bug
// waiting for the machine that would find it.
func (t *transpiler) ctype(typ types.Type, pos token.Pos) string {
	if named, ok := typ.(*types.Named); ok {
		if _, isStruct := named.Underlying().(*types.Struct); isStruct {
			return t.structType(named, pos)
		}
	}
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
	case types.Float64:
		if !t.allowFloat64 {
			t.fail(pos, "float64 needs %s on kernel %s, or on its file's package comment: the device runs double at a fraction of the float32 rate, so it is opt-in", Float64Directive, t.kernelName)
			return "void"
		}
		return "double"
	case types.Int, types.Int32, types.UntypedInt:
		return "int"
	case types.Int64:
		return "long long"
	case types.Uint32:
		return "unsigned int"
	case types.Uint64:
		return "unsigned long long"
	case types.Bool, types.UntypedBool:
		return "bool"
	}
	switch basic.Kind() {
	case types.Int8, types.Int16, types.Uint8, types.Uint16:
		// Storage only, and this is the position that is not storage. The
		// widths match exactly; the arithmetic does not. Go computes int8*int8
		// in 8 bits and wraps, while C promotes both to int, computes in 32 and
		// truncates only at the assignment, so `a, b := int8(100), int8(3);
		// a*b/2` is 22 in Go and -106 in C. A variable, a by-value parameter or
		// a result exists in order to be computed with, so one of these here
		// would be an invitation to write exactly that expression -- whereas a
		// slice element, an array element or a struct field is a place bytes
		// live, which ctypeElem does accept.
		t.fail(pos, "%s is storage only on the device: it may be a slice element, an array element or a struct field, but not a variable, a parameter or a result, because Go computes %s arithmetic in %d bits and C promotes it to int; hold the value in an int32 and convert with %s(...) when you store it", typ, typ, 8*t.sizes.Sizeof(basic), typ)
		return "void"
	case types.Uint, types.Uintptr:
		// int gets the narrowing because an index is bounded by the grid.
		// Nothing indexes with uint, so there is no such argument here.
		t.fail(pos, "unsupported type %s on the device; use uint32 or uint64, which have a width the device shares", typ)
		return "void"
	}
	t.fail(pos, "unsupported type %s on the device (kernels are float32/float64, int32/int64, uint32/uint64 and bool)", typ)
	return "void"
}

// ctypeElem maps a Go type that will be laid out in memory -- a slice element,
// an array element, a struct field -- rather than passed as a value.
//
// The difference is int. As a parameter it is narrowed to C's 32-bit int and
// the loss is the documented infidelity above: BuildArgs passes it as an int32
// and the value is bounded by the grid anyway. As an *element* the narrowing is
// not a lost high word, it is a different stride: cuda.Upload copies 8 bytes
// per int and the kernel reads 4, so a []int lowers cleanly, compiles cleanly,
// launches cleanly and returns the wrong numbers. This was live until the
// commit that added this function.
func (t *transpiler) ctypeElem(typ types.Type, pos token.Pos) string {
	if basic, ok := typ.Underlying().(*types.Basic); ok {
		switch basic.Kind() {
		case types.Int, types.Uint, types.Uintptr:
			t.fail(pos, "%s cannot cross to the device: Go's %s is %d bytes and CUDA's int is 4, so the elements would not line up; use int32 or int64", typ, typ, t.sizes.Sizeof(basic))
			return "void"
		}
	}
	// The narrow integers are the mirror image of int: here the widths do line
	// up, so the bytes cross unchanged, and it is only arithmetic that the two
	// languages disagree about. An image buffer is the case that asks for this
	// -- holding a []uint8 as []int32 quadruples both the transfer and the
	// footprint to store numbers that fit in a byte -- so they are accepted
	// exactly where a layout is what is being described. Every operator on one
	// is refused (see narrowOperand), which is what keeps the disagreement from
	// ever being reachable.
	if c, ok := narrowCType(typ); ok {
		return c
	}
	return t.ctype(typ, pos)
}

// narrowCType spells the storage-only integers, and reports false for every
// other type.
//
// int8 becomes "signed char" and never "char": C leaves plain char's signedness
// to the implementation, and nvcc's is signed on x86 and unsigned on aarch64,
// so the one spelling that means int8 everywhere is the explicit one.
func narrowCType(typ types.Type) (string, bool) {
	basic, ok := typ.Underlying().(*types.Basic)
	if !ok {
		return "", false
	}
	switch basic.Kind() {
	case types.Int8:
		return "signed char", true
	case types.Uint8:
		return "unsigned char", true
	case types.Int16:
		return "short", true
	case types.Uint16:
		return "unsigned short", true
	}
	return "", false
}

// boundsHelper is the name of the range check WithBoundsChecks emits, and
// boundsHelperDef is its definition.
//
// Four things about this shape are deliberate.
//
// It is a function rather than a macro or a conditional expression, because
// the index may have side effects: a[f(i)] is in the subset, and any form that
// names the index twice would call f twice. One parameter, one evaluation.
//
// The index is long long rather than int. Go allows any integer type as an
// index and the subset allows four of them, and one signed 64-bit parameter
// takes all four by promotion with neither a narrowing nor a second overload.
// It also removes a whole warning class: written inline, `i < 0` on an
// unsigned index is a "comparison is always false" warning, and NVRTC's
// warnings reach the caller through Kernel.Log. Inside the helper the
// parameter is signed, so nothing warns. The honest limit is that a uint64
// index above 2^63 arrives negative and traps -- which is the right answer for
// an index no allocation can hold, and sits under the int-narrowing
// infidelity that is already documented.
//
// The bound is long long too, so that a length which is a C int and one which
// is an array's extent compare the same way, with no implementation-defined
// conversion in between.
//
// __trap() and __forceinline__ are both used because both were measured
// declared with no header rather than assumed; see
// docs/toolchain.md#what-nvrtc-declares-with-no-header-included.
const boundsHelper = "gocuda_bounds"

const boundsHelperDef = `__device__ __forceinline__ long long gocuda_bounds(long long i, long long n)
{
	if (i < 0 || i >= n)
	{
		__trap();
	}
	return i;
}
`

// boundOf reports the C expression giving the number of elements in x, or ""
// when the emitter cannot name one.
//
// It never fails. A build option must not change which programs are accepted,
// so an indexable whose bound is unknown is emitted unchecked rather than
// refused -- which is why this exists at all instead of calling lengthOf, the
// one that reports a diagnostic. Through Transpile the empty case is
// unreachable, every slice in the subset being a parameter with a generated
// length; internal/lower is also driven by the analyzer over packages this
// emitter did not construct, which is where it earns its keep.
func (t *transpiler) boundOf(x ast.Expr) string {
	if typ := t.typeOf(x); typ != nil {
		if arr, ok := typ.Underlying().(*types.Array); ok {
			return strconv.FormatInt(arr.Len(), 10)
		}
	}
	if id, ok := unparen(x).(*ast.Ident); ok {
		obj := t.info.Uses[id]
		if obj == nil {
			obj = t.info.Defs[id]
		}
		if l, ok := t.lens[obj]; ok {
			return l
		}
	}
	return ""
}

// checked wraps a rendered subscript in the bounds helper, or returns it
// unchanged when this build asked for no checks or the bound is unknown.
//
// index is the subscript's syntax where the caller still has it, and nil where
// the subscript has already been lifted into a temporary. It is used only to
// recognise a constant index into an array, which go/types has already refused
// if it is out of range, so there is nothing left for a run-time check to say.
// A constant index into a *slice* is still checked: nobody knows the length.
func (t *transpiler) checked(x ast.Expr, index ast.Expr, rendered string) string {
	if !t.boundsChecks {
		return rendered
	}
	bound := t.boundOf(x)
	if bound == "" {
		return rendered
	}
	if index != nil && t.info.Types[index].Value != nil {
		if typ := t.typeOf(x); typ != nil {
			if _, ok := typ.Underlying().(*types.Array); ok {
				return rendered
			}
		}
	}
	t.usedBounds = true
	return fmt.Sprintf("%s(%s, %s)", boundsHelper, rendered, bound)
}
