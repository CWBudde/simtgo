package lower

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"strings"
)

// structType registers named as a C struct, once, and reports the C name a use
// of it is spelled with.
//
// Registration is lazy and in first-mention order, which is deterministic for a
// given source -- SourceHash depends on it, so "whatever order the map yields"
// would make an artifact unfindable from one build to the next.
//
// The body is built into a buffer and only appended once every field has been
// rendered without complaint. A half-emitted struct must never reach the
// output, for the same reason Kernel returns (nil, diags) rather than partial
// CUDA: C that mentions a type it never defined is an error about code the
// author did not write.
func (t *transpiler) structType(named *types.Named, pos token.Pos) string {
	if name, ok := t.structNames[named]; ok {
		return name
	}
	st, ok := named.Underlying().(*types.Struct)
	if !ok {
		t.fail(pos, "unsupported type %s on the device", named)
		return "void"
	}
	// cname, because a Go type may be called Class or Template, which C++ would
	// read as a keyword.
	name := cname(named.Obj().Name())
	// A struct tag shares the ordinary namespace in C++, so a type spelled
	// like one of the emitter's own file-scope names collides with it.
	t.checkReserved(name, named.Obj().Pos())

	// Reserved before the fields are walked so that a struct reachable from one
	// of its own fields cannot recurse forever. Go needs a pointer to express
	// that and pointers are not in the subset, so this is belt and braces.
	if t.structNames == nil {
		t.structNames = map[*types.Named]string{}
	}
	t.structNames[named] = name

	before := len(t.diags)
	var b strings.Builder
	fmt.Fprintf(&b, "struct %s\n{\n", name)
	fields := make([]*types.Var, st.NumFields())
	for i := range fields {
		fields[i] = st.Field(i)
	}
	// Where Go puts each field, which is the thing the C++ compiler has to be
	// made to agree with. See padding below for what is done with it.
	offsets := t.sizes.Offsetsof(fields)
	layout := structLayout{padBefore: make([]bool, len(fields))}
	next, pads := int64(0), 0

	for i, f := range fields {
		if f.Embedded() {
			// A promoted field is selected through a path rather than a name,
			// and flattening that buys nothing a named field does not.
			t.fail(f.Pos(), "%s has an embedded field %s; a device struct needs plain named fields", named.Obj().Name(), f.Name())
			continue
		}
		if f.Name() == "_" {
			// Go's blank field is padding: it takes up layout and nothing can
			// read it. C++ has no such name, and `_` is an ordinary identifier
			// there, so two of them in one struct emitted two `int _;` members
			// and NVRTC rejected the redeclaration. Naming it costs the author
			// one word and keeps the layout the assertions below pin.
			t.fail(f.Pos(), "%s has a blank field; the device struct would need a name for it, so call it Pad or similar", named.Obj().Name())
			continue
		}
		if gap := offsets[i] - next; gap > 0 {
			fmt.Fprintf(&b, "\tunsigned char %s%d[%d];\n", padPrefix, pads, gap)
			layout.padBefore[i], pads = true, pads+1
		}
		next = offsets[i] + t.sizes.Sizeof(f.Type())
		fmt.Fprintf(&b, "\t%s;\n", t.cfield(named, f))
	}
	if gap := t.sizes.Sizeof(named) - next; gap > 0 {
		fmt.Fprintf(&b, "\tunsigned char %s%d[%d];\n", padPrefix, pads, gap)
		layout.padTail = true
	}
	b.WriteString("};\n")

	// What Go believes about the layout, stated for the C++ compiler to check
	// against what it believes. The emitter never models the C ABI: it asserts
	// Go's numbers and lets NVRTC refuse them, which makes the guarantee hold
	// on every architecture and CUDA version rather than on this one.
	//
	// Together with the padding above, sizeof is the offset check. NVRTC has no
	// offsetof to assert one directly -- it compiles a string with no include
	// path, so there is no <cstddef>, and __builtin_offsetof is an nvcc
	// spelling its frontend does not define; both were measured again against
	// NVRTC 12.9 rather than taken on trust. What is left is an argument about
	// sizes. Every hole in Go's layout, and the trailing one, is declared as an
	// unsigned char array, so the members this struct declares occupy exactly
	// the number of bytes Go's struct does. C++ lays members out in declaration
	// order, each at or after the end of the one before it, so if the compiler
	// inserted a single byte anywhere the last member would end past Go's size
	// and sizeof would exceed it. sizeof agreeing therefore means nothing was
	// inserted, which means every field sits at the offset Go gave it.
	//
	// Padding is what makes that argument available, and nothing else here
	// would: alignas can only raise a field's alignment, so it cannot describe
	// a hole; a pack pragma would change the struct's alignment and take the
	// assertion below with it. An unsigned char array has alignment 1, so it
	// adds nothing to alignof and the second assertion still means what it
	// always meant.
	//
	// A struct whose Go layout has no holes and no trailing slack emits nothing
	// new, which is most of them -- the two structs committed in kernels/ are
	// both in that case -- so this does not churn the generated C to say
	// something it was already saying.
	fmt.Fprintf(&b, "static_assert(sizeof(%s) == %d, \"gocuda: %s is a different size in CUDA than in Go\");\n",
		name, t.sizes.Sizeof(named), named.Obj().Name())
	fmt.Fprintf(&b, "static_assert(alignof(%s) == %d, \"gocuda: %s is differently aligned in CUDA than in Go\");\n",
		name, t.sizes.Alignof(named), named.Obj().Name())

	if len(t.diags) > before {
		return "void"
	}
	if t.structLayouts == nil {
		t.structLayouts = map[*types.Named]structLayout{}
	}
	t.structLayouts[named] = layout
	t.structDefs = append(t.structDefs, b.String())
	return name
}

// padPrefix names the members gocuda emits to fill Go's holes. A field spelled
// this way is refused rather than renamed, because two members with one name is
// an NVRTC error about generated code and a silently renamed field is worse.
const padPrefix = "gocuda_pad"

// structLayout records where those members sit, so that a positional literal
// can step over them. C++17 has no designated initialisers -- which is why
// composite writes every field out in order -- so the padding has to be
// initialised by position too, and the emitter is the only thing that knows
// where it went.
type structLayout struct {
	padBefore []bool // one entry per Go field
	padTail   bool
}

// cfield renders one struct field's declaration.
//
// A field is a layout position, so it goes through ctypeElem rather than ctype:
// an int field is 8 bytes in Go and 4 in C, and unlike an int parameter there
// is nothing to narrow it at -- cuda.Upload would copy Go's offsets into a
// struct C reads with its own.
//
// An array field goes through cdecl, because C puts the extent after the name.
// It was refused until the padding above existed: a [4]float32 member lays out
// identically in both languages, but nothing could say where inside the struct
// it began, and a field the assertions cannot reach is a field that can quietly
// move. It can be reached now, so the rule that stood in for the check goes.
func (t *transpiler) cfield(named *types.Named, f *types.Var) string {
	name := cname(f.Name())
	if strings.HasPrefix(name, padPrefix) {
		t.fail(f.Pos(), "field %s of %s is spelled like the padding gocuda emits to pin the field offsets; rename it", f.Name(), named.Obj().Name())
		return ""
	}
	if isArray(f.Type()) {
		return t.cdecl(f.Type(), name, f.Pos())
	}
	return t.ctypeElem(f.Type(), f.Pos()) + " " + name
}

// selector renders a field access.
//
// Only a direct field of a struct value: Kind is checked rather than assumed,
// and a selection through an embedded field has an index path longer than one,
// which the struct rule refuses at the declaration anyway.
func (t *transpiler) selector(e *ast.SelectorExpr) cexpr {
	sel := t.info.Selections[e]
	if sel == nil || sel.Kind() != types.FieldVal {
		if sel != nil && sel.Kind() == types.MethodVal {
			t.fail(e.Pos(), "methods are not supported in kernels; %s cannot be called on the device", e.Sel.Name)
			return atom("")
		}
		t.fail(e.Pos(), "unsupported selector %s", e.Sel.Name)
		return atom("")
	}
	if len(sel.Index()) != 1 {
		t.fail(e.Pos(), "%s is promoted through an embedded field, which a device struct cannot have", e.Sel.Name)
		return atom("")
	}
	// Registering here as well as at the declaration means a struct first met
	// through a field access is still defined before the code that reads it.
	if named, ok := t.typeOf(e.X).(*types.Named); ok {
		t.structType(named, e.Pos())
	}
	return cexpr{fmt.Sprintf("%s.%s", t.expr(e.X).at(precPostfix), cname(e.Sel.Name)), precPostfix}
}

// composite renders a struct literal.
//
// Always positionally, in field order, with every field written out. Designated
// initialisers (`.gain = 2.0f`) are C++20 and NVRTC defaults to C++17, and
// leaning on C++'s value-initialisation of the fields left off would be relying
// on a rule for no reason when writing them costs one loop.
func (t *transpiler) composite(e *ast.CompositeLit) cexpr {
	typ := t.typeOf(e)
	if typ == nil {
		return atom("")
	}
	named, ok := typ.(*types.Named)
	if !ok {
		t.fail(e.Pos(), "only a named struct type can be written as a literal in a kernel")
		return atom("")
	}
	st, ok := named.Underlying().(*types.Struct)
	if !ok {
		t.fail(e.Pos(), "unsupported literal of type %s", typ)
		return atom("")
	}
	name := t.structType(named, e.Pos())
	if t.failed() {
		return atom("")
	}

	values := make([]string, st.NumFields())
	byName := map[string]int{}
	for i := range values {
		f := st.Field(i)
		byName[f.Name()] = i
		values[i] = t.zeroValue(f.Type())
	}

	keyed := false
	for i, elt := range e.Elts {
		if kv, ok := elt.(*ast.KeyValueExpr); ok {
			keyed = true
			key, ok := kv.Key.(*ast.Ident)
			if !ok {
				t.fail(kv.Pos(), "unsupported field name in a struct literal")
				return atom("")
			}
			at, ok := byName[key.Name]
			if !ok {
				t.fail(kv.Pos(), "%s has no field %s", named.Obj().Name(), key.Name)
				return atom("")
			}
			values[at] = t.fieldValue(st.Field(at), kv.Value)
			continue
		}
		if keyed {
			// Go rejects this itself; saying so here would be a second opinion
			// about a program that never type-checked.
			return atom("")
		}
		if i >= len(values) {
			t.fail(elt.Pos(), "too many values for %s", named.Obj().Name())
			return atom("")
		}
		values[i] = t.fieldValue(st.Field(i), elt)
	}
	return cexpr{fmt.Sprintf("%s{%s}", name, strings.Join(t.withPadding(named, values), ", ")), precPostfix}
}

// withPadding interleaves an empty initialiser for each padding member the
// struct carries, so that a positional literal lines up with the C declaration
// rather than putting the first field's value into a hole.
func (t *transpiler) withPadding(named *types.Named, values []string) []string {
	layout, ok := t.structLayouts[named]
	if !ok {
		return values
	}
	out := make([]string, 0, len(values)+len(layout.padBefore)+1)
	for i, v := range values {
		if i < len(layout.padBefore) && layout.padBefore[i] {
			out = append(out, "{}")
		}
		out = append(out, v)
	}
	if layout.padTail {
		out = append(out, "{}")
	}
	return out
}

// fieldValue renders one field's initialiser.
func (t *transpiler) fieldValue(_ *types.Var, e ast.Expr) string {
	return t.expr(e).s
}
