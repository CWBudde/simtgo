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
		f := st.Field(i)
		fields[i] = f
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
		fmt.Fprintf(&b, "\t%s;\n", t.cfield(named, f))
	}
	b.WriteString("};\n")

	// What Go believes about the layout, stated for the C++ compiler to check
	// against what it believes. The emitter never models the C ABI: it asserts
	// Go's numbers and lets NVRTC refuse them, which makes the guarantee hold
	// on every architecture and CUDA version rather than on this one.
	//
	// Only size and alignment, because NVRTC compiles a string with no include
	// path: it has no <cstddef> and so no offsetof, and __builtin_offsetof is
	// an nvcc spelling its frontend does not define. Field offsets are pinned
	// by a round-trip test instead, which exercises the cuda.Upload copy path
	// the assertion could only have had an opinion about.
	fmt.Fprintf(&b, "static_assert(sizeof(%s) == %d, \"gocuda: %s is a different size in CUDA than in Go\");\n",
		name, t.sizes.Sizeof(named), named.Obj().Name())
	fmt.Fprintf(&b, "static_assert(alignof(%s) == %d, \"gocuda: %s is differently aligned in CUDA than in Go\");\n",
		name, t.sizes.Alignof(named), named.Obj().Name())

	if len(t.diags) > before {
		return "void"
	}
	t.structDefs = append(t.structDefs, b.String())
	return name
}

// cfield renders one struct field's declaration.
//
// A field is a layout position, so it goes through ctypeElem rather than ctype:
// an int field is 8 bytes in Go and 4 in C, and unlike an int parameter there
// is nothing to narrow it at -- cuda.Upload would copy Go's offsets into a
// struct C reads with its own.
//
// An array field is refused for now. The plan asks for structs of scalars, and
// while a [4]float32 member would lay out identically in both languages, the
// size and alignment assertions below cannot pin where inside the struct it
// begins. Lifting that belongs with the offset checking, not here.
func (t *transpiler) cfield(named *types.Named, f *types.Var) string {
	if isArray(f.Type()) {
		t.fail(f.Pos(), "field %s of %s is an array; a device struct holds scalars, so pass the array as a slice instead", f.Name(), named.Obj().Name())
		return ""
	}
	name := cname(f.Name())
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
	return cexpr{fmt.Sprintf("%s{%s}", name, strings.Join(values, ", ")), precPostfix}
}

// fieldValue renders one field's initialiser.
func (t *transpiler) fieldValue(_ *types.Var, e ast.Expr) string {
	return t.expr(e).s
}
