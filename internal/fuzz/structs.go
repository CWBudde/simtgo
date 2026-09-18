package fuzz

// The struct shapes the generator may use, and the machinery for reaching one
// field of one element.
//
// # Why a fixed catalogue
//
// Everything else in this package is generated freely, so a fixed list of four
// shapes wants justifying. The closure renderer has to hold *real values* --
// that is the whole reason it is trustworthy, being Go running the same
// operations on the same types rather than an interpreter of Go semantics --
// and Go cannot make a type at run time. A shape invented per program could
// therefore be rendered as source but not as a closure, and the two renderings
// are the point.
//
// So the shapes are declared here as ordinary Go types, the generated source
// declares a type with the same fields in the same order, and the two have the
// same layout because Go computed both.
//
// # Why these four
//
// They are chosen for their holes, which is what a struct is here to test. The
// emitter emits every hole as a gocuda_padN member and asserts the whole
// struct's size, and that assertion is what stands in for the field offsets
// NVRTC has no offsetof to check. A shape with no padding would exercise none
// of it.
//
//   - SPair has no hole at all, and is the control.
//   - SHole pads three bytes after an int8, the commonest shape.
//   - SWide pads four to reach an alignment of eight, which is the case a
//     32-bit host would get wrong.
//   - STail pads six at the *end*, which is the one the emitter had to be
//     taught about: a trailing hole changes sizeof without changing any field
//     offset, so nothing but the size assertion can see it.
//
// The narrow fields are deliberate. int8 and uint16 are storage-only in the
// subset -- legal in a struct and accepted by no operator -- so a struct is
// where the ctype/ctypeElem split is actually exercised rather than described.

// SPair has no padding: two four-byte fields.
type SPair struct {
	A int32
	B float32
}

// SHole pads three bytes between its fields.
type SHole struct {
	A int8
	B float32
}

// SWide pads four bytes, the int64 forcing an alignment of eight.
type SWide struct {
	A float32
	B int64
}

// STail pads six bytes after its last field, which changes sizeof without
// moving anything.
type STail struct {
	A int64
	B uint16
}

// A StructField is one field: what to call it, what it holds, and how to reach
// it in a buffer of its shape.
//
// get and set are typed closures stored as any -- func(any, int) int32 and
// func(any, int, int32), say -- because a Go field is reached by a different
// expression for every (shape, field) pair and there is no way to write that
// once. They are asserted to their type when the closure is *compiled*, not
// when it runs, so the frame stays unboxed in the loop, which is the property
// closure.go's frame exists to have.
type StructField struct {
	Name string
	Kind Kind
	get  any
	set  any
}

// A StructShape is one of the catalogue's types.
type StructShape struct {
	Name   string
	Fields []StructField
	// make and clone keep the real Go type inside this file: everywhere else
	// holds a struct buffer as an any and never names the type.
	make   func(n int) any
	clone  func(buf any) any
	length func(buf any) int
}

// Shapes is the catalogue, in a fixed order so that a seed keeps meaning the
// same program as it grows.
var Shapes = []*StructShape{{
	Name: "SPair",
	Fields: []StructField{
		{Name: "A", Kind: KI32,
			get: func(b any, i int) int32 { return b.([]SPair)[i].A },
			set: func(b any, i int, v int32) { b.([]SPair)[i].A = v }},
		{Name: "B", Kind: KF32,
			get: func(b any, i int) float32 { return b.([]SPair)[i].B },
			set: func(b any, i int, v float32) { b.([]SPair)[i].B = v }},
	},
	make:   func(n int) any { return make([]SPair, n) },
	clone:  func(b any) any { s := b.([]SPair); out := make([]SPair, len(s)); copy(out, s); return out },
	length: func(b any) int { return len(b.([]SPair)) },
}, {
	Name: "SHole",
	Fields: []StructField{
		{Name: "A", Kind: KI8,
			get: func(b any, i int) int8 { return b.([]SHole)[i].A },
			set: func(b any, i int, v int8) { b.([]SHole)[i].A = v }},
		{Name: "B", Kind: KF32,
			get: func(b any, i int) float32 { return b.([]SHole)[i].B },
			set: func(b any, i int, v float32) { b.([]SHole)[i].B = v }},
	},
	make:   func(n int) any { return make([]SHole, n) },
	clone:  func(b any) any { s := b.([]SHole); out := make([]SHole, len(s)); copy(out, s); return out },
	length: func(b any) int { return len(b.([]SHole)) },
}, {
	Name: "SWide",
	Fields: []StructField{
		{Name: "A", Kind: KF32,
			get: func(b any, i int) float32 { return b.([]SWide)[i].A },
			set: func(b any, i int, v float32) { b.([]SWide)[i].A = v }},
		{Name: "B", Kind: KI64,
			get: func(b any, i int) int64 { return b.([]SWide)[i].B },
			set: func(b any, i int, v int64) { b.([]SWide)[i].B = v }},
	},
	make:   func(n int) any { return make([]SWide, n) },
	clone:  func(b any) any { s := b.([]SWide); out := make([]SWide, len(s)); copy(out, s); return out },
	length: func(b any) int { return len(b.([]SWide)) },
}, {
	Name: "STail",
	Fields: []StructField{
		{Name: "A", Kind: KI64,
			get: func(b any, i int) int64 { return b.([]STail)[i].A },
			set: func(b any, i int, v int64) { b.([]STail)[i].A = v }},
		{Name: "B", Kind: KU16,
			get: func(b any, i int) uint16 { return b.([]STail)[i].B },
			set: func(b any, i int, v uint16) { b.([]STail)[i].B = v }},
	},
	make:   func(n int) any { return make([]STail, n) },
	clone:  func(b any) any { s := b.([]STail); out := make([]STail, len(s)); copy(out, s); return out },
	length: func(b any) int { return len(b.([]STail)) },
}}

// writable reports whether a field can be stored into by generated code.
//
// A narrow field cannot: no operator in the subset accepts one, so the only
// value that could be stored is a conversion, and the generator would be
// writing `p[i].A = int8(x)` -- which is legal but says nothing the slice case
// does not. Reading one through a conversion is the interesting direction, and
// that is what the generator does.
func (f StructField) writable() bool { return !f.Kind.narrow() }
