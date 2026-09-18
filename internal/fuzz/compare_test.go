package fuzz

import (
	"math"
	"testing"
)

// shapeNamed finds one catalogue shape, so a test can name the layout it means
// rather than index into a list whose order is free to change.
func shapeNamed(t *testing.T, name string) *StructShape {
	t.Helper()
	for _, s := range Shapes {
		if s.Name == name {
			return s
		}
	}
	t.Fatalf("no shape named %s in the catalogue", name)
	return nil
}

// pairBuf is a []SPair of n elements with A = i and B = NaN.
//
// The NaN is the point: it is what reflect.DeepEqual gets wrong, since a NaN
// is equal to nothing including itself.
func pairBuf(shape *StructShape, n int) any {
	setA := shape.Fields[0].set.(func(any, int, int32))
	setB := shape.Fields[1].set.(func(any, int, float32))
	b := shape.make(n)
	for i := range n {
		setA(b, i, int32(i))
		setB(b, i, float32(math.NaN()))
	}
	return b
}

// TestCompareStructsIsFieldByField pins the two things the struct comparison
// exists for, both of which the fuzzer found the hard way.
//
// A buffer holding a NaN must compare equal to itself -- reflect.DeepEqual
// says it does not, and reported the two as differing with a message that
// printed them as identical -- and a difference must name the field it is in,
// once. The report is a string a person reads against the generated source, so
// what it says is part of what is being tested.
func TestCompareStructsIsFieldByField(t *testing.T) {
	shape := shapeNamed(t, "SPair")
	spec := ParamSpec{Name: "p", Kind: KStruct, Slice: true, Len: 4, AliasOf: -1, Shape: shape}
	p := &Program{Params: []ParamSpec{spec}}
	setB := shape.Fields[1].set.(func(any, int, float32))

	t.Run("a NaN equals itself", func(t *testing.T) {
		got, want := pairBuf(shape, 4), pairBuf(shape, 4)
		if ms := p.Compare(&Args{Vals: []any{got}}, &Args{Vals: []any{want}}, 0); len(ms) != 0 {
			t.Errorf("two identical buffers differ: %v", ms)
		}
	})

	t.Run("a difference names its field once", func(t *testing.T) {
		got, want := pairBuf(shape, 4), pairBuf(shape, 4)
		setB(got, 3, 7)
		setB(want, 3, 8)
		ms := p.Compare(&Args{Vals: []any{got}}, &Args{Vals: []any{want}}, 0)
		if len(ms) != 1 {
			t.Fatalf("got %d mismatches, want 1: %v", len(ms), ms)
		}
		// The element index belongs to String, not to Param: writing it in
		// both is how p[3].B once got reported as p[3].B[3].
		if ms[0].Param != "p.B" || ms[0].Index != 3 {
			t.Errorf("Param %q, Index %d; want \"p.B\" and 3", ms[0].Param, ms[0].Index)
		}
		if s := ms[0].String(); s != "p.B[3] = 7, want 8" {
			t.Errorf("String() = %q", s)
		}
	})

	t.Run("a length difference is not an element difference", func(t *testing.T) {
		ms := p.Compare(&Args{Vals: []any{pairBuf(shape, 4)}}, &Args{Vals: []any{pairBuf(shape, 3)}}, 0)
		if len(ms) != 1 || ms[0].Index != -1 || ms[0].Param != "p" {
			t.Fatalf("got %v, want one mismatch on p with index -1", ms)
		}
	})

	t.Run("a narrow field is compared too", func(t *testing.T) {
		tail := shapeNamed(t, "STail")
		spec := ParamSpec{Name: "q", Kind: KStruct, Slice: true, Len: 2, AliasOf: -1, Shape: tail}
		q := &Program{Params: []ParamSpec{spec}}
		set := tail.Fields[1].set.(func(any, int, uint16))
		got, want := tail.make(2), tail.make(2)
		set(got, 1, 9)
		ms := q.Compare(&Args{Vals: []any{got}}, &Args{Vals: []any{want}}, 0)
		if len(ms) != 1 || ms[0].Param != "q.B" || ms[0].Index != 1 {
			t.Fatalf("got %v, want one mismatch on q.B[1]", ms)
		}
	})
}
