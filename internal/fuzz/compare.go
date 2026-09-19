package fuzz

import (
	"fmt"
	"reflect"

	"github.com/CWBudde/simtgo/internal/tolerance"
)

// A Mismatch is one element two backends disagreed about.
//
// It names the parameter rather than its index because that is what the
// generated source calls it, so a failure can be read against the source the
// same report prints.
type Mismatch struct {
	Param string
	Index int
	Got   any
	Want  any
}

func (m Mismatch) String() string {
	return fmt.Sprintf("%s[%d] = %v, want %v", m.Param, m.Index, m.Got, m.Want)
}

// maxMismatches is how many elements a comparison reports before it stops
// looking.
//
// A kernel that is wrong is usually wrong everywhere, and a report of sixty
// thousand differing elements is not sixty thousand times as informative as a
// report of the first few -- it is less informative, because nobody reads it.
// The count that matters is whether there were any.
const maxMismatches = 8

// compareStructs compares a struct buffer field by field.
//
// It cannot go through reflect.DeepEqual, which is what the rest of this falls
// back on, because DeepEqual compares a float with == and a NaN is equal to
// nothing -- so two buffers holding the same NaN would be reported as
// differing, with a message that printed the two as identical. The fuzzer
// produced exactly that, which is the third time the oracle rather than the
// emitter has been the thing at fault; asking tolerance.Agree per field is
// what the rest of the comparison already does.
func compareStructs(spec ParamSpec, got, want any, tol float64) []Mismatch {
	shape := spec.Shape
	var out []Mismatch
	n := shape.length(got)
	if m := shape.length(want); m != n {
		return []Mismatch{{Param: spec.Name, Index: -1,
			Got: fmt.Sprintf("%d elements", n), Want: fmt.Sprintf("%d elements", m)}}
	}
	for i := range n {
		for _, f := range shape.Fields {
			g, w := fieldValue(f, got, i), fieldValue(f, want, i)
			if tolerance.Agree(g, w, tol) {
				continue
			}
			// The field path carries no index: String appends one, and a
			// path that spelled it too would report p[3].B as p[3].B[3].
			out = append(out, Mismatch{Param: spec.Name + "." + f.Name,
				Index: i, Got: g, Want: w})
			if len(out) >= maxMismatches {
				return out
			}
		}
	}
	return out
}

// fieldValue reads one field as an any, so that tolerance.Agree can judge it
// by its own type rather than by the struct's.
func fieldValue(f StructField, buf any, i int) any {
	switch f.Kind {
	case KF32:
		return f.get.(func(any, int) float32)(buf, i)
	case KF64:
		return f.get.(func(any, int) float64)(buf, i)
	case KI32:
		return f.get.(func(any, int) int32)(buf, i)
	case KI64:
		return f.get.(func(any, int) int64)(buf, i)
	case KU32:
		return f.get.(func(any, int) uint32)(buf, i)
	case KU64:
		return f.get.(func(any, int) uint64)(buf, i)
	case KBool:
		return f.get.(func(any, int) bool)(buf, i)
	case KI8:
		return f.get.(func(any, int) int8)(buf, i)
	case KI16:
		return f.get.(func(any, int) int16)(buf, i)
	case KU8:
		return f.get.(func(any, int) uint8)(buf, i)
	case KU16:
		return f.get.(func(any, int) uint16)(buf, i)
	}
	panic("fuzz: cannot compare a struct field of kind " + f.Kind.goName())
}

// Compare reports where two runs of this program disagree.
//
// Only the buffers are compared. A scalar parameter is passed by value and
// cannot come back changed, and a buffer two read-only parameters share is
// compared once rather than once per parameter, so a single wrong element is
// reported once and not twice.
//
// Every buffer is compared, written and read-only alike. A read-only one that
// differs is a finding of a different kind -- something wrote through a
// pointer the generated C declared const -- and losing it to save a loop over
// a buffer nothing was supposed to touch would be a poor trade.
//
// tol is the bound for float32 elements, and internal/tolerance.Agree is what
// applies it: bit equality first, the values no tolerance relates settled
// before any bound is consulted. Pass 0 to demand bit equality throughout,
// which is what the two CPU-side backends should manage.
func (p *Program) Compare(got, want *Args, tol float64) []Mismatch {
	var out []Mismatch
	for i, spec := range p.Params {
		if !spec.Slice || spec.AliasOf >= 0 {
			continue
		}
		if spec.Shape != nil {
			out = append(out, compareStructs(spec, got.Vals[i], want.Vals[i], tol)...)
			if len(out) >= maxMismatches {
				return out[:maxMismatches]
			}
			continue
		}
		g, w := reflect.ValueOf(got.Vals[i]), reflect.ValueOf(want.Vals[i])
		if g.Len() != w.Len() {
			// Not something a backend can cause -- both sides were cloned from
			// one Args -- so it means the caller paired up the wrong runs, and
			// saying so beats reporting every element as different.
			out = append(out, Mismatch{Param: spec.Name, Index: -1,
				Got: fmt.Sprintf("%d elements", g.Len()), Want: fmt.Sprintf("%d elements", w.Len())})
			continue
		}
		for j := range g.Len() {
			gj, wj := g.Index(j).Interface(), w.Index(j).Interface()
			if tolerance.Agree(gj, wj, tol) {
				continue
			}
			out = append(out, Mismatch{Param: spec.Name, Index: j, Got: gj, Want: wj})
			if len(out) >= maxMismatches {
				return out
			}
		}
	}
	return out
}
