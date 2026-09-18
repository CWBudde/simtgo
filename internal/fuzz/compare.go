package fuzz

import (
	"fmt"
	"reflect"

	"github.com/CWBudde/gocuda/internal/tolerance"
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
