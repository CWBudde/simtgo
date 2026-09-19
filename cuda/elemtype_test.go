package cuda

import (
	"errors"
	"math"
	"strings"
	"testing"
	"unsafe"
)

// These need no device and no build tag, which is the point of putting the
// rule in an untagged file: the one check that keeps a caller from storing a
// Go pointer where the collector cannot see it is verified on every CI run,
// not only on the machine with a GPU.

type plainStruct struct {
	A float32
	B int32
}

type withString struct {
	A float32
	S string
}

type nestedBad struct {
	A     float32
	Inner withString
}

type withArrayOfPointers struct {
	A  float32
	Ps [4]*int
}

func TestCheckElemRefusesWhatTheCollectorWouldFollow(t *testing.T) {
	for _, tc := range []struct {
		name string
		call func(string) error
		bad  bool
		path string // the member the error should name
		kind string
	}{
		{name: "float32", call: checkElem[float32]},
		{name: "int64", call: checkElem[int64]},
		{name: "bool", call: checkElem[bool]},
		{name: "uintptr", call: checkElem[uintptr]},
		{name: "array of float32", call: checkElem[[8]float32]},
		{name: "plain struct", call: checkElem[plainStruct]},
		{name: "array of plain structs", call: checkElem[[2]plainStruct]},

		{name: "pointer", call: checkElem[*int], bad: true, kind: "ptr"},
		{name: "string", call: checkElem[string], bad: true, kind: "string"},
		{name: "slice", call: checkElem[[]byte], bad: true, kind: "slice"},
		{name: "map", call: checkElem[map[int]int], bad: true, kind: "map"},
		{name: "interface", call: checkElem[any], bad: true, kind: "interface"},
		{name: "unsafe.Pointer", call: checkElem[unsafe.Pointer], bad: true, kind: "unsafe.Pointer"},
		{name: "func", call: checkElem[func()], bad: true, kind: "func"},
		{name: "chan", call: checkElem[chan int], bad: true, kind: "chan"},

		{name: "struct with a string", call: checkElem[withString], bad: true, path: "S", kind: "string"},
		{name: "struct within a struct", call: checkElem[nestedBad], bad: true, path: "Inner.S", kind: "string"},
		// The path says Ps[] rather than Ps: it is the array's elements that
		// carry the pointer, and a reader looking for a field called Ps would
		// otherwise find one that is fine by itself.
		{name: "array of pointers in a struct", call: checkElem[withArrayOfPointers], bad: true, path: "Ps[]", kind: "ptr"},

		// Not about pointers, but the same refusal: there is no allocation to
		// make, so a length would have no memory behind it.
		{name: "empty struct", call: checkElem[struct{}], bad: true, kind: "zero-sized"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.call("NewHostSlice")
			if !tc.bad {
				if err != nil {
					t.Fatalf("refused a pointer-free type: %v", err)
				}
				return
			}
			var e *ElementTypeError
			if !errors.As(err, &e) {
				t.Fatalf("got %v, want an *ElementTypeError", err)
			}
			if e.Kind != tc.kind {
				t.Errorf("Kind = %q, want %q", e.Kind, tc.kind)
			}
			if tc.path != "" && e.Path != tc.path {
				t.Errorf("Path = %q, want %q -- the message has to say where the pointer is", e.Path, tc.path)
			}
			// The message is the whole value of the check: somebody reading it
			// should not have to come here to learn why.
			if msg := e.Error(); !strings.Contains(msg, "NewHostSlice") && !strings.Contains(msg, e.Op) {
				t.Errorf("the message does not name the operation: %s", msg)
			}
		})
	}
}

// TestCheckExtentRefusesWhatWouldWrap covers the multiplication rather than
// the type. A product that wraps to a small positive number is the dangerous
// one: the allocation succeeds at the wrong size and Slice() then reports
// every element the caller asked for.
func TestCheckExtentRefusesWhatWouldWrap(t *testing.T) {
	if err := checkExtent[float32]("NewHostSlice", 1<<20); err != nil {
		t.Errorf("refused an ordinary length: %v", err)
	}
	if err := checkExtent[float32]("NewHostSlice", 0); err != nil {
		t.Errorf("refused zero: %v", err)
	}

	var e *LengthError
	if err := checkExtent[float32]("NewHostSlice", -1); !errors.As(err, &e) {
		t.Errorf("negative length: got %v, want a *LengthError", err)
	}

	// Four bytes an element, so anything past a quarter of the int range
	// cannot be expressed as a byte count.
	if err := checkExtent[float32]("NewHostSlice", math.MaxInt/4+1); err == nil {
		t.Error("accepted a length whose byte size overflows an int")
	}
	if err := checkExtent[float32]("NewHostSlice", math.MaxInt/4); err != nil {
		t.Errorf("refused the largest expressible length: %v", err)
	}
	// A wider element type has to bite sooner, which is what makes the check
	// about the product and not about a constant.
	if err := checkExtent[[16]float64]("NewHostSlice", math.MaxInt/4); err == nil {
		t.Error("accepted a length that overflows for a 128-byte element")
	}
}
