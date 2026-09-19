package cuda

import (
	"fmt"
	"reflect"
	"strings"
)

// What may be the element type of a buffer that is not Go memory.
//
// Slice and HostSlice are generic over any, which is more than either can
// honour. HostSlice is the sharper case and the reason this file exists: its
// memory comes from cuMemHostAlloc, the Go collector does not scan it, and
// Slice() hands back a []T over it. Store the only reference to a Go object
// in such a slice and nothing keeps that object alive -- the collector cannot
// see the reference, frees the target, and a later read through the slice
// dereferences reclaimed memory. There is no panic and no race detector
// report; the value is simply wrong, later, somewhere else.
//
// runtime.Pinner is not the answer. It pins one object for as long as the
// Pinner lives, and a buffer a caller fills at will has no such moment to
// hang it on -- the whole point of the type is that it outlives any single
// call.
//
// So the rule is the same one the transpiler follows: refuse, rather than
// accept something that cannot be made to work. It is checked once, at
// allocation, with reflect, which costs nothing against a driver call and
// happens off the device entirely -- which is also why it lives in an
// untagged file and is tested without a GPU.

// ElementTypeError reports an element type a device or page-locked buffer
// cannot hold.
type ElementTypeError struct {
	// Op is the operation that refused.
	Op string
	// Type is the element type, as Go spells it.
	Type string
	// Path names the member carrying the pointer, e.g. "Header.Name", or is
	// empty when the element type is itself the problem.
	Path string
	// Kind is what that member is, e.g. "string".
	Kind string
}

func (e *ElementTypeError) Error() string {
	where := "it"
	if e.Path != "" {
		where = e.Path
	}
	if e.Kind == "zero-sized" {
		return fmt.Sprintf("cuda: %s: %s has no storage to copy", e.Op, e.Type)
	}
	return fmt.Sprintf("cuda: %s: %s cannot be the element type of a buffer outside the Go heap: "+
		"%s is a %s, and the collector does not scan driver memory -- a pointer stored there keeps "+
		"nothing alive, so its target can be freed while the buffer still refers to it",
		e.Op, e.Type, where, e.Kind)
}

// checkElem refuses an element type that cannot live outside the Go heap.
//
// Zero-sized types are refused as well, and for a duller reason: an
// allocation of no bytes is not one the driver makes, so a HostSlice of a
// thousand empty structs would have to report either a length with no memory
// behind it or a Slice() that does not match its Len().
func checkElem[T any](op string) error {
	t := reflect.TypeFor[T]()
	if path, kind, bad := findPointer(t, ""); bad {
		return &ElementTypeError{Op: op, Type: t.String(), Path: path, Kind: kind}
	}
	if t.Size() == 0 {
		return &ElementTypeError{Op: op, Type: t.String(), Kind: "zero-sized"}
	}
	return nil
}

// findPointer walks a type for anything the garbage collector would follow,
// and names the first one it finds.
//
// The allowed set is written as the default rather than the exception, so a
// kind added to Go in future is refused until somebody has thought about it.
// uintptr is allowed: the collector does not follow one, which is exactly
// what makes it a number here rather than a reference.
func findPointer(t reflect.Type, path string) (string, string, bool) {
	switch t.Kind() {
	case reflect.Bool,
		reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
		reflect.Uintptr,
		reflect.Float32, reflect.Float64,
		reflect.Complex64, reflect.Complex128:
		return "", "", false
	case reflect.Array:
		return findPointer(t.Elem(), path+"[]")
	case reflect.Struct:
		for i := range t.NumField() {
			f := t.Field(i)
			if p, kind, bad := findPointer(f.Type, join(path, f.Name)); bad {
				return p, kind, true
			}
		}
		return "", "", false
	default:
		return path, t.Kind().String(), true
	}
}

func join(path, name string) string {
	if path == "" {
		return name
	}
	return strings.TrimSuffix(path, "[]") + "." + name
}

// checkExtent refuses an element count whose byte size would not survive the
// multiplication.
//
// n is a Go int and the driver takes a size_t, so the product is computed in
// an int first and can wrap. A wrapped size is the dangerous outcome rather
// than the obvious one: a negative product converts to an enormous uint64 and
// the allocation simply fails, but a product that wraps to a small positive
// number allocates far too little and hands back a []T claiming every element
// the caller asked for. The same shape as simt.Kernel.LaunchSharedDim's
// 32-bit check, for the same reason.
func checkExtent[T any](op string, n int) error {
	if n < 0 {
		return &LengthError{Op: op, Want: 0, Got: n}
	}
	if size := int(sizeOfType[T]()); size > 0 && n > maxInt/size {
		return fmt.Errorf("cuda: %s: %d elements of %d bytes does not fit in an int", op, n, size)
	}
	return nil
}

const maxInt = int(^uint(0) >> 1)

func sizeOfType[T any]() uintptr { return reflect.TypeFor[T]().Size() }
