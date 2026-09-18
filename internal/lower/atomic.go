package lower

import (
	"fmt"
	"go/ast"
	"strings"
)

// gpuAtomics maps package gpu's atomic vocabulary to the CUDA built-in each
// one is. Every overload named here exists at compute_75 with no header
// included, which is all NVRTC has: it compiles a bare string.
//
// The set is smaller than CUDA's because the Go side is typed. atomicMin,
// atomicMax and atomicCAS have no float overload, which is why the only
// float32 entry is the add; see package gpu for the rest of the gaps and why
// they are left open rather than emulated.
var gpuAtomics = map[string]string{
	"AtomicAddF32":  "atomicAdd",
	"AtomicAddI32":  "atomicAdd",
	"AtomicMinI32":  "atomicMin",
	"AtomicMaxI32":  "atomicMax",
	"AtomicExchI32": "atomicExch",
	"AtomicCASI32":  "atomicCAS",
}

// atomic lowers gpu.AtomicAddF32(s, i, v) to atomicAdd(&s[i], v).
//
// It cannot go through the ordinary call path, which renders each argument as
// a plain expression: here the first two arguments together become one C
// operand, and that operand contains the only "&" this emitter ever writes.
// The Go vocabulary takes a buffer and an index precisely because the subset
// has no address-of -- &s[i] is something the emitter produces and a kernel
// can never say.
//
// That is also why the buffer is checked here rather than rendered as an
// ordinary expression. The only operands whose address means anything on the
// device are a slice parameter, which is already a pointer, and a __shared__
// tile; t.lens holds exactly those two kinds of object and nothing else, so
// membership of it is the test. Taking the address of a __shared__ array
// yields a generic pointer, which the hardware resolves back to a shared
// atomic, so a tile works here and keeps working when it is passed on to a
// device function.
func (t *transpiler) atomic(c *ast.CallExpr, goName, cfn string) cexpr {
	if len(c.Args) < 2 {
		// go/types has already checked the call against the declared
		// signature; this only keeps a malformed AST from indexing off the end.
		t.fail(c.Pos(), "gpu.%s takes a buffer, an index and a value", goName)
		return atom("")
	}
	buf, ok := unparen(c.Args[0]).(*ast.Ident)
	if !ok {
		t.fail(c.Args[0].Pos(), "gpu.%s needs the buffer itself as its first argument, a slice parameter or a shared buffer: the device takes the address of one element, and an expression has no address to take", goName)
		return atom("")
	}
	// Rendering the name through the ordinary identifier path rather than
	// re-deriving it keeps its refusals -- a package-level variable, the blank
	// identifier, an object already complained about -- and keeps the escaping
	// of a name that collides with a C++ keyword in one place. A buffer called
	// "float" is declared as "float_", and spelling it a second time here is
	// how the emitted C would come to name something that was never declared.
	name := t.expr(buf)
	if t.failed() {
		return atom("")
	}
	obj := t.info.Uses[buf]
	if obj == nil {
		obj = t.info.Defs[buf]
	}
	if _, ok := t.lens[obj]; !ok {
		// Defensive rather than reachable through Transpile, where a local
		// slice is refused at its declaration and a device function cannot
		// return one. internal/lower is also driven by the analyzer over
		// packages this emitter did not construct, so the check earns its keep.
		t.fail(buf.Pos(), "gpu.%s needs a slice parameter or a shared buffer as its first argument; %s is neither", goName, buf.Name)
		return atom("")
	}
	// The subscript is delimited by its brackets, exactly as in any index
	// expression, so it needs no precedence of its own; "&" binds as a prefix
	// operator over a postfix subscript, and the whole thing sits in an
	// argument slot. No new precedence level is involved.
	args := make([]string, 0, len(c.Args)-1)
	args = append(args, fmt.Sprintf("&%s[%s]", name.at(precPostfix), t.expr(c.Args[1]).s))
	for _, a := range c.Args[2:] {
		args = append(args, t.expr(a).at(precArg))
	}
	return atom("%s(%s)", cfn, strings.Join(args, ", "))
}
