// Package tile implements the tile track: a pipeline is described with
// ordinary Go calls, nothing runs until it is materialised, and the whole
// graph is then lowered into a single generated kernel.
//
// CUDA Rust's tile track embeds a kernel AST in the host binary with a proc
// macro. Go has no macros, so the graph is instead recorded at run time -- the
// Go program builds it by executing Go code, and the code generator sees the
// finished graph. The payoff is the same: the user never writes a thread
// index, and a chain of operations becomes one kernel with one pass over
// memory instead of one kernel and one temporary buffer per operation.
package tile

import (
	"errors"
	"fmt"

	"github.com/CWBudde/gocuda/cuda"
)

const (
	// BlockSize is the tile width: one block cooperates on this many samples.
	BlockSize = 256
	// MaxTaps bounds a windowed operation's halo, because the shared tile it
	// stages into needs a compile-time extent.
	MaxTaps = 64
)

type kind int

const (
	opInput kind = iota
	opScale
	opAdd
	opMul
	opUnary
	opHypot
	opFIR
)

type node struct {
	kind  kind
	id    int
	n     int // element count
	args  []*node
	host  []float32            // opInput: host data, uploaded on materialisation
	dev   *cuda.Slice[float32] // opInput: data already on the device
	scale float32              // opScale
	cfn   string               // opUnary: the CUDA function to call
}

// Graph records a pipeline. Operations append to it; nothing touches the
// device until Materialize.
type Graph struct {
	ctx   *cuda.Context
	nodes []*node
	err   error
}

// New starts a graph on the given device context.
func New(ctx *cuda.Context) *Graph { return &Graph{ctx: ctx} }

// Tensor is a one-dimensional value in a graph. It holds no data: it is a
// handle to a node whose value will be computed if and when it is needed.
type Tensor struct {
	g    *Graph
	node *node
}

// Len reports the tensor's element count, known while the graph is built.
func (t *Tensor) Len() int { return t.node.n }

func (g *Graph) add(n *node) *Tensor {
	n.id = len(g.nodes)
	g.nodes = append(g.nodes, n)
	return &Tensor{g: g, node: n}
}

func (g *Graph) fail(format string, args ...any) {
	if g.err == nil {
		g.err = fmt.Errorf("tile: "+format, args...)
	}
}

// Input introduces host data into the graph. The upload happens on
// materialisation, not here.
func (g *Graph) Input(xs []float32) *Tensor {
	return g.add(&node{kind: opInput, n: len(xs), host: xs})
}

// binary checks that two operands come from the same graph and agree in
// shape. Go cannot express that in its type system the way Rust's const
// generics can, so it is a runtime check made at graph-build time.
func binary(a, b *Tensor, k kind, name string) *Tensor {
	if a.g != b.g {
		a.g.fail("%s: operands come from different graphs", name)
		return a
	}
	if a.Len() != b.Len() {
		a.g.fail("%s: length %d does not match %d", name, a.Len(), b.Len())
		return a
	}
	return a.g.add(&node{kind: k, n: a.Len(), args: []*node{a.node, b.node}})
}

// Add returns a + b elementwise.
func Add(a, b *Tensor) *Tensor { return binary(a, b, opAdd, "Add") }

// Mul returns a * b elementwise.
func Mul(a, b *Tensor) *Tensor { return binary(a, b, opMul, "Mul") }

// Hypot returns sqrt(a*a + b*b) elementwise, the magnitude of a complex
// signal held as separate real and imaginary parts.
func Hypot(a, b *Tensor) *Tensor { return binary(a, b, opHypot, "Hypot") }

// Scale multiplies every element by k.
func Scale(a *Tensor, k float32) *Tensor {
	return a.g.add(&node{kind: opScale, n: a.Len(), args: []*node{a.node}, scale: k})
}

// Sqrt returns the elementwise square root.
func Sqrt(a *Tensor) *Tensor {
	return a.g.add(&node{kind: opUnary, n: a.Len(), args: []*node{a.node}, cfn: "sqrtf"})
}

// Abs returns the elementwise absolute value.
func Abs(a *Tensor) *Tensor {
	return a.g.add(&node{kind: opUnary, n: a.Len(), args: []*node{a.node}, cfn: "fabsf"})
}

// FIR filters a with the coefficients h, y[n] = sum(h[k] * a[n-k]).
//
// This is the windowed operation: each block stages its own tile plus a halo
// of len(h)-1 preceding values into shared memory, so a FIR fused after other
// operations still reads each input once.
func FIR(a *Tensor, h []float32) *Tensor {
	g := a.g
	if len(h) == 0 || len(h) > MaxTaps {
		g.fail("FIR: %d taps is outside 1..%d", len(h), MaxTaps)
		return a
	}
	if containsFIR(a.node) {
		// The halo would have to be evaluated at indices the inner window has
		// no value for; rather than mistranslate it, say so.
		g.fail("FIR: chaining windowed operations is not supported; materialise the first one")
		return a
	}
	taps := g.add(&node{kind: opInput, n: len(h), host: h})
	return g.add(&node{kind: opFIR, n: a.Len(), args: []*node{a.node, taps.node}})
}

func containsFIR(n *node) bool {
	if n.kind == opFIR {
		return true
	}
	for _, a := range n.args {
		if containsFIR(a) {
			return true
		}
	}
	return false
}

// Err reports the first error recorded while building the graph.
func (g *Graph) Err() error { return g.err }

// ErrEmpty is returned when a graph has nothing to compute.
var ErrEmpty = errors.New("tile: empty graph")
