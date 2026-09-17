package tile

import (
	"github.com/CWBudde/gocuda/cuda"
	"github.com/CWBudde/gocuda/internal/jit"
)

// Source returns the CUDA kernel this tensor would be computed by, without
// compiling or running anything.
func (t *Tensor) Source() (string, error) {
	if t.g.err != nil {
		return "", t.g.err
	}
	src, _ := generate(t.node)
	return src, nil
}

// Materialize compiles the graph into one kernel, runs it, and brings the
// result back. This is where anything happens at all.
func (t *Tensor) Materialize() ([]float32, error) {
	g := t.g
	if g.err != nil {
		return nil, g.err
	}
	out, cleanup, err := t.run()
	if cleanup != nil {
		defer cleanup()
	}
	if err != nil {
		return nil, err
	}
	return out.Download()
}

// run executes the fused kernel and returns the output buffer plus a cleanup
// function releasing everything the call allocated.
func (t *Tensor) run() (*cuda.Slice[float32], func(), error) {
	g := t.g
	src, inputs := generate(t.node)
	res, err := jit.Load(g.ctx, src, "fused")
	if err != nil {
		return nil, nil, err
	}

	var owned []*cuda.Slice[float32]
	cleanup := func() {
		for _, s := range owned {
			s.Free()
		}
	}

	out, err := cuda.NewSlice[float32](g.ctx, t.node.n)
	if err != nil {
		return nil, cleanup, err
	}
	owned = append(owned, out)

	args := []any{out}
	for _, in := range inputs {
		s := in.dev
		if s == nil {
			if s, err = cuda.Upload(g.ctx, in.host); err != nil {
				return nil, cleanup, err
			}
			owned = append(owned, s)
		}
		args = append(args, s)
	}

	flat, err := cuda.BuildArgs(args...)
	if err != nil {
		return nil, cleanup, err
	}
	grid := (t.node.n + BlockSize - 1) / BlockSize
	if err := res.Func.Launch(cuda.D1(grid), cuda.D1(BlockSize), 0, flat...); err != nil {
		return nil, cleanup, err
	}
	return out, cleanup, nil
}

// MaterializeStepwise computes the graph the way an op-at-a-time library
// would: one kernel per operation, with a temporary buffer between each. It
// exists to make the cost of not fusing visible, and returns the number of
// kernels it had to launch.
func (t *Tensor) MaterializeStepwise() ([]float32, int, error) {
	g := t.g
	if g.err != nil {
		return nil, 0, g.err
	}
	cache := map[*node]*cuda.Slice[float32]{}
	var owned []*cuda.Slice[float32]
	defer func() {
		for _, s := range owned {
			s.Free()
		}
	}()

	launches := 0
	res, err := g.step(t.node, cache, &owned, &launches)
	if err != nil {
		return nil, launches, err
	}
	vals, err := res.Download()
	return vals, launches, err
}

// step materialises one node, having first materialised its arguments into
// device buffers of their own.
func (g *Graph) step(n *node, cache map[*node]*cuda.Slice[float32], owned *[]*cuda.Slice[float32], launches *int) (*cuda.Slice[float32], error) {
	if s, ok := cache[n]; ok {
		return s, nil
	}
	if n.kind == opInput {
		s, err := cuda.Upload(g.ctx, n.host)
		if err != nil {
			return nil, err
		}
		*owned = append(*owned, s)
		cache[n] = s
		return s, nil
	}

	// Rebuild this one operation over already-computed inputs, so the same
	// code generator serves both paths.
	args := make([]*node, len(n.args))
	for i, a := range n.args {
		s, err := g.step(a, cache, owned, launches)
		if err != nil {
			return nil, err
		}
		args[i] = &node{kind: opInput, n: a.n, dev: s}
	}
	single := &Tensor{g: g, node: &node{
		kind: n.kind, id: n.id, n: n.n, args: args, scale: n.scale, cfn: n.cfn,
	}}

	out, cleanup, err := single.run()
	*launches++
	if err != nil {
		if cleanup != nil {
			cleanup()
		}
		return nil, err
	}
	// run's own cleanup is dropped on success: its only allocation is out,
	// which has to outlive this call, so it joins the caller's list instead.
	*owned = append(*owned, out)
	cache[n] = out
	return out, nil
}
