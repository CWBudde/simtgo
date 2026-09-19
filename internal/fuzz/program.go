package fuzz

import (
	"fmt"

	"github.com/CWBudde/simtgo/gpu"
)

// A Program is one generated kernel package, together with the launch it is
// meant to be run with and a description of the buffers it takes.
//
// The geometry is generated independently of the source and is deliberately
// ragged: a launch whose thread count divides the buffer length exactly never
// takes the guard a kernel writes around its tail, and a divergence bug has
// already hidden behind one here. The single exception is a kernel that
// declares AssumeBlockDim, which by construction declares the block size that
// was already chosen rather than one of its own.
type Program struct {
	// Seed is what Generate was given. The same seed produces an identical
	// Program, which is what makes a fuzz failure reproducible from one number.
	Seed int64

	Pkg   string
	Funcs []*Func // in the order the source declares them
	Main  *Func   // the kernel; also a member of Funcs

	// Float64 says the program carries //simtgo:float64, which it does exactly
	// when something in it is double precision.
	Float64 bool

	Grid, Block gpu.Dim

	// DynLen is the element count the launch gives the dynamic shared tile,
	// and HasDyn whether the kernel declares one. They are separate because
	// zero is a length a launch may legitimately give.
	DynLen int
	HasDyn bool

	Params []ParamSpec

	// Shapes are the struct types this program declares, in declaration order.
	// It is empty for most programs: a struct is one of the things the
	// generator may reach for, not something every kernel has.
	Shapes []*StructShape
}

// A ParamSpec describes one kernel parameter to whoever has to supply it.
//
// AliasOf is the one piece that is not a property of the source. Two read-only
// parameters may deliberately share a buffer -- what __restrict__ forbids is
// reaching a modified object through another pointer, so dot(x, x) is sound --
// and that accept/refuse boundary is subtle enough to be worth generating. A
// buffer is never shared where either parameter is written: the emulator
// cannot see aliasing at all, so such a program would manufacture a mismatch
// that says nothing about the emitter.
type ParamSpec struct {
	Name     string
	Kind     Kind
	Slice    bool
	Len      int
	ReadOnly bool
	AliasOf  int // -1, or the index of the earlier parameter sharing this buffer
	// Shape is the struct type when Kind is KStruct, and nil otherwise. It is
	// what Inputs needs to make a buffer of, since nothing outside structs.go
	// names the type.
	Shape *StructShape
}

// Name is the kernel's name, which is also the generated C entry point.
func (p *Program) Name() string { return p.Main.Name }

// Args is one set of argument values, in the order Params lists them.
//
// Each element is the Go value the parameter takes: a []float32, an int32 and
// so on. Two entries are the same slice value exactly where ParamSpec.AliasOf
// says so.
type Args struct{ Vals []any }

// CloneArgs copies every buffer, so two backends can be handed the same inputs
// and write their outputs independently.
//
// The aliasing is preserved from ParamSpec rather than rediscovered from the
// values: a copy that quietly unshared two parameters bound to one buffer
// would be running a different program from the one the other backend ran.
func (p *Program) CloneArgs(a *Args) *Args {
	out := &Args{Vals: make([]any, len(a.Vals))}
	for i, spec := range p.Params {
		if spec.AliasOf >= 0 {
			out.Vals[i] = out.Vals[spec.AliasOf]
			continue
		}
		if spec.Shape != nil {
			// The shape knows its own type; nothing else here does.
			out.Vals[i] = spec.Shape.clone(a.Vals[i])
			continue
		}
		out.Vals[i] = cloneValue(a.Vals[i])
	}
	return out
}

// cloneValue copies one argument. A scalar arrives here as a copy already, so
// only the buffers have anything to do.
func cloneValue(v any) any {
	switch v := v.(type) {
	case []float32:
		return append([]float32(nil), v...)
	case []float64:
		return append([]float64(nil), v...)
	case []int32:
		return append([]int32(nil), v...)
	case []int64:
		return append([]int64(nil), v...)
	case []uint32:
		return append([]uint32(nil), v...)
	case []uint64:
		return append([]uint64(nil), v...)
	case []bool:
		return append([]bool(nil), v...)
	case []int8:
		return append([]int8(nil), v...)
	case []int16:
		return append([]int16(nil), v...)
	case []uint8:
		return append([]uint8(nil), v...)
	case []uint16:
		return append([]uint16(nil), v...)
	}
	return v
}

// Threads is how many threads one block has, across all three axes -- which is
// what AssumeBlockDim counts, so a 16x16 block satisfies AssumeBlockDim(256).
func (p *Program) Threads() int { return p.Block.X * p.Block.Y * p.Block.Z }

// Run executes the program on the CPU emulator with the given arguments,
// choosing the launch form the kernel's shared memory requires. Using the
// wrong one is refused on the device and would be an unsized tile here, so
// the choice is made from the program rather than left to the caller.
func (p *Program) Run(a *Args) {
	fn := p.Closure(a)
	if p.HasDyn {
		gpu.RunCPUSharedDim(p.Grid, p.Block, p.DynLen, fn)
		return
	}
	gpu.RunCPUDim(p.Grid, p.Block, fn)
}

// String is a one-line summary for a test failure to quote.
func (p *Program) String() string {
	return fmt.Sprintf("fuzz program %s (seed %d) grid=%v block=%v", p.Name(), p.Seed, p.Grid, p.Block)
}
