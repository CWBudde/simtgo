package fuzz

import (
	"math"
	"math/rand/v2"
)

// Inputs materialises one set of argument values for the program.
//
// seed is separate from the program's, so the same program can be run against
// several inputs: what a kernel computes and what it is handed are independent
// choices, and only the first of them is what the transpiler is being tested on.
func (p *Program) Inputs(seed int64) *Args {
	r := rand.New(rand.NewPCG(uint64(seed), 0x9e3779b97f4a7c15))
	a := &Args{Vals: make([]any, len(p.Params))}
	for i, spec := range p.Params {
		if spec.AliasOf >= 0 {
			// Two read-only parameters deliberately bound to one buffer. What
			// __restrict__ forbids is reaching a modified object through
			// another pointer, so this is sound exactly because neither is
			// written; CloneArgs keeps the sharing.
			a.Vals[i] = a.Vals[spec.AliasOf]
			continue
		}
		if spec.Slice {
			a.Vals[i] = makeSlice(r, spec.Kind, spec.Len)
			continue
		}
		a.Vals[i] = makeScalar(r, spec.Kind)
	}
	return a
}

// specialF64 are the float values a fuzzer exists to reach.
//
// NaN, the two zeros, the two infinities and a subnormal are all values the
// two backends may treat differently and the parity tests never generate:
// NUMERICS.md records that no subnormal has ever been through a device from
// this repository, and that Go's builtin min and max disagree with CUDA's
// about a NaN operand with nothing testing it. Putting them in the input
// distribution is how a generated min(a, b) walks into that.
var specialF64 = []float64{
	math.NaN(),
	math.Inf(1), math.Inf(-1),
	0, math.Copysign(0, -1),
	math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32,
	float64(math.Float32frombits(0x00400000)), // a float32 subnormal, 2^-127
	math.MaxFloat32, -math.MaxFloat32,
	1, -1, 0.5, -0.5,
}

// specialChance is how often an element is drawn from specialF64 rather than
// from the ordinary distribution. It is high enough that a buffer of any
// useful length holds several, and low enough that a kernel still computes
// with ordinary numbers most of the time.
const specialChance = 6

func makeFloat(r *rand.Rand) float64 {
	if r.IntN(specialChance) == 0 {
		return specialF64[r.IntN(len(specialF64))]
	}
	return (r.Float64()*2 - 1) * float64(int64(1)<<r.IntN(8))
}

// makeInt draws an integer input.
//
// The magnitude is bounded deliberately. Signed overflow wraps in Go and is
// undefined in C, so a generated a*b on two large inputs would be a mismatch
// about the generator rather than about the emitter; a bound of a thousand
// leaves room for a couple of multiplications inside int32. The cost is that
// the extremes of the integer types are not exercised, which is stated here
// rather than hidden.
func makeInt(r *rand.Rand) int64 {
	switch r.IntN(8) {
	case 0:
		return 0
	case 1:
		return 1
	case 2:
		return -1
	}
	return int64(r.IntN(2001) - 1000)
}

func makeSlice(r *rand.Rand, k Kind, n int) any {
	switch k {
	case KF32:
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(makeFloat(r))
		}
		return s
	case KF64:
		s := make([]float64, n)
		for i := range s {
			s[i] = makeFloat(r)
		}
		return s
	case KI32:
		s := make([]int32, n)
		for i := range s {
			s[i] = int32(makeInt(r))
		}
		return s
	case KI64:
		s := make([]int64, n)
		for i := range s {
			s[i] = makeInt(r)
		}
		return s
	case KU32:
		s := make([]uint32, n)
		for i := range s {
			s[i] = uint32(makeInt(r) & 0xffff)
		}
		return s
	case KU64:
		s := make([]uint64, n)
		for i := range s {
			s[i] = uint64(makeInt(r) & 0xffff)
		}
		return s
	case KBool:
		s := make([]bool, n)
		for i := range s {
			s[i] = r.IntN(2) == 0
		}
		return s
	case KI8:
		s := make([]int8, n)
		for i := range s {
			s[i] = int8(r.IntN(256) - 128)
		}
		return s
	case KI16:
		s := make([]int16, n)
		for i := range s {
			s[i] = int16(makeInt(r))
		}
		return s
	case KU8:
		s := make([]uint8, n)
		for i := range s {
			s[i] = uint8(r.IntN(256))
		}
		return s
	case KU16:
		s := make([]uint16, n)
		for i := range s {
			s[i] = uint16(makeInt(r) & 0x7ff)
		}
		return s
	}
	panic("fuzz: no slice of " + k.goName())
}

func makeScalar(r *rand.Rand, k Kind) any {
	switch k {
	case KF32:
		return float32(makeFloat(r))
	case KF64:
		return makeFloat(r)
	case KI32:
		return int32(makeInt(r))
	case KI64:
		return makeInt(r)
	case KU32:
		return uint32(makeInt(r) & 0xffff)
	case KU64:
		return uint64(makeInt(r) & 0xffff)
	case KBool:
		return r.IntN(2) == 0
	case KInt:
		return int(makeInt(r))
	}
	panic("fuzz: no scalar of " + k.goName())
}
