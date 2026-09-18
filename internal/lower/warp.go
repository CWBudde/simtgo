package lower

import (
	"fmt"
	"go/ast"
	"strings"
)

// warpFullMask is the participation mask the emitter writes into every _sync
// built-in.
//
// The Go vocabulary has no mask argument. CUDA's does, and getting it wrong is
// undefined behaviour rather than a wrong number, so the emitter writes the
// one value whose meaning the Go side can state as a contract: every thread of
// the warp reaches the call. That is the same trade the atomics made in taking
// a buffer and an index rather than a pointer -- one Go name, one built-in,
// and something both backends can check. gpu.Ctx.Ballot and gpu.Ctx.ActiveMask
// are how a kernel reads a mask; none of them is how it writes one.
const warpFullMask = "0xffffffff"

// laneIDExpr is gpu.Ctx.LaneID.
//
// It is the flat thread index modulo the warp size rather than threadIdx.x &
// 31, because CUDA fills a warp with consecutive flat indices, x fastest: in a
// 16x16 block, thread (0,2) is lane 0 of warp 1. The literal 32 is CUDA's
// warpSize, which cannot be spelled as warpSize here -- that is an ordinary
// variable, so the division would be by a runtime value -- and gpu.WarpSize is
// the same number on the Go side.
const laneIDExpr = "(int)((threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) % 32)"

// warpOp is how one warp-level method of gpu.Ctx is spelled in CUDA.
type warpOp struct {
	// c is the built-in to call, empty for an operation that is not a call.
	c string
	// expr and prec spell the ones that are not: LaneID is arithmetic over the
	// thread indices, which is a C expression an operator around it can
	// regroup.
	expr string
	prec int
	// mask is whether the built-in takes a participation mask, which the
	// emitter supplies. __activemask is the one that does not.
	mask bool
	// lane is whether the second argument names a lane, a delta or a lane
	// mask. Those are checked; a value is not.
	lane bool
	// cbool is whether the built-in returns an int where the Go method returns
	// a bool. C++ would convert it silently, and the generated source is read
	// by people, so the comparison is written out.
	cbool bool
}

// ctxWarp maps gpu.Ctx's warp-level vocabulary to the CUDA built-ins.
//
// Every one of these is declared by NVRTC with no header included, which is
// all it has: simt/nvrtc_cuda_test.go compiles one kernel per entry, because a
// comment claiming a built-in exists is not a measurement.
//
// It is a table of its own rather than an extension of ctxBuiltins because
// these take arguments and that one maps a method to a fixed expression. Two
// tables also mean the existing kernels lower to byte-identical CUDA, which is
// what the goldens assert.
var ctxWarp = map[string]warpOp{
	"LaneID": {expr: laneIDExpr, prec: precPrefix},

	"ShuffleF32":     {c: "__shfl_sync", mask: true, lane: true},
	"ShuffleI32":     {c: "__shfl_sync", mask: true, lane: true},
	"ShuffleXorF32":  {c: "__shfl_xor_sync", mask: true, lane: true},
	"ShuffleXorI32":  {c: "__shfl_xor_sync", mask: true, lane: true},
	"ShuffleUpF32":   {c: "__shfl_up_sync", mask: true, lane: true},
	"ShuffleUpI32":   {c: "__shfl_up_sync", mask: true, lane: true},
	"ShuffleDownF32": {c: "__shfl_down_sync", mask: true, lane: true},
	"ShuffleDownI32": {c: "__shfl_down_sync", mask: true, lane: true},

	"Ballot":     {c: "__ballot_sync", mask: true},
	"Any":        {c: "__any_sync", mask: true, cbool: true},
	"All":        {c: "__all_sync", mask: true, cbool: true},
	"ActiveMask": {c: "__activemask"},
	"SyncWarp":   {c: "__syncwarp", mask: true},
}

// warp lowers a warp-level Ctx method to its built-in.
//
// Unlike the atomics this needs no special handling of the arguments the
// kernel wrote -- they are ordinary expressions in ordinary argument positions
// -- but it does have to put the mask in front of them, which no ordinary call
// does, and it is the only place a Go bool comes back from a C int.
func (t *transpiler) warp(c *ast.CallExpr, name string, op warpOp) cexpr {
	if op.c == "" {
		if len(c.Args) != 0 {
			// go/types has already checked the call against the declared
			// signature; this only keeps a malformed AST from being lowered
			// into a built-in that silently drops what it was handed.
			t.fail(c.Pos(), "gpu.Ctx.%s takes no arguments", name)
			return atom("")
		}
		return cexpr{op.expr, op.prec}
	}
	args := make([]string, 0, len(c.Args)+1)
	if op.mask {
		args = append(args, warpFullMask)
	}
	for i, a := range c.Args {
		if op.lane && i == 1 {
			t.checkLane(name, a)
		}
		args = append(args, t.expr(a).at(precArg))
	}
	call := fmt.Sprintf("%s(%s)", op.c, strings.Join(args, ", "))
	if op.cbool {
		return cexpr{call + " != 0", precEq}
	}
	return atom("%s", call)
}

// checkLane refuses a lane offset that is a negative constant.
//
// CUDA reads every one of these as an unsigned lane number, so -1 is not the
// lane below but lane 4294967295: __shfl_sync would wrap it modulo the warp
// size and __shfl_down_sync would clamp it, and in neither case is it the
// neighbour the minus sign says. The CPU emulator, meanwhile, has a definite
// answer for it -- the caller's own value -- so this is exactly a spelling
// that means one thing where it is read and another where it runs.
//
// Only a constant can be checked. A negative value arrived at at run time is
// the kernel author's to avoid, and saying so in the documentation is all
// either backend can do about it.
func (t *transpiler) checkLane(name string, a ast.Expr) {
	if n, ok := t.constInt(a); ok && n < 0 {
		t.fail(a.Pos(), "gpu.Ctx.%s takes a lane offset and %d is negative; CUDA reads it as an unsigned lane number, so the device would not do what the sign says", name, n)
	}
}
