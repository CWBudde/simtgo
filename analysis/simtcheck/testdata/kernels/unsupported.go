package kernels

import "github.com/CWBudde/simtgo/gpu"

func Float64Slice(ctx gpu.Ctx, a []float64) { // want `float64 needs //simtgo:float64`
	a[0] = 1
}

// Float64Allowed is the other half, and the more important one: the analyzer
// and the transpiler have to agree that the directive works, not just that its
// absence is refused. Nothing else makes them.
//
//simtgo:float64
func Float64Allowed(ctx gpu.Ctx, a []float64) {
	a[0] = a[1] * 2
}

// IntSlice lowered cleanly and returned wrong numbers until the type work of
// Phase 2: Go's int is 8 bytes and the emitted C int is 4.
func IntSlice(ctx gpu.Ctx, a []int) { // want `cannot cross to the device`
	a[0] = 1
}

// NarrowArithmetic is the refusal the narrow storage types exist around: the
// bytes cross fine, and an operator on one is where Go and C part company.
func NarrowArithmetic(ctx gpu.Ctx, a []uint8) {
	a[0] = a[1] * 2 // want `storage only on the device`
}

// NarrowLocal is the other half of the same rule, and the one the analyzer
// reaches through a different path: a declaration rather than an expression.
func NarrowLocal(ctx gpu.Ctx, a []uint8) {
	v := a[0] // want `not a variable, a parameter or a result`
	a[1] = v
}

func Goroutine(ctx gpu.Ctx, a []float32) {
	go func() {}() // want `unsupported statement`
	a[0] = 1
}

// A kernel may call a Go function, but not one that reaches itself: there is
// no stack depth on the device to spend on recursion.
func down(x float32) float32 {
	if x > 0 {
		return down(x - 1) // want `down calls itself`
	}
	return x
}

func CallsItself(ctx gpu.Ctx, a []float32) {
	a[0] = down(a[1])
}

func RuntimeSharedSize(ctx gpu.Ctx, a []float32) {
	s := ctx.SharedF32(len(a)) // want `constant size`
	s[0] = 1
}

var gain float32 = 2

func PackageLevelVar(ctx gpu.Ctx, a []float32) {
	a[0] = gain // want `declared outside the kernel`
}

// AtomicOnAnExpression is refused because the device takes the address of one
// element, and a slice expression has no address to take.
func AtomicOnAnExpression(ctx gpu.Ctx, h []int32) {
	gpu.AtomicAddI32(h[0:2], 0, 1) // want `needs the buffer itself as its first argument`
}

// Float64MathWithoutTheDirective is refused on the call rather than on a type,
// because there is no float64 in the signature to refuse.
func Float64MathWithoutTheDirective(ctx gpu.Ctx, a []float32) {
	a[0] = float32(gpu.Sqrt64(2)) // want `is double precision and needs //simtgo:float64`
}

// A kernel has one dynamic __shared__ block, so a second name for it would be
// another view of the same bytes -- which NVRTC accepts without a word.
func TwoDynamicTiles(ctx gpu.Ctx, y []float32) {
	a := ctx.SharedDynF32()
	b := ctx.SharedDynI32() // want `at most one dynamically sized shared tile`
	y[0] = a[0] + float32(b[0])
}

// The dynamic tile's length is a parameter of the kernel, which a device
// function cannot see. A statically sized tile there is fine.
//
//simtgo:ignore
func dynStage(ctx gpu.Ctx) float32 {
	s := ctx.SharedDynF32() // want `may only be declared in a kernel`
	return s[0]
}

func DynamicTileInADeviceFunction(ctx gpu.Ctx, y []float32) {
	y[0] = dynStage(ctx)
}

// A tile of doubles is double-precision vocabulary like any other, so it needs
// the directive even though the kernel writes no float64 down.
func Float64Tile(ctx gpu.Ctx, y []float32) {
	s := ctx.SharedF64(4) // want `float64 needs //simtgo:float64`
	y[0] = float32(s[0])
}

// NegativeLane is refused because CUDA reads a lane offset as unsigned: -1 is
// not the lane below but lane 4294967295, and the CPU emulator has an answer
// for it that the device does not.
func NegativeLane(ctx gpu.Ctx, y []float32) {
	y[0] = ctx.ShuffleUpF32(y[1], -1) // want `takes a lane offset and -1 is negative`
}

// Marked is decoration: it takes no gpu.Ctx, so nothing about it would have
// been a kernel to begin with. Nothing calls it either, which is the point --
// the check is package-wide, not part of the lowering.
//
//simtgo:device
func Marked(x float32) float32 { return x } // want `//simtgo:device does nothing on Marked`

// blend writes through one of its two buffers, so passing it the same one
// twice makes two __restrict__ pointers alias inside the generated C.
func blend(out, a []float32) { out[0] = a[0] * 2 }

func AliasedCall(ctx gpu.Ctx, y []float32) {
	blend(y, y) // want `passed the same buffer as both out and a`
}

// DivergentBarrier is the refused half of the barrier rules. The analyzer runs
// the same lowering, so what it has to agree with the transpiler about is that
// a barrier only some threads reach is refused at all -- the rule lives in
// internal/lower and neither of them may have its own copy.
func DivergentBarrier(ctx gpu.Ctx, y []float32) {
	s := ctx.SharedF32(4)
	if ctx.ThreadIdx() == 0 {
		ctx.SyncThreads() // want `is under a thread-varying if`
	}
	y[0] = s[0]
}

// DivergentLoopBarrier is the trip-count rule: nothing around the barrier is
// conditional, and the threads still do not execute it the same number of
// times. It is the staging loop every kernel here writes, with the barrier
// moved inside it.
func DivergentLoopBarrier(ctx gpu.Ctx, y []float32) {
	s := ctx.SharedF32(4)
	for k := ctx.ThreadIdx(); k < 4; k += ctx.BlockDim() {
		ctx.SyncThreads() // want `is inside a loop whose trip count differs between threads`
		s[k] = 1
	}
	y[0] = s[0]
}

// stagedTile is where the barrier the kernel below never spells lives.
//
//simtgo:ignore
func stagedTile(ctx gpu.Ctx) float32 {
	s := ctx.SharedF32(4)
	ctx.SyncThreads()
	return s[0]
}

// ReturnsBeforeBarrier is the return rule, and the interprocedural half of it
// at once: the threads that returned are gone, and the rendezvous they are
// missing is inside a function this kernel only calls.
func ReturnsBeforeBarrier(ctx gpu.Ctx, y []float32) {
	if ctx.GlobalID() >= len(y) {
		return
	}
	y[0] = stagedTile(ctx) // want `which reaches ctx.SyncThreads\(\), is preceded by a return at`
}
