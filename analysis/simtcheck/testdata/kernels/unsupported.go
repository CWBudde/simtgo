package kernels

import "github.com/CWBudde/gocuda/gpu"

func Float64Slice(ctx gpu.Ctx, a []float64) { // want `float64 needs //gocuda:float64`
	a[0] = 1
}

// Float64Allowed is the other half, and the more important one: the analyzer
// and the transpiler have to agree that the directive works, not just that its
// absence is refused. Nothing else makes them.
//
//gocuda:float64
func Float64Allowed(ctx gpu.Ctx, a []float64) {
	a[0] = a[1] * 2
}

// IntSlice lowered cleanly and returned wrong numbers until the type work of
// Phase 2: Go's int is 8 bytes and the emitted C int is 4.
func IntSlice(ctx gpu.Ctx, a []int) { // want `cannot cross to the device`
	a[0] = 1
}

func Int8Slice(ctx gpu.Ctx, a []int8) { // want `C promotes it to int`
	a[0] = 1
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
	a[0] = float32(gpu.Sqrt64(2)) // want `is double precision and needs //gocuda:float64`
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
//gocuda:ignore
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
	s := ctx.SharedF64(4) // want `float64 needs //gocuda:float64`
	y[0] = float32(s[0])
}
