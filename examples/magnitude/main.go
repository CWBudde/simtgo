// Command magnitude turns a complex spectrum into a magnitude spectrum,
// exercising the float32 math mapping (gpu.Hypot becomes hypotf).
//
//	go run -tags cuda ./examples/magnitude
package main

import (
	"fmt"
	"log"
	"math"
	"math/rand/v2"

	"github.com/CWBudde/gocuda"
	"github.com/CWBudde/gocuda/cuda"
	"github.com/CWBudde/gocuda/simt"
)

const (
	n     = 1 << 20
	block = 256
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	re := make([]float32, n)
	im := make([]float32, n)
	r := rand.New(rand.NewPCG(3, 5))
	for i := range re {
		re[i] = float32(r.NormFloat64())
		im[i] = float32(r.NormFloat64())
	}

	ctx, err := cuda.NewContext(0)
	if err != nil {
		return err
	}
	defer ctx.Close()

	k, err := simt.Build(ctx, gocuda.Kernels(), "Magnitude")
	if err != nil {
		return err
	}
	fmt.Printf("device: %s (%s)\n\n%s\n", ctx.Name(), ctx.Arch(), k.Source)

	dm, err := cuda.NewSlice[float32](ctx, n)
	if err != nil {
		return err
	}
	defer dm.Free()
	dre, err := cuda.Upload(ctx, re)
	if err != nil {
		return err
	}
	defer dre.Free()
	dim, err := cuda.Upload(ctx, im)
	if err != nil {
		return err
	}
	defer dim.Free()

	if err := k.LaunchN(n, block, dm, dre, dim); err != nil {
		return err
	}
	got, err := dm.Download()
	if err != nil {
		return err
	}

	// The reference is Go's own math.Hypot in float64, so this also shows how
	// far the device's single-precision hypotf drifts from it.
	var worst float64
	for i := range got {
		want := math.Hypot(float64(re[i]), float64(im[i]))
		worst = max(worst, math.Abs(float64(got[i])-want)/max(want, 1))
	}
	if worst > 1e-6 {
		return fmt.Errorf("FAIL: worst relative error %.3g", worst)
	}
	fmt.Printf("PASS: %d magnitudes, worst relative error %.3g vs float64 math.Hypot\n", n, worst)
	return nil
}
