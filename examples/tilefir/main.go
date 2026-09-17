// Command tilefir builds a DSP pipeline with the tile DSL -- scale, FIR,
// magnitude -- and shows it collapsing into a single kernel.
//
//	go run -tags cuda ./examples/tilefir
package main

import (
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"time"

	"github.com/CWBudde/gocuda/cuda"
	"github.com/CWBudde/gocuda/tile"
)

const n = 1 << 22

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	re, im := signal(n, 1), signal(n, 2)
	h := make([]float32, 33)
	for i := range h {
		h[i] = 1 / float32(len(h))
	}

	ctx, err := cuda.NewContext(0)
	if err != nil {
		return err
	}
	defer ctx.Close()
	fmt.Printf("device: %s (%s)\n", ctx.Name(), ctx.Arch())
	fmt.Printf("pipeline: Hypot(FIR(Scale(re, 0.5), h), im) over %d samples\n\n", n)

	build := func() *tile.Tensor {
		g := tile.New(ctx)
		return tile.Hypot(tile.FIR(tile.Scale(g.Input(re), 0.5), h), g.Input(im))
	}

	src, err := build().Source()
	if err != nil {
		return err
	}
	fmt.Printf("%s\n", src)

	// Warm up, so neither timing pays for the NVRTC compile.
	if _, err := build().Materialize(); err != nil {
		return err
	}
	if _, _, err := build().MaterializeStepwise(); err != nil {
		return err
	}

	t0 := time.Now()
	fused, err := build().Materialize()
	if err != nil {
		return err
	}
	fusedTime := time.Since(t0)

	t0 = time.Now()
	stepwise, launches, err := build().MaterializeStepwise()
	if err != nil {
		return err
	}
	stepTime := time.Since(t0)

	want := make([]float32, n)
	for i := range want {
		var acc float32
		for k := range h {
			if i-k >= 0 {
				acc += h[k] * 0.5 * re[i-k]
			}
		}
		want[i] = float32(math.Hypot(float64(acc), float64(im[i])))
	}
	if err := compare(fused, want); err != nil {
		return err
	}
	if err := compare(stepwise, want); err != nil {
		return err
	}

	fmt.Printf("fused:     1 kernel,  %s\n", fusedTime.Round(time.Microsecond))
	fmt.Printf("stepwise:  %d kernels, %s  (%.2fx slower, plus 2 temporary buffers)\n",
		launches, stepTime.Round(time.Microsecond), float64(stepTime)/float64(fusedTime))
	fmt.Printf("\nPASS: both paths match the Go reference over %d samples\n", n)
	return nil
}

func signal(n int, seed uint64) []float32 {
	r := rand.New(rand.NewPCG(seed, 99))
	xs := make([]float32, n)
	for i := range xs {
		xs[i] = float32(r.NormFloat64())
	}
	return xs
}

func compare(got, want []float32) error {
	for i := range got {
		d := math.Abs(float64(got[i] - want[i]))
		if d > 1e-5*max(math.Abs(float64(want[i])), 1) {
			return fmt.Errorf("FAIL: element %d: got %v, want %v", i, got[i], want[i])
		}
	}
	return nil
}
