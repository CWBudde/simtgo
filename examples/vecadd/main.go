// Command vecadd runs one Go function as both a CPU grid and a CUDA kernel.
//
//	go run -tags cuda ./examples/vecadd
package main

import (
	"fmt"
	"log"

	"github.com/CWBudde/gocuda"
	"github.com/CWBudde/gocuda/cuda"
	"github.com/CWBudde/gocuda/gpu"
	"github.com/CWBudde/gocuda/kernels"
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
	a := make([]float32, n)
	b := make([]float32, n)
	for i := range a {
		a[i] = float32(i)
		b[i] = float32(2 * i)
	}

	// The CPU path calls the kernel directly -- it is just a Go function.
	cpu := make([]float32, n)
	gpu.RunCPU((n+block-1)/block, block, func(c gpu.Ctx) { kernels.VecAdd(c, cpu, a, b) })

	ctx, err := cuda.NewContext(0)
	if err != nil {
		return err
	}
	defer ctx.Close()
	fmt.Printf("device: %s (%s)\n\n", ctx.Name(), ctx.Arch())

	// The GPU path transpiles that same function and compiles it with NVRTC.
	k, err := simt.Build(ctx, gocuda.Kernels(), "VecAdd")
	if err != nil {
		return err
	}
	fmt.Printf("generated from kernels/kernels.go:\n%s\n", k.Source)

	dc, err := cuda.NewSlice[float32](ctx, n)
	if err != nil {
		return err
	}
	defer dc.Free()
	da, err := cuda.Upload(ctx, a)
	if err != nil {
		return err
	}
	defer da.Free()
	db, err := cuda.Upload(ctx, b)
	if err != nil {
		return err
	}
	defer db.Free()

	if err := k.LaunchN(n, block, dc, da, db); err != nil {
		return err
	}
	got, err := dc.Download()
	if err != nil {
		return err
	}

	for i := range got {
		if got[i] != cpu[i] {
			return fmt.Errorf("FAIL: element %d: gpu %v, cpu %v", i, got[i], cpu[i])
		}
	}
	fmt.Printf("PASS: %d elements identical on CPU and GPU\n", n)
	return nil
}
