// Command fir filters a signal with a direct-form FIR kernel that stages its
// samples in shared memory, and times it against plain Go.
//
//	go run -tags cuda ./examples/fir
package main

import (
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"runtime"
	"sync"
	"time"

	"github.com/CWBudde/gocuda"
	"github.com/CWBudde/gocuda/cuda"
	"github.com/CWBudde/gocuda/kernels"
	"github.com/CWBudde/gocuda/simt"
)

const (
	n     = 1 << 22 // 4M samples
	taps  = 33
	block = kernels.FIRBlock
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	x := make([]float32, n)
	r := rand.New(rand.NewPCG(7, 11))
	for i := range x {
		x[i] = float32(r.NormFloat64())
	}
	h := lowpass(taps)

	// Baselines: one core, then every core. A GPU only earns its keep against
	// a CPU that is actually trying.
	y1 := make([]float32, n)
	t0 := time.Now()
	firSerial(y1, x, h)
	serial := time.Since(t0)

	yP := make([]float32, n)
	t0 = time.Now()
	firParallel(yP, x, h)
	parallel := time.Since(t0)

	ctx, err := cuda.NewContext(0)
	if err != nil {
		return err
	}
	defer ctx.Close()
	fmt.Printf("device: %s (%s)\n", ctx.Name(), ctx.Arch())
	fmt.Printf("signal: %d samples, %d taps, block %d\n\n", n, taps, block)

	t0 = time.Now()
	k, err := simt.Build(ctx, gocuda.Kernels(), "FIR")
	if err != nil {
		return err
	}
	build := time.Since(t0)

	t0 = time.Now()
	dy, err := cuda.NewSlice[float32](ctx, n)
	if err != nil {
		return err
	}
	defer dy.Free()
	dx, err := cuda.Upload(ctx, x)
	if err != nil {
		return err
	}
	defer dx.Free()
	dh, err := cuda.Upload(ctx, h)
	if err != nil {
		return err
	}
	defer dh.Free()
	upload := time.Since(t0)

	// Warm up: the first launch pays for module load and context set-up.
	if err := k.LaunchN(n, block, dy, dx, dh); err != nil {
		return err
	}
	t0 = time.Now()
	if err := k.LaunchN(n, block, dy, dx, dh); err != nil {
		return err
	}
	kernel := time.Since(t0)

	t0 = time.Now()
	y, err := dy.Download()
	if err != nil {
		return err
	}
	download := time.Since(t0)

	if err := compare(y, y1); err != nil {
		return err
	}
	if err := compare(yP, y1); err != nil {
		return err
	}

	// Which path ran is worth saying, because it is most of this number: with
	// prebuilt PTX the kernel is only transpiled and loaded, and NVRTC -- some
	// 23 ms of it -- is skipped entirely.
	how := "transpile + NVRTC compile"
	if k.Prebuilt {
		how = fmt.Sprintf("transpile + load %s PTX", k.Arch)
	}
	fmt.Printf("%-27s %8s\n", how, build.Round(time.Microsecond))
	fmt.Printf("upload  (%d MiB)             %8s\n", (n*4)>>20, upload.Round(time.Microsecond))
	fmt.Printf("kernel                      %8s\n", kernel.Round(time.Microsecond))
	fmt.Printf("download                    %8s\n\n", download.Round(time.Microsecond))
	fmt.Printf("CPU, one core               %8s\n", serial.Round(time.Microsecond))
	fmt.Printf("CPU, %2d cores               %8s  (%.1fx)\n", runtime.NumCPU(), parallel.Round(time.Microsecond), float64(serial)/float64(parallel))
	fmt.Printf("GPU kernel only             %8s  (%.1fx)\n", kernel.Round(time.Microsecond), float64(serial)/float64(kernel))
	fmt.Printf("GPU incl. transfers         %8s  (%.1fx)\n", (upload + kernel + download).Round(time.Microsecond), float64(serial)/float64(upload+kernel+download))
	fmt.Printf("\nPASS: %d samples match the Go reference\n", n)
	return nil
}

// lowpass is a windowed-sinc low-pass at a quarter of the sample rate.
func lowpass(taps int) []float32 {
	h := make([]float32, taps)
	mid := float64(taps-1) / 2
	var sum float64
	for i := range h {
		d := float64(i) - mid
		v := 0.5 // sinc(0) for fc = 0.25
		if d != 0 {
			v = math.Sin(math.Pi*0.5*d) / (math.Pi * d)
		}
		v *= 0.54 - 0.46*math.Cos(2*math.Pi*float64(i)/float64(taps-1)) // Hamming
		h[i] = float32(v)
		sum += v
	}
	for i := range h {
		h[i] /= float32(sum)
	}
	return h
}

func firSerial(y, x, h []float32) {
	for i := range y {
		var acc float32
		for k := range h {
			if i-k >= 0 {
				acc += h[k] * x[i-k]
			}
		}
		y[i] = acc
	}
}

func firParallel(y, x, h []float32) {
	workers := runtime.NumCPU()
	chunk := (len(y) + workers - 1) / workers
	var wg sync.WaitGroup
	for w := range workers {
		lo := w * chunk
		hi := min(lo+chunk, len(y))
		if lo >= hi {
			continue
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := lo; i < hi; i++ {
				var acc float32
				for k := range h {
					if i-k >= 0 {
						acc += h[k] * x[i-k]
					}
				}
				y[i] = acc
			}
		}()
	}
	wg.Wait()
}

func compare(got, want []float32) error {
	for i := range got {
		d := math.Abs(float64(got[i] - want[i]))
		if d > 1e-5*max(math.Abs(float64(want[i])), 1) {
			return fmt.Errorf("FAIL: sample %d: got %v, want %v", i, got[i], want[i])
		}
	}
	return nil
}
