//go:build cuda

package simt

import (
	"errors"
	"testing"

	"github.com/CWBudde/simtgo/cuda"
)

// TestLaunchOnQueuesOnAStream is the simt side of the asynchronous launch:
// the same checks, the same arguments, and the result only readable once the
// stream says so.
func TestLaunchOnQueuesOnAStream(t *testing.T) {
	dev := freshDevice(t)
	k, err := Build(dev, kernelSources, "VecAdd")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	st, err := dev.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer st.Close()

	const n = 4096
	a := make([]float32, n)
	b := make([]float32, n)
	for i := range a {
		a[i], b[i] = float32(i), float32(2*i)
	}
	da, err := cuda.Upload(dev, a)
	if err != nil {
		t.Fatalf("Upload: %v", err)
	}
	defer da.Free()
	db, err := cuda.Upload(dev, b)
	if err != nil {
		t.Fatalf("Upload: %v", err)
	}
	defer db.Free()
	dc, err := cuda.NewSlice[float32](dev, n)
	if err != nil {
		t.Fatalf("NewSlice: %v", err)
	}
	defer dc.Free()

	if err := k.LaunchOn(st, (n+255)/256, 256, dc, da, db); err != nil {
		t.Fatalf("LaunchOn: %v", err)
	}
	if err := st.Wait(t.Context()); err != nil {
		t.Fatalf("Wait: %v", err)
	}
	got, err := dc.Download()
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	for i := range got {
		if want := a[i] + b[i]; got[i] != want {
			t.Fatalf("element %d = %v, want %v", i, got[i], want)
		}
	}
}

// TestLaunchOnKeepsTheHostSideChecks pins that going asynchronous did not
// quietly drop the contract. All three of these are decided before the
// kernel would start, so none of them has any excuse to wait for a stream.
func TestLaunchOnKeepsTheHostSideChecks(t *testing.T) {
	dev := freshDevice(t)
	k, err := Build(dev, kernelSources, "VecAdd")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	st, err := dev.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer st.Close()

	const n = 256
	d, err := cuda.NewSlice[float32](dev, n)
	if err != nil {
		t.Fatalf("NewSlice: %v", err)
	}
	defer d.Free()
	other, err := cuda.NewSlice[float32](dev, n)
	if err != nil {
		t.Fatalf("NewSlice: %v", err)
	}
	defer other.Free()

	t.Run("argument count", func(t *testing.T) {
		var e *ArgCountError
		if err := k.LaunchOn(st, 1, n, d); !errors.As(err, &e) {
			t.Errorf("got %v, want an *ArgCountError", err)
		}
	})
	t.Run("aliasing", func(t *testing.T) {
		var e *AliasError
		if err := k.LaunchOn(st, 1, n, d, d, other); !errors.As(err, &e) {
			t.Errorf("got %v, want an *AliasError", err)
		}
	})
	t.Run("no stream", func(t *testing.T) {
		if err := k.LaunchOn(nil, 1, n, d, other, other); err == nil {
			t.Error("a nil stream was accepted")
		}
	})
}
