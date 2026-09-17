package simt_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/CWBudde/gocuda"
	"github.com/CWBudde/gocuda/simt"
)

// TestGolden pins the generated CUDA for every example kernel. Run with
// GOCUDA_UPDATE=1 to refresh the golden files after an intentional change.
func TestGolden(t *testing.T) {
	for _, name := range []string{"VecAdd", "Magnitude", "Scale", "FIR"} {
		t.Run(name, func(t *testing.T) {
			got, err := simt.Transpile(gocuda.Kernels(), name)
			if err != nil {
				t.Fatalf("Transpile: %v", err)
			}
			path := filepath.Join("testdata", name+".cu")
			if os.Getenv("GOCUDA_UPDATE") == "1" {
				if err := os.WriteFile(path, []byte(got), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			want, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("read golden (set GOCUDA_UPDATE=1 to create): %v", err)
			}
			if got != string(want) {
				t.Errorf("generated CUDA differs from %s:\n--- got ---\n%s", path, got)
			}
		})
	}
}
