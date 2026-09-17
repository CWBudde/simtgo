package simtcheck_test

import (
	"testing"

	"golang.org/x/tools/go/analysis/analysistest"

	"github.com/CWBudde/gocuda/analysis/simtcheck"
)

// TestAnalyzer checks the wiring, not the subset: which functions are treated
// as kernels, that the import rule is applied per package, that a package
// without kernels is left alone, and that positions and text arrive intact.
//
// What the subset actually is stays pinned by simt's own tests, which drive
// the same lowering without needing a go subprocess. The two would only
// duplicate each other.
//
// testdata carries a go.mod so that analysistest loads the fixtures in module
// mode rather than its GOPATH default. That matters: in GOPATH mode they could
// not import the real gpu package, and type-checking against the real one is
// the entire point of running the lowering from an analysis pass.
func TestAnalyzer(t *testing.T) {
	analysistest.Run(t, analysistest.TestData(), simtcheck.Analyzer,
		"gocuda.test/kernels", "gocuda.test/nokernels")
}
