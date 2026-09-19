//go:build cuda

package simt_test

import (
	"errors"
	"regexp"
	"strings"
	"testing"
	"testing/fstest"

	"github.com/CWBudde/gocuda/cuda"
	"github.com/CWBudde/gocuda/internal/fuzz"
	"github.com/CWBudde/gocuda/simt"
)

// fuzzArch is what the generated C is compiled for, and it matches the
// architecture kernels/prebuilt is filed under so that a fuzz failure and a
// committed artifact are answers to the same question.
const fuzzArch = "compute_75"

// FuzzNVRTCAcceptsTheGeneratedC is the oracle for the other half of the
// defects: valid Go that lowers to C++ which does not compile.
//
// A golden test compares text and a parity test never gets as far as a
// compiler, so this class -- a labelled continue jumping into a later
// declaration's scope, two case clauses declaring one name, a device function
// whose named result nobody emitted -- is invisible to both. Every one of them
// was found by NVRTC and only by NVRTC, and every one of them is an error
// message about code the author never wrote.
//
// It runs over the whole of what the generator produces, not the third the
// host oracle is in scope for: barriers, warp primitives, atomics and shared
// tiles all compile here, and those are where four of the defect catalogue's
// eight ranks live.
//
// This target needs the toolkit but no device: NVRTC compiles to PTX and
// nothing is launched. Without libnvrtc there is no question to answer, so it
// skips -- ci.yml compiles this file and skips the assertion, which is the
// whole reason the build tag is here rather than a runtime check.
//
// That skip is also why fuzz.yml's nvrtc leg sets GOCUDA_REQUIRE_NVRTC. A leg
// that installed the library and then failed to find it would otherwise search
// for nothing and report success; see requireNVRTC.
func FuzzNVRTCAcceptsTheGeneratedC(f *testing.F) {
	for _, s := range seeds {
		f.Add(s[0])
	}
	// Seeds the host oracle skips, so that this target's corpus covers the
	// constructs that exist precisely because it does not share that scope.
	for _, s := range []int64{0, 1, 2, 4, 7, 42} {
		f.Add(s)
	}

	requireNVRTC(f)

	f.Fuzz(func(t *testing.T, seed int64) {
		p := fuzz.Generate(seed)
		src := p.Source()
		fsys := fstest.MapFS{"k.go": &fstest.MapFile{Data: []byte(src)}}

		u, err := simt.Transpile(fsys, p.Name())
		if err != nil {
			t.Fatalf("seed %d: the subset refused a generated program: %v\n%s", seed, err, src)
		}

		ptx, err := cuda.Compile(u.Source, p.Name()+".cu", fuzzArch)
		if err != nil {
			var ce *cuda.CompileError
			if errors.As(err, &ce) {
				t.Fatalf("seed %d: NVRTC refused the generated CUDA C.\n%s\n\n=== Go ===\n%s\n=== C ===\n%s",
					seed, ce.Log, src, u.Source)
			}
			t.Fatalf("seed %d: %v", seed, err)
		}

		// A clean compile that still had something to say is a finding of its
		// own. The premise of the whole track is that a kernel author never
		// sees a message about code they did not write -- and that is how the
		// scoping defect in TestShadowingInitialiserReadsTheOuterVariable
		// surfaced: NVRTC compiled the kernel and remarked that a variable was
		// used before its value was set, which was the only outward sign that
		// "int a = a + 1;" was reading itself.
		if log := remainingWarnings(ptx.Log); log != "" {
			t.Fatalf("seed %d: NVRTC accepted the generated C but warned about it:\n%s\n\n=== Go ===\n%s\n=== C ===\n%s",
				seed, log, src, u.Source)
		}
	})
}

// generatorNoise is the warnings that are true of the generated Go and so say
// nothing about its translation.
//
// The generator writes filler -- a variable declared to be assigned to itself,
// a branch under a constant false, a comparison the types make constant -- so
// that a random program is still a legal Go one. NVRTC is right about every
// one of them, and the author it is describing is a random number generator.
// Anything not on this list is a finding, which is what caught #549-D.
var generatorNoise = map[string]string{
	"#550-D": `"was set but never used": the generator's own "v = v", which is how it keeps Go from refusing an unused variable`,
	"#177-D": `"declared but never referenced": the same filler, where even the self-assignment was not emitted`,
	"#186-D": `"pointless comparison of unsigned integer with zero": a generated comparison the types make constant`,
	"#128-D": `"loop is not reachable": a generated loop under a constant false`,
}

// warningLine matches one diagnostic's first line and captures its number,
// which is the part generatorNoise is keyed on.
var warningLine = regexp.MustCompile(`warning (#\d+-\w)`)

// remainingWarnings strips the noise and returns what is left, or "".
//
// It works over whole lines rather than the whole log because NVRTC follows
// each diagnostic with the source line and a caret, and a filter that dropped
// only the numbered line would leave those behind and report an empty finding.
func remainingWarnings(log string) string {
	var kept []string
	skipping := false
	for _, line := range strings.Split(log, "\n") {
		if m := warningLine.FindStringSubmatch(line); m != nil {
			_, noise := generatorNoise[m[1]]
			skipping = noise
			if noise {
				continue
			}
		} else if skipping && (strings.HasPrefix(line, " ") || strings.TrimSpace(line) == "") {
			continue // the quoted source and caret belonging to a skipped one
		} else if strings.TrimSpace(line) != "" {
			skipping = false
		}
		if !skipping {
			kept = append(kept, line)
		}
	}
	out := strings.TrimSpace(strings.Join(kept, "\n"))
	// NVRTC ends a log that had any warning at all with this, so a log that is
	// nothing but the remark is a log that was entirely noise.
	if strings.HasPrefix(out, "Remark: The warnings can be suppressed") {
		return ""
	}
	return out
}
