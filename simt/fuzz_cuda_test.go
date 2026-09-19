//go:build cuda

package simt_test

import (
	"errors"
	"regexp"
	"strings"
	"testing"
	"testing/fstest"

	"github.com/CWBudde/simtgo/cuda"
	"github.com/CWBudde/simtgo/internal/fuzz"
	"github.com/CWBudde/simtgo/simt"
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
// That skip is also why fuzz.yml's nvrtc leg sets SIMTGO_REQUIRE_NVRTC. A leg
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
		if log := triagedWarnings(t, ptx.Log, src, u.Source); log != "" {
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
//
// What this list cannot do is tell those apart from a mistranslation that
// happens to produce the same diagnostic number, and each of these numbers has
// such a twin: an emitter that drops a use writes the same "#177-D declared
// but never referenced" as a generator that never wrote one. A number is not
// the distinction the filter needs. That is what triagedWarnings is for, and
// why anything not on this list is still a finding on its own.
var generatorNoise = map[string]string{
	"#550-D": `"was set but never used": the generator's own "v = v", which is how it keeps Go from refusing an unused variable`,
	"#177-D": `"declared but never referenced": the same filler, where even the self-assignment was not emitted`,
	"#186-D": `"pointless comparison of unsigned integer with zero": a generated comparison the types make constant`,
	"#128-D": `"loop is not reachable": a generated loop under a constant false`,
}

// warningLine matches one diagnostic's first line and captures its number,
// which is the part generatorNoise is keyed on.
var warningLine = regexp.MustCompile(`warning (#\d+-\w)`)

// quotedIdent captures the name NVRTC quotes in a diagnostic -- the "scale" in
// `variable "scale" was declared but never referenced`. It is what the Go and
// the C are sliced around before either is shown to a judgment, and it is done
// here rather than there because finding a quoted word is exact work.
var quotedIdent = regexp.MustCompile(`"([A-Za-z_][A-Za-z0-9_]*)"`)

// suppressible is NVRTC's boilerplate tail, printed once whenever the log had
// any warning at all.
const suppressible = "Remark: The warnings can be suppressed"

// warningBlock is one diagnostic together with the lines NVRTC prints under
// it, which are the quoted source and a caret. Blocks exist because a filter
// that dropped only the numbered line would leave those behind and report a
// caret as a finding.
//
// num is empty for a run of lines belonging to no warning.
type warningBlock struct {
	num   string
	lines []string
}

// splitWarnings cuts a log into blocks. A warning block runs from its numbered
// line until a line that is neither indented nor blank, which is where NVRTC
// starts the next thing it has to say.
func splitWarnings(log string) []warningBlock {
	var out []warningBlock
	var cur *warningBlock
	for _, line := range strings.Split(log, "\n") {
		switch {
		case warningLine.MatchString(line):
			out = append(out, warningBlock{num: warningLine.FindStringSubmatch(line)[1], lines: []string{line}})
			cur = &out[len(out)-1]
		case cur != nil && cur.num != "" && (strings.HasPrefix(line, " ") || strings.TrimSpace(line) == ""):
			cur.lines = append(cur.lines, line)
		default:
			if cur == nil || cur.num != "" {
				out = append(out, warningBlock{})
				cur = &out[len(out)-1]
			}
			cur.lines = append(cur.lines, line)
		}
	}
	return out
}

// noise reports whether the allowlist claims this block says nothing about the
// translation. A block that is not a numbered warning is never noise.
func (b warningBlock) noise() bool {
	if b.num == "" {
		return false
	}
	_, ok := generatorNoise[b.num]
	return ok
}

// ident is the name the diagnostic quoted, or "".
func (b warningBlock) ident() string {
	if len(b.lines) == 0 {
		return ""
	}
	if m := quotedIdent.FindStringSubmatch(b.lines[0]); m != nil {
		return m[1]
	}
	return ""
}

func (b warningBlock) String() string { return strings.Join(b.lines, "\n") }

// join renders blocks back into a log, dropping NVRTC's suppressible remark.
//
// The remark is boilerplate and carries nothing, but dropping it is not a
// tidying: it used to be tested for as a prefix of the whole residue, and
// nvcc prints it after the *first* warning rather than at the end. So a log
// whose first warning was noise and whose second was a finding filtered down
// to a residue beginning with the remark, and the finding was discarded with
// it. #549-D -- the diagnostic that caught both halves of the shadowed
// initialiser defect -- is exactly the kind that arrives second.
func join(blocks []warningBlock) string {
	var kept []string
	for _, b := range blocks {
		for _, line := range b.lines {
			if strings.HasPrefix(strings.TrimSpace(line), suppressible) {
				continue
			}
			kept = append(kept, line)
		}
	}
	return strings.TrimSpace(strings.Join(kept, "\n"))
}

// remainingWarnings strips the noise the allowlist knows and returns what is
// left, or "".
func remainingWarnings(log string) string {
	var kept []warningBlock
	for _, b := range splitWarnings(log) {
		if !b.noise() {
			kept = append(kept, b)
		}
	}
	return join(kept)
}

// suppressedWarnings is the other half: the blocks the allowlist dropped.
func suppressedWarnings(log string) []warningBlock {
	var out []warningBlock
	for _, b := range splitWarnings(log) {
		if b.noise() {
			out = append(out, b)
		}
	}
	return out
}
