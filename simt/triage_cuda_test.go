//go:build cuda

package simt_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/CWBudde/simtgo/internal/typesafe"
)

// TriageEnv turns the warning triage on. It is off by default and wired into
// the nightly fuzz workflow rather than ci.yml, because it reaches a network
// service and ci.yml must keep running on a machine with nothing installed.
const TriageEnv = "SIMTGO_WARNING_TRIAGE"

// triageContext is how many lines either side of a mention are shown.
//
// The number is not arbitrary. The same question about the same defect was
// answered at confidence 0.29 when it was asked over a whole generated
// program and 0.93 when it was asked over this window, on the same day
// against the same model. A generated program is mostly unrelated helpers,
// and unrelated detail is what costs the answer.
const triageContext = 3

// triagedWarnings is remainingWarnings plus a second look at what the
// allowlist dropped.
//
// The direction matters more than the mechanism. The judgment may only put a
// warning back; it is never asked whether a warning that survived the
// allowlist should be dropped, so the refusals this gate already makes cannot
// regress, and a kernel is never declared correct by a model. Everything that
// can go wrong -- no key, no network, a timeout, a malformed answer -- lands
// on today's behaviour rather than on a failure, which is the same bargain the
// CUDA tests strike with a missing device.
// triageDown is set the first time the service could not be reached.
//
// Without it, an unreachable service is worse than a disabled one: the
// generator's filler produces an allowlisted warning constantly, so every
// input carrying one would open its own 90-second budget and a 20-minute
// search would spend most of itself waiting instead of fuzzing. One failure
// is enough to know; the rest of the process falls back to the plain NVRTC
// oracle, which is what the fuzzer had before any of this.
var triageDown atomic.Bool

func triagedWarnings(t *testing.T, log, goSrc, cSrc string) string {
	t.Helper()
	kept := remainingWarnings(log)

	if os.Getenv(TriageEnv) == "" || !typesafe.Available() || triageDown.Load() {
		return kept
	}
	suppressed := suppressedWarnings(log)
	if len(suppressed) == 0 {
		return kept
	}

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	raised := []string{kept}
	for _, b := range suppressed {
		verdict, confidence, err := judgeWarning(ctx, b, goSrc, cSrc)
		if err != nil {
			// A judgment nobody can reach is a judgment not made. Say so once,
			// stop asking for the rest of the process, and leave the
			// allowlist's answer standing.
			if !triageDown.Swap(true) {
				t.Logf("warning triage unavailable, disabled for the rest of this process "+
					"(%s left suppressed): %v", b.num, err)
			}
			break
		}
		if verdict != "mistranslation" {
			continue
		}
		raised = append(raised, fmt.Sprintf(
			"%s\n\t^ %s is on the generator-noise allowlist, but the Go source does not appear to\n"+
				"\t  say what this warning describes (confidence %.2f). The allowlist keys on the\n"+
				"\t  diagnostic number, which cannot tell this case from generated filler.",
			b, b.num, confidence))
	}

	out := strings.TrimSpace(strings.Join(raised, "\n\n"))
	return out
}

// judgeWarning asks whether one suppressed warning describes something the Go
// source says too.
//
// Both sources are cut down before either is sent, because the measurement
// that justifies this whole path was a measurement about that: unrelated
// detail is what costs the answer.
func judgeWarning(ctx context.Context, b warningBlock, goSrc, cSrc string) (verdict string, confidence float64, err error) {
	goPart, cPart, err := slice(b, goSrc, cSrc)
	if err != nil {
		return "", 0, err
	}
	res, err := typesafe.Ask(ctx,
		map[string]any{
			"go_kernel_source": goPart,
			"generated_cuda_c": cPart,
			"compiler_warning": b.String(),
		},
		map[string]typesafe.Question{
			"verdict": typesafe.Choice(
				map[string]any{
					"question": "The CUDA C in `generated_cuda_c` was produced automatically from `go_kernel_source`. " +
						"The CUDA compiler emitted `compiler_warning` about the C. What does the warning tell us?",
					"focus": "Compare the construct the warning names against the Go source it was translated from.",
				},
				map[string]any{
					"faithful": map[string]any{
						"what": "The Go source already contains what the warning describes, so the C is a " +
							"faithful translation and the warning is about the author's own program.",
						"example": "The Go writes `v = v` and the warning says v was set but never used.",
					},
					"mistranslation": map[string]any{
						"what": "The Go source does not contain what the warning describes; the C says " +
							"something the Go does not, so the translation changed the program's meaning.",
						"example": "The Go reads a variable in an expression and the C declares it and never reads it.",
					},
				}),
		})
	if err != nil {
		return "", 0, err
	}
	a, ok := res.Answers["verdict"]
	if !ok {
		return "", 0, errors.New("no verdict in the response")
	}
	return a.Choice, a.Confidence, nil
}

// warningLoc captures the line number NVRTC reports, the 4 in "k.cu(4):".
var warningLoc = regexp.MustCompile(`\((\d+)\):`)

// goSrcCap bounds a whole-program fallback. It is generous rather than tuned:
// the point of the cap is that a runaway generated program cannot turn one
// judgment into a very large request, not that this is a good size to ask at.
const goSrcCap = 8000

// slice cuts the two sources down to what the warning is about.
//
// There are two handles and NVRTC gives whichever it has. Most diagnostics
// quote the name they are about -- the "scale" in `variable "scale" was
// declared but never referenced` -- and most such names exist on both sides.
// But #186-D and #128-D, two of the four numbers on the allowlist, quote
// nothing; they report a line instead. A line number locates the C exactly
// and the Go not at all, because no mapping from generated C back to Go
// source exists in this repository, so the C is cut around it and the Go goes
// whole, capped.
//
// A name NVRTC quotes is the **C** spelling, which is not always the Go one.
// A Go variable named after a C++ keyword is emitted with a trailing
// underscore (`cname` in internal/lower), and the fuzz generator picks from
// `keywordPool` one time in five, so `class` in the Go is `class_` in the C
// and in the diagnostic. Slicing the Go by the C spelling would find nothing
// there and hand over an empty Go source -- and a judgment shown no Go at all
// has every reason to call faithful filler a mistranslation, which would fail
// a nightly fuzz run over nothing. So the Go is tried by the quoted name and
// then by the name with one trailing underscore removed, and if neither finds
// anything the whole capped source goes rather than an empty string. The same
// fallback covers every other name the emitter invents and the Go never had,
// `y_len` among them.
func slice(b warningBlock, goSrc, cSrc string) (goPart, cPart string, err error) {
	if ident := b.ident(); ident != "" {
		cPart = mentions(cSrc, ident, triageContext)
		for _, name := range []string{ident, strings.TrimSuffix(ident, "_")} {
			if goPart = mentions(goSrc, name, triageContext); goPart != "" {
				break
			}
		}
		if goPart == "" {
			goPart = truncate(goSrc, goSrcCap)
		}
		if cPart == "" {
			cPart = truncate(cSrc, goSrcCap)
		}
		return goPart, cPart, nil
	}
	if len(b.lines) > 0 {
		if m := warningLoc.FindStringSubmatch(b.lines[0]); m != nil {
			line, convErr := strconv.Atoi(m[1])
			if convErr == nil {
				return truncate(goSrc, goSrcCap), around(cSrc, line, triageContext), nil
			}
		}
	}
	return "", "", errors.New("diagnostic names neither an identifier nor a line")
}

// around returns the 1-based line of src with n either side.
func around(src string, line, n int) string {
	lines := strings.Split(src, "\n")
	lo, hi := max(line-1-n, 0), min(line-1+n, len(lines)-1)
	if lo > hi {
		return src
	}
	var b strings.Builder
	if lo > 0 {
		b.WriteString("// ...\n")
	}
	b.WriteString(strings.Join(lines[lo:hi+1], "\n"))
	if hi < len(lines)-1 {
		b.WriteString("\n// ...")
	}
	return b.String()
}

// truncate cuts src to n bytes on a line boundary.
func truncate(src string, n int) string {
	if len(src) <= n {
		return src
	}
	cut := src[:n]
	if i := strings.LastIndexByte(cut, '\n'); i > 0 {
		cut = cut[:i]
	}
	return cut + "\n// ... truncated"
}

// mentions returns the lines of src naming ident, each with n lines either
// side, elisions marked. Filtering here rather than in the question is the
// half of this design that the measurement actually cared about.
//
// The match is on whole identifiers, and that is not a detail. Half of the
// generator's `namePool` is single letters, so a substring test for "a" holds
// on almost every line of any program -- the window would quietly become the
// whole source, the filtering would buy nothing, and the measurement this is
// built on would not describe what the code does. Go's \b counts "_" as a
// word character, so \bclass\b does not match inside `class_` and \blen\b
// does not match inside `y_len`, which is exactly the distinction wanted.
func mentions(src, ident string, n int) string {
	if ident == "" {
		return ""
	}
	word := regexp.MustCompile(`\b` + regexp.QuoteMeta(ident) + `\b`)
	lines := strings.Split(src, "\n")
	keep := make(map[int]bool, len(lines))
	for i, line := range lines {
		if !word.MatchString(line) {
			continue
		}
		for j := max(i-n, 0); j <= min(i+n, len(lines)-1); j++ {
			keep[j] = true
		}
	}
	var out []string
	last := -2
	for i := range lines {
		if !keep[i] {
			continue
		}
		if i != last+1 {
			out = append(out, "// ...")
		}
		out = append(out, lines[i])
		last = i
	}
	return strings.Join(out, "\n")
}

// triageCase is one labelled (warning, Go, C) triple.
type triageCase struct {
	name  string
	log   string
	goSrc string
	cSrc  string
	ident string
	noise bool // the warning is true of the Go source too
}

// triageCases are the corpus the policy was measured on. Every one of them
// carries a diagnostic number the allowlist suppresses, so the allowlist alone
// scores zero on the findings here by construction -- that is the point.
//
// The findings are written to sit in the blind spot rather than sampled from
// real fuzz output, which measures the blind spot honestly and says nothing
// about how often such a defect occurs.
var triageCases = []triageCase{{
	name:  "self-assignment filler",
	log:   "k.cu(5): warning #550-D: variable \"v\" was set but never used\n",
	goSrc: "func K(ctx gpu.Ctx, y []float32) {\n\ti := ctx.GlobalIdxX()\n\tv := 3\n\tv = v\n\tif i < len(y) {\n\t\ty[i] = 1\n\t}\n}",
	cSrc:  "extern \"C\" __global__ void K(float* __restrict__ y, int y_len)\n{\n\tint i = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n\tint v = 3;\n\tv = v;\n\tif (i < y_len)\n\t{\n\t\ty[i] = 1.0f;\n\t}\n}",
	ident: "v", noise: true,
}, {
	name:  "unreferenced filler variable",
	log:   "k.cu(4): warning #177-D: variable \"t3\" was declared but never referenced\n",
	goSrc: "func K(ctx gpu.Ctx, y []float32) {\n\ti := ctx.GlobalIdxX()\n\tvar t3 int32\n\t_ = t3\n\tif i < len(y) {\n\t\ty[i] = 5\n\t}\n}",
	cSrc:  "extern \"C\" __global__ void K(float* __restrict__ y, int y_len)\n{\n\tint i = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n\tint t3 = 0;\n\tif (i < y_len)\n\t{\n\t\ty[i] = 5.0f;\n\t}\n}",
	ident: "t3", noise: true,
}, {
	name:  "unsigned compared with zero, as the Go wrote it",
	log:   "k.cu(4): warning #186-D: pointless comparison of unsigned integer with zero\n",
	goSrc: "func K(ctx gpu.Ctx, y []float32, n uint32) {\n\ti := ctx.GlobalIdxX()\n\tif n >= 0 {\n\t\tif i < len(y) {\n\t\t\ty[i] = 2\n\t\t}\n\t}\n}",
	cSrc:  "extern \"C\" __global__ void K(float* __restrict__ y, int y_len, unsigned int n)\n{\n\tint i = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n\tif (n >= 0u)\n\t{\n\t\tif (i < y_len) { y[i] = 2.0f; }\n\t}\n}",
	ident: "n", noise: true,
}, {
	name:  "a use the emitter dropped",
	log:   droppedUseLog,
	goSrc: "func K(ctx gpu.Ctx, y []float32, x []float32) {\n\ti := ctx.GlobalIdxX()\n\tscale := float32(2)\n\tif i < len(y) {\n\t\ty[i] = scale * x[i]\n\t}\n}",
	cSrc:  "extern \"C\" __global__ void K(float* __restrict__ y, int y_len, const float* __restrict__ x, int x_len)\n{\n\tint i = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n\tfloat scale = 2.0f;\n\tif (i < y_len)\n\t{\n\t\ty[i] = x[i];\n\t}\n}",
	ident: "scale",
}, {
	name:  "a store the emitter dropped",
	log:   "k.cu(4): warning #550-D: variable \"sum\" was set but never used\n",
	goSrc: "func K(ctx gpu.Ctx, y []float32, x []float32) {\n\ti := ctx.GlobalIdxX()\n\tsum := float32(0)\n\tfor j := 0; j < 4; j++ {\n\t\tsum += x[j]\n\t}\n\tif i < len(y) {\n\t\ty[i] = sum\n\t}\n}",
	cSrc:  "extern \"C\" __global__ void K(float* __restrict__ y, int y_len, const float* __restrict__ x, int x_len)\n{\n\tint i = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n\tfloat sum = 0.0f;\n\tfor (int j = 0; j < 4; j++)\n\t{\n\t\tsum += x[j];\n\t}\n\tif (i < y_len)\n\t{\n\t\ty[i] = x[i];\n\t}\n}",
	ident: "sum",
}, {
	name:  "an unsigned comparison the Go never wrote",
	log:   "k.cu(4): warning #186-D: pointless comparison of unsigned integer with zero\n",
	goSrc: "func K(ctx gpu.Ctx, y []float32, n int32) {\n\ti := ctx.GlobalIdxX()\n\tif n >= 0 {\n\t\tif i < len(y) {\n\t\t\ty[i] = 2\n\t\t}\n\t}\n}",
	cSrc:  "extern \"C\" __global__ void K(float* __restrict__ y, int y_len, int n)\n{\n\tint i = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n\tif ((unsigned int)(n) >= 0u)\n\t{\n\t\tif (i < y_len) { y[i] = 2.0f; }\n\t}\n}",
	ident: "n",
}}

// TestTriageSeparatesNoiseFromMistranslation runs the corpus through the
// judgment the fuzz gate uses.
//
// It skips without a key, the way every test here skips without a device: a
// question nobody can ask is not a failure. What it asserts is the property
// the policy rests on -- that genuine generator noise is never re-raised --
// and reports the findings it recovered rather than demanding all of them,
// because the allowlist recovers none of them and any number above zero is
// the improvement being claimed.
func TestTriageSeparatesNoiseFromMistranslation(t *testing.T) {
	if os.Getenv(TriageEnv) == "" {
		t.Skipf("%s is not set", TriageEnv)
	}
	if !typesafe.Available() {
		t.Skipf("%s is not set", typesafe.KeyEnv)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	var recovered, findings, asked, unreachable int
	for _, tc := range triageCases {
		blocks := suppressedWarnings(tc.log)
		if len(blocks) != 1 {
			t.Fatalf("%s: the corpus case is not on the allowlist", tc.name)
		}
		verdict, confidence, err := judgeWarning(ctx, blocks[0], tc.goSrc, tc.cSrc)
		if err != nil {
			// One unreachable judgment is not a verdict on the rest.
			t.Logf("%-46s unreachable: %v", tc.name, err)
			unreachable++
			continue
		}
		asked++
		t.Logf("%-46s noise=%-5t verdict=%-14s confidence=%.2f", tc.name, tc.noise, verdict, confidence)

		if tc.noise {
			if verdict == "mistranslation" {
				t.Errorf("%s: genuine generator noise was re-raised as a finding (confidence %.2f). "+
					"The gate becomes noisy, which is what the policy is meant to prevent.", tc.name, confidence)
			}
			continue
		}
		findings++
		if verdict == "mistranslation" {
			recovered++
		}
	}
	if asked == 0 {
		t.Skipf("no case could be asked about (%d unreachable)", unreachable)
	}
	t.Logf("asked %d of %d cases; recovered %d of %d mistranslations the allowlist suppresses, "+
		"which the allowlist alone recovers none of", asked, len(triageCases), recovered, findings)
	if findings > 0 && recovered == 0 {
		t.Error("the triage recovered nothing the allowlist misses, so it is buying nothing")
	}
}
