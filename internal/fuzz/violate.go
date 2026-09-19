package fuzz

import (
	"fmt"
	"strings"
)

// A Violation is one edit that takes an accepted program outside the subset,
// together with the diagnostic the subset owes it.
//
// The point of applying these to *generated* programs rather than to the
// minimal snippets simt/errors_test.go uses is that a refusal which only fires
// in a three-line kernel is not a refusal. Every rule here has to hold with a
// few hundred lines of unrelated control flow around it, under whatever names
// and types the generator happened to choose -- which is the shape real kernels
// have and the shape a rule quietly stops firing in.
type Violation struct {
	// Name says which rule is being broken, for a failure to quote.
	Name string
	// Want is a substring of the diagnostic. It is a substring rather than the
	// whole message because the message carries positions and identifiers from
	// the surrounding program, and pinning those would be pinning the
	// generator rather than the rule.
	Want string

	// edit rewrites the source. It is given the program so that an edit can
	// refer to the kernel by name, which is the one thing about a generated
	// program that every edit needs.
	edit func(src string, p *Program) string
}

// Violations are the edits Violate can apply, in a fixed order so that a fuzz
// failure carrying an index stays reproducible as the list grows at the end.
//
// Each one is deliberately independent of what the program contains: nothing
// here inspects the kernel's parameters or its body, because an edit that only
// applies to programs of a certain shape would silently stop being applied.
var Violations = []Violation{{
	Name: "an import other than gpu",
	Want: "may not import",
	edit: func(src string, _ *Program) string {
		src = strings.Replace(src,
			`import "github.com/CWBudde/simtgo/gpu"`,
			"import (\n\t\"math\"\n\n\t\"github.com/CWBudde/simtgo/gpu\"\n)", 1)
		return src + "\nfunc simtgoViolation(x float32) float32 { return x * float32(math.Pi) }\n"
	},
}, {
	Name: "a recursive device function",
	Want: "recursion is not supported in kernels",
	edit: func(src string, p *Program) string {
		src += "\nfunc simtgoViolation(x int32) int32 {\n" +
			"\tif x <= 0 {\n\t\treturn 0\n\t}\n\treturn simtgoViolation(x - 1)\n}\n"
		return inject(src, p, "\tsimtgoV := simtgoViolation(3)\n\tsimtgoV = simtgoV\n")
	},
}, {
	Name: "calling another kernel",
	Want: "is a kernel",
	edit: func(src string, p *Program) string {
		src += "\nfunc SimtgoViolation(ctx gpu.Ctx, v []float32) {\n\tv[0] = 1\n}\n"
		return inject(src, p, "\tSimtgoViolation(ctx, nil)\n")
	},
}, {
	Name: "a package-level variable",
	Want: "declared outside the kernel",
	edit: func(src string, p *Program) string {
		src += "\nvar simtgoViolation int32\n"
		return inject(src, p, "\tsimtgoV := simtgoViolation\n\tsimtgoV = simtgoV\n")
	},
}, {
	Name: "a barrier under a thread-varying condition",
	Want: "is under a thread-varying if",
	edit: func(src string, p *Program) string {
		return inject(src, p, "\tif ctx.ThreadIdx() == 0 {\n\t\tctx.SyncThreads()\n\t}\n")
	},
}}

// Violate applies the i'th violation to the program's source.
//
// It returns the mutated source and the violation, so a caller that finds the
// subset accepting it can say which rule stopped holding rather than only that
// something did.
func Violate(p *Program, i int) (string, Violation) {
	v := Violations[((i%len(Violations))+len(Violations))%len(Violations)]
	return v.edit(p.Source(), p), v
}

// inject puts statements at the top of the kernel body.
//
// The top is where they are least likely to be unreachable: the generator
// writes guards around most of what it emits, and a statement placed after one
// of them could be refused for a rule nobody was testing, or -- worse -- lower
// cleanly because nothing ever reaches it.
//
// A self-assignment follows every value the injected code produces, which is
// the same thing the generator does with a value it has no further use for:
// Go refuses an unused variable, and the blank identifier is outside the
// subset, so this is the one spelling that is legal in both languages.
func inject(src string, p *Program, stmts string) string {
	head := fmt.Sprintf("func %s(ctx gpu.Ctx", p.Main.Name)
	at := strings.Index(src, head)
	if at < 0 {
		panic("fuzz: the generated source does not declare its own kernel: " + head)
	}
	brace := strings.Index(src[at:], " {\n")
	if brace < 0 {
		panic("fuzz: the generated kernel has no body: " + head)
	}
	cut := at + brace + len(" {\n")
	return src[:cut] + stmts + src[cut:]
}
