// Package nokernels has no kernel in it, so the analyzer must say nothing --
// not even about the import, which is only restricted inside kernel packages.
package nokernels

import "math"

func Pi() float64 { return math.Pi }
