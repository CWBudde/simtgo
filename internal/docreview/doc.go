// Package docreview holds an advisory check on where a piece of writing in
// this repository lives.
//
// docs/README.md states which page owns which subject, and CLAUDE.md gives
// the reason: a claim in the wrong file drifts, because the next person looks
// for it where it belongs and finds an older answer somewhere else. Nothing
// enforced that table, and nothing in it is the kind of rule a regular
// expression can apply -- it is a judgment about what a section is about.
//
// It is a package of its own for the same reason faulttest is: what it checks
// belongs to no other package. SPEC.md is simt's contract and is checked from
// simt; docs/ is the whole repository's engineering record and is not.
//
// Read the measurement in docfiling_test.go before trusting a result. This
// check is the weakest of the three that were built, it is opt-in, and it
// cannot fail a build.
package docreview
