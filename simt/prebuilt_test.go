package simt

import (
	"strings"
	"testing"
	"testing/fstest"
)

// These tests are in package simt rather than simt_test because the choice
// they pin down -- which registration a device gets, and whether it gets one
// at all -- has no exported spelling: Build answers it, but only with a device
// to ask, and the answers that matter most are the ones on hardware nobody
// running the untagged tests has.

// swapRegistry empties the registry for the duration of a test, so that a
// fabricated entry cannot be served to a kernel a later test builds for real.
func swapRegistry(t testing.TB) {
	t.Helper()
	prebuiltMu.Lock()
	old := prebuilts
	prebuilts = map[string][]Prebuilt{}
	prebuiltMu.Unlock()
	t.Cleanup(func() {
		prebuiltMu.Lock()
		prebuilts = old
		prebuiltMu.Unlock()
	})
}

// The hashes are fabricated: pickPrebuilt only ever compares them, and using
// something no real kernel lowers to keeps these tests from colliding with the
// registrations a generated file makes in its init.
const (
	hashA = "0000000000000000000000000000000000000000000000000000000000000001"
	hashB = "0000000000000000000000000000000000000000000000000000000000000002"
)

func TestPickPrebuiltArch(t *testing.T) {
	swapRegistry(t)
	RegisterPrebuilt(Prebuilt{Name: "K", SourceSHA256: hashA, Arch: "compute_70", PTX: []byte("70")})
	RegisterPrebuilt(Prebuilt{Name: "K", SourceSHA256: hashA, Arch: "compute_75", PTX: []byte("75")})
	RegisterPrebuilt(Prebuilt{Name: "K", SourceSHA256: hashA, Arch: "compute_90", PTX: []byte("90")})

	cases := []struct {
		name         string
		major, minor int
		want         string // the PTX expected, or "" for no usable registration
	}{
		// PTX is forward compatible, so a newer device takes the newest
		// baseline it satisfies and an older one takes what it can still load.
		{"exactly the newest baseline", 9, 0, "90"},
		{"newer than every baseline", 12, 0, "90"},
		{"between two baselines", 8, 6, "75"},
		{"exactly a baseline", 7, 5, "75"},
		{"one minor version below a baseline", 7, 4, "70"},
		{"older than every baseline", 6, 1, ""},
		// A build without the "cuda" tag has no device to ask and reports
		// (0, 0). Nothing is usable, so Build compiles, which is the only
		// thing it could do there anyway.
		{"no device", 0, 0, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, ok := pickPrebuilt(hashA, tc.major, tc.minor)
			if ok != (tc.want != "") {
				t.Fatalf("pickPrebuilt(%d.%d) ok = %v, want %v", tc.major, tc.minor, ok, tc.want != "")
			}
			if ok && string(p.PTX) != tc.want {
				t.Errorf("pickPrebuilt(%d.%d) chose the %s image, want %s", tc.major, tc.minor, p.PTX, tc.want)
			}
		})
	}
}

// A registration whose Arch cannot be read is skipped rather than reported. A
// generated file edited by hand must not be able to stop a kernel from
// running, since compiling it is always still possible.
func TestPickPrebuiltIgnoresUnreadableArch(t *testing.T) {
	swapRegistry(t)
	RegisterPrebuilt(Prebuilt{Name: "K", SourceSHA256: hashA, Arch: "sm_75", PTX: []byte("bad")})
	RegisterPrebuilt(Prebuilt{Name: "K", SourceSHA256: hashA, Arch: "", PTX: []byte("worse")})
	// Arch-conditional PTX is not forward compatible, so cuda.ParseArch
	// refuses it and it may not be handed to a device it was not built for.
	RegisterPrebuilt(Prebuilt{Name: "K", SourceSHA256: hashA, Arch: "compute_90a", PTX: []byte("conditional")})

	if p, ok := pickPrebuilt(hashA, 9, 0); ok {
		t.Fatalf("pickPrebuilt chose %q with no readable architecture registered", p.PTX)
	}

	// The unreadable entries do not poison the key: a sound one registered
	// beside them is still found.
	RegisterPrebuilt(Prebuilt{Name: "K", SourceSHA256: hashA, Arch: "compute_75", PTX: []byte("75")})
	p, ok := pickPrebuilt(hashA, 9, 0)
	if !ok || string(p.PTX) != "75" {
		t.Errorf("pickPrebuilt = %q, %v, want the compute_75 image", p.PTX, ok)
	}
}

// A source nobody registered is simply not found, which is the whole of the
// staleness story: Build then transpiles and compiles as if there were no
// registry at all.
func TestPickPrebuiltUnknownSource(t *testing.T) {
	swapRegistry(t)
	RegisterPrebuilt(Prebuilt{Name: "K", SourceSHA256: hashA, Arch: "compute_75", PTX: []byte("75")})
	if _, ok := pickPrebuilt(hashB, 7, 5); ok {
		t.Error("pickPrebuilt found an image for a source that was never registered")
	}
}

// Registering the same artifact twice has to be harmless: a generated file can
// reach a binary through more than one import path.
func TestRegisterPrebuiltDuplicate(t *testing.T) {
	swapRegistry(t)
	p := Prebuilt{Name: "K", SourceSHA256: hashA, Arch: "compute_75", PTX: []byte("75")}
	RegisterPrebuilt(p)
	RegisterPrebuilt(p)

	prebuiltMu.Lock()
	n := len(prebuilts[hashA])
	prebuiltMu.Unlock()
	if n != 1 {
		t.Errorf("an identical re-registration produced %d entries, want 1", n)
	}
}

// Two different images for one source and architecture cannot both be right,
// so the disagreement is raised where it is caused rather than resolved by
// whichever init ran first.
func TestRegisterPrebuiltConflict(t *testing.T) {
	swapRegistry(t)
	RegisterPrebuilt(Prebuilt{Name: "K", SourceSHA256: hashA, Arch: "compute_75", PTX: []byte("75")})

	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("registering different PTX for one source and arch did not panic")
		}
		if msg, _ := r.(string); !strings.Contains(msg, "K") {
			t.Errorf("panic = %v, want a message naming the kernel", r)
		}
	}()
	RegisterPrebuilt(Prebuilt{Name: "K", SourceSHA256: hashA, Arch: "compute_75", PTX: []byte("something else")})
}

// VerifyPrebuilt is what a user's own test calls to find out that "go
// generate" has not been run since the kernels changed.
func TestVerifyPrebuilt(t *testing.T) {
	swapRegistry(t)
	fsys := fstest.MapFS{"k.go": &fstest.MapFile{Data: []byte(`package kernels

import "github.com/CWBudde/gocuda/gpu"

func Scale(ctx gpu.Ctx, y, x []float32, a float32) {
	i := ctx.GlobalID()
	if i < len(y) {
		y[i] = a * x[i]
	}
}
`)}}

	err := VerifyPrebuilt(fsys)
	if err == nil {
		t.Fatal("VerifyPrebuilt found a prebuilt for a source nothing registered")
	}
	// The message has to name the kernel and say what to do about it: it is
	// read by someone who has just seen a test fail in a package they did not
	// write.
	for _, want := range []string{"Scale", "go generate ./..."} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("VerifyPrebuilt error = %q, want it to mention %q", err, want)
		}
	}

	u, err := Transpile(fsys, "Scale")
	if err != nil {
		t.Fatalf("Transpile: %v", err)
	}
	RegisterPrebuilt(Prebuilt{Name: "Scale", SourceSHA256: u.SourceHash, Arch: "compute_75", PTX: []byte("75")})
	if err := VerifyPrebuilt(fsys, "Scale"); err != nil {
		t.Errorf("VerifyPrebuilt after registering the current source: %v", err)
	}

	// A registration is matched on the source alone. The architectures it was
	// built for decide what a device can load, not whether the artifact is
	// current, and this check runs without a device.
	if err := VerifyPrebuilt(fsys); err != nil {
		t.Errorf("VerifyPrebuilt over every declared kernel: %v", err)
	}
}
