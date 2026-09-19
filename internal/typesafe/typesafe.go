// Package typesafe is a minimal client for the TypeSafe System One API, used
// by tests that need a judgment rather than a parse.
//
// It exists because two checks in this repository ask a question no regular
// expression answers: whether an NVRTC warning about generated CUDA C
// describes something the Go source also says, and whether a sentence in
// SPEC.md still describes the rule its test pins. Both are semantic, both are
// advisory or fail-closed, and neither may reach the library.
//
// Three properties are load-bearing, in the order they matter.
//
// Nothing here is imported outside a _test.go file, so the library's surface
// and its dependency list are unchanged: this is stdlib only, and go.mod does
// not gain an entry. It sits in internal/ next to fuzz and faulttest, which
// are test-support packages for the same reason.
//
// A caller that cannot reach the service must carry on as if it had not
// asked. ErrNotConfigured says the key is absent, and every caller treats it
// -- and every transport error -- the way the CUDA tests treat a missing
// device: skip the extra question, keep the rest of the test.
//
// And an answer is evidence, never proof. The questions asked through this
// package can add a finding for a human to read. None of them may decide that
// generated CUDA C is correct.
package typesafe

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

// Endpoint is the evaluation endpoint. Model is the flagship alias rather than
// a pinned version: a judgment here is advisory, and the alternative is a
// version string that goes stale silently.
const (
	Endpoint = "https://api.typesafe.ai/v1/systemone"
	Model    = "jev-latest"

	// KeyEnv holds the API key. The name is TypeSafe's own.
	KeyEnv = "TYPESAFE_API_KEY"
)

// ErrNotConfigured means no API key was in the environment. It is not a
// failure: a caller reports it the way it reports a missing CUDA device.
var ErrNotConfigured = errors.New("typesafe: " + KeyEnv + " is not set")

// Question is one typed question. Type is "noul" or "choice"; Instructions and
// Criteria take a string or any JSON shape, because a rubric with a
// what/not_for/example split reads better than the same text run together.
type Question struct {
	Type         string `json:"type"`
	Instructions any    `json:"instructions"`
	Criteria     any    `json:"criteria,omitempty"`
}

// Noul builds a yes/no question. yes and no describe the two ends; they are
// what stops the model answering a question next to the one that was meant.
func Noul(instructions any, yes, no string) Question {
	return Question{
		Type:         "noul",
		Instructions: instructions,
		Criteria:     map[string]string{"true": yes, "false": no},
	}
}

// Choice builds a one-of-a-set question. criteria maps each option to its
// rubric.
func Choice(instructions any, criteria map[string]any) Question {
	return Question{Type: "choice", Instructions: instructions, Criteria: criteria}
}

// Answer is one answer. Which fields are set follows the question's type:
// Noul sets Noul, Choice sets Choice, Probabilities and Confidence.
//
// Confidence summarises how concentrated Probabilities is, and says nothing
// about whether the answer is right. Callers here use it in one direction
// only -- a low confidence withholds an action, a high one never compels it.
type Answer struct {
	Type          string             `json:"type"`
	Noul          float64            `json:"noul"`
	Choice        string             `json:"choice"`
	Probabilities map[string]float64 `json:"probabilities"`
	Confidence    float64            `json:"confidence"`
}

// Result is one response: an answer per question, under the keys asked.
type Result struct {
	Model   string            `json:"model"`
	Answers map[string]Answer `json:"answers"`
	Usage   struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

type request struct {
	State     any                 `json:"state"`
	Model     string              `json:"model"`
	Questions map[string]Question `json:"questions"`
}

// Available reports whether a key is set, so a test can skip before building
// any state.
func Available() bool { return os.Getenv(KeyEnv) != "" }

// Ask evaluates state against questions.
//
// state should carry the fields the questions name and nothing else. The
// service's own guidance is that accuracy falls as unrelated detail grows,
// and the fuzz triage measured it: the same question about the same defect
// answered at confidence 0.29 inside a whole generated program and 0.93 with
// the relevant lines sliced out. Filtering is the caller's job because the
// caller is the one that knows what is relevant.
func Ask(ctx context.Context, state any, questions map[string]Question) (*Result, error) {
	key := os.Getenv(KeyEnv)
	if key == "" {
		return nil, ErrNotConfigured
	}
	body, err := json.Marshal(request{State: state, Model: Model, Questions: questions})
	if err != nil {
		return nil, fmt.Errorf("typesafe: encoding request: %w", err)
	}

	// 429 and 529 are the documented back-off codes; 5xx is retried with them
	// because a judgment nobody is waiting on is worth one more try, and
	// because the alternative is a flaky test for a reason unrelated to CUDA.
	var last error
	for attempt := range 4 {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(time.Duration(1<<attempt) * time.Second):
			}
		}
		res, retry, err := post(ctx, key, body)
		if err == nil {
			return res, nil
		}
		last = err
		if !retry {
			return nil, err
		}
	}
	return nil, last
}

// post makes one attempt and says whether another is worth making.
func post(ctx context.Context, key string, body []byte) (*Result, bool, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, Endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, false, fmt.Errorf("typesafe: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+key)
	req.Header.Set("Content-Type", "application/json")

	resp, err := (&http.Client{Timeout: 60 * time.Second}).Do(req)
	if err != nil {
		return nil, true, fmt.Errorf("typesafe: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	payload, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, true, fmt.Errorf("typesafe: reading response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		retry := resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500
		return nil, retry, fmt.Errorf("typesafe: status %d: %s", resp.StatusCode, bytes.TrimSpace(payload))
	}
	var out Result
	if err := json.Unmarshal(payload, &out); err != nil {
		return nil, false, fmt.Errorf("typesafe: decoding response: %w", err)
	}
	return &out, false, nil
}
