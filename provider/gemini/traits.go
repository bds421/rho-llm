package gemini

import (
	"strings"

	llm "github.com/bds421/rho-llm"
)

// finishContinuation is the finishReason of a long-decode hop that has more
// to say (Gemini 4; aistudio.google.com/docs/long-decode, 2026-11-18).
const finishContinuation = "CONTINUATION"

// modelTraits are the per-model request rules of the Gemini API that the
// registry does not carry. Sources: aistudio.google.com/docs/whats-new-gemini-3.8,
// whats-new-gemini-4-carbon, whats-new-gemini-4-hydrogen and long-decode
// (read 2026-11-18).
type modelTraits struct {
	// noMinimalThinking: thinkingLevel "minimal" is an error (3.8 Flash,
	// which 3.7 Flash is routed to, and every Gemini 4 model).
	noMinimalThinking bool
	// xhighThinking: thinkingLevel "xhigh" exists (Gemini 4 Hydrogen).
	xhighThinking bool
	// longDecode: a turn may continue past one request's output limit with
	// finishReason CONTINUATION and a continuationToken (Gemini 4).
	longDecode bool
	// noExplicitCache: cachedContent is an error; caching is implicit only
	// (Gemini 4).
	noExplicitCache bool
}

// traitsFor returns the request rules for model (an ID or a registry alias).
// A Gemini 4 model not yet registered gets the Gemini 4 rules by prefix.
func traitsFor(model string) modelTraits {
	id := llm.ResolveModelAlias(model)
	gemini4 := strings.HasPrefix(id, "gemini-4-")
	return modelTraits{
		noMinimalThinking: gemini4 || id == "gemini-3.8-flash" || id == "gemini-3.7-flash",
		xhighThinking:     id == "gemini-4-hydrogen",
		longDecode:        gemini4,
		noExplicitCache:   gemini4,
	}
}
