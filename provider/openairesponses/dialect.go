package openairesponses

import (
	"net/url"
	"strings"

	llm "github.com/bds421/rho-llm"
)

// dialect holds the per-host differences from the OpenAI Responses API, as
// data. The default (zero) dialect is OpenAI's.
type dialect struct {
	// name is the Provider() name when Config.ProviderName is unset.
	name string
	// systemAsInstructions sends Request.System as the top-level
	// "instructions" field instead of a system input item.
	systemAsInstructions bool
	// keepMaxEffort sends ThinkingMax as reasoning.effort "max" instead of
	// OpenAI's deepest, "xhigh".
	keepMaxEffort bool
	// rootBaseURLPath is appended to a base URL given as a bare host, so a
	// BaseURL written for the old Chat Completions endpoint still reaches
	// the Responses route.
	rootBaseURLPath string
}

// perplexityDialect applies to the opt-in "perplexity-agent" preset and to any
// BaseURL on api.perplexity.ai reached through this adapter; the "perplexity"
// preset (Chat Completions) never reaches it. The Agent API (POST /v1/agent, alias POST /v1/responses;
// docs.perplexity.ai/api-reference/agent-post, 2026-11-18). System prompts
// belong in "instructions" (the documented place); reasoning.effort accepts
// minimal..max.
var perplexityDialect = dialect{
	name:                 "perplexity-agent",
	systemAsInstructions: true,
	keepMaxEffort:        true,
	rootBaseURLPath:      "/v1",
}

func dialectFor(cfg llm.Config) dialect {
	if cfg.Provider == "perplexity-agent" {
		return perplexityDialect
	}
	if u, err := url.Parse(cfg.BaseURL); err == nil && strings.EqualFold(u.Hostname(), "api.perplexity.ai") {
		return perplexityDialect
	}
	return dialect{}
}

// fixBaseURL appends the dialect's root path to a base URL that is just a host.
func (d dialect) fixBaseURL(base string) string {
	if d.rootBaseURLPath == "" {
		return base
	}
	u, err := url.Parse(base)
	if err != nil || u.Host == "" || strings.Trim(u.Path, "/") != "" {
		return base
	}
	return strings.TrimRight(base, "/") + d.rootBaseURLPath
}
