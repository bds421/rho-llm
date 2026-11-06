package llm

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// =============================================================================
// SERVER TOOLS
// =============================================================================

// Server tool names (ServerTool.Name).
const (
	ServerToolWebSearch = "web_search"
)

// Web search tool versions (ServerTool.Type). The 20260209 and later versions
// filter results with code execution ("dynamic filtering") and default
// allowed_callers to the code-execution caller; set AllowedCallers to
// []string{CallerDirect} for models without programmatic tool calling or to
// keep a request eligible for zero data retention.
const (
	WebSearch20250305 = "web_search_20250305"
	WebSearch20260209 = "web_search_20260209"
	WebSearch20260318 = "web_search_20260318"
)

// CallerDirect is the allowed_callers value for a tool the model calls
// itself, without code execution.
const CallerDirect = "direct"

// ServerTool is a tool the provider runs itself (Anthropic server tools such
// as web search). The model's call and the result come back as
// ContentServerToolUse / ContentServerToolResult parts of the same assistant
// turn; the caller never answers them with a tool_result.
//
// Type is the provider's versioned tool type (WebSearch20260209, …) and Name
// its fixed name (ServerToolWebSearch). The typed fields cover the web tools;
// Params carries any further version-specific field (for example web fetch's
// "max_content_tokens" or web search's "response_inclusion") and may not
// repeat a typed one. Adapters without server tools reject a Request that
// sets any.
//
// A ServerTool encodes to the same bytes every time, so a definition reused
// across pause_turn continuations keeps the prompt cache valid.
type ServerTool struct {
	Type           string         `json:"type"`
	Name           string         `json:"name"`
	MaxUses        int            `json:"max_uses,omitempty"`
	AllowedDomains []string       `json:"allowed_domains,omitempty"`
	BlockedDomains []string       `json:"blocked_domains,omitempty"`
	UserLocation   *UserLocation  `json:"user_location,omitempty"`
	AllowedCallers []string       `json:"allowed_callers,omitempty"`
	CacheControl   bool           `json:"cache_control,omitempty"`
	CacheTTL       string         `json:"cache_ttl,omitempty"`
	Params         map[string]any `json:"params,omitempty"`
}

// UserLocation localizes web search results. Type is "approximate" (filled in
// by the adapter when empty); Country is an ISO 3166-1 alpha-2 code and
// Timezone an IANA zone ID. Set at least one of the four.
type UserLocation struct {
	Type     string `json:"type,omitempty"`
	City     string `json:"city,omitempty"`
	Region   string `json:"region,omitempty"`
	Country  string `json:"country,omitempty"`
	Timezone string `json:"timezone,omitempty"`
}

// NewWebSearchTool returns a web search ServerTool of the given version
// (WebSearch20260209, …) capped at maxUses searches per turn (0 = no cap).
func NewWebSearchTool(version string, maxUses int) ServerTool {
	return ServerTool{Type: version, Name: ServerToolWebSearch, MaxUses: maxUses}
}

// =============================================================================
// SERVER TOOL RESULTS AND CITATIONS
// =============================================================================

// WebSearchResult is one result of a web_search_tool_result block.
// EncryptedContent is opaque and must reach the provider unchanged on the
// next request; the ContentPart's ProviderBlock carries it byte for byte.
type WebSearchResult struct {
	URL              string `json:"url"`
	Title            string `json:"title,omitempty"`
	PageAge          string `json:"page_age,omitempty"`
	EncryptedContent string `json:"encrypted_content,omitempty"`
}

// ServerToolError is the error form of a server tool result block: the API
// still answers HTTP 200 and puts the error in the block (for web search:
// too_many_requests, invalid_tool_input, max_uses_exceeded, query_too_long,
// request_too_large, unavailable).
type ServerToolError struct {
	Type      string `json:"type"`
	ErrorCode string `json:"error_code"`
}

// Citation types (Citation.Type).
const (
	CitationCharLocation            = "char_location"
	CitationPageLocation            = "page_location"
	CitationContentBlockLocation    = "content_block_location"
	CitationSearchResultLocation    = "search_result_location"
	CitationWebSearchResultLocation = "web_search_result_location"
)

// Citation is one source reference on a text part. Type selects which
// location fields apply: URL/Title/EncryptedIndex for web search results,
// DocumentIndex/DocumentTitle plus a char, page or block range for documents,
// Source/Title/SearchResultIndex plus a block range for search results.
//
// Raw is the citation exactly as the provider sent it. Adapters replay Raw
// when it is set (web search's encrypted_index must come back unchanged), so
// the typed fields are for reading; an unknown citation type survives in Raw.
type Citation struct {
	Type              string          `json:"type"`
	CitedText         string          `json:"cited_text,omitempty"`
	URL               string          `json:"url,omitempty"`
	Title             string          `json:"title,omitempty"`
	EncryptedIndex    string          `json:"encrypted_index,omitempty"`
	DocumentIndex     int             `json:"document_index,omitempty"`
	DocumentTitle     string          `json:"document_title,omitempty"`
	StartCharIndex    int             `json:"start_char_index,omitempty"`
	EndCharIndex      int             `json:"end_char_index,omitempty"`
	StartPageNumber   int             `json:"start_page_number,omitempty"`
	EndPageNumber     int             `json:"end_page_number,omitempty"`
	StartBlockIndex   int             `json:"start_block_index,omitempty"`
	EndBlockIndex     int             `json:"end_block_index,omitempty"`
	Source            string          `json:"source,omitempty"`
	SearchResultIndex int             `json:"search_result_index,omitempty"`
	Raw               json.RawMessage `json:"raw,omitempty"`
}

// CitationFromRaw decodes a provider citation object, keeping it in Raw.
// Unknown fields survive in Raw only.
func CitationFromRaw(raw json.RawMessage) (Citation, error) {
	var c Citation
	if err := json.Unmarshal(raw, &c); err != nil {
		return Citation{}, err
	}
	c.Raw = compactRaw(raw)
	return c, nil
}

// compactRaw returns raw in compact form, the form encoding/json re-emits, so
// a stored block marshals to the same bytes on every turn.
func compactRaw(raw json.RawMessage) json.RawMessage {
	var buf bytes.Buffer
	if err := json.Compact(&buf, raw); err != nil {
		return append(json.RawMessage(nil), raw...)
	}
	return json.RawMessage(buf.Bytes())
}

// =============================================================================
// SERVER TOOL USAGE
// =============================================================================

// ServerToolUsage counts the server tool calls a response was billed for.
type ServerToolUsage struct {
	WebSearchRequests     int `json:"web_search_requests,omitempty"`
	WebFetchRequests      int `json:"web_fetch_requests,omitempty"`
	CodeExecutionRequests int `json:"code_execution_requests,omitempty"`
}

// UsageIteration is one entry of Anthropic's usage.iterations: one sampling
// attempt inside a single request, such as a compaction step ("compaction"),
// a declined attempt ("message") or the fallback model that answered
// ("fallback_message"). When a response reports iterations they are the
// billing record: the top-level token counts cover only the attempt that
// produced the returned message. Model is empty when the provider did not name
// one (the response's model ran it). Token counts follow the disjoint
// cache-token contract of Response.InputTokens.
type UsageIteration struct {
	Type                string `json:"type"`
	Model               string `json:"model,omitempty"`
	InputTokens         int    `json:"input_tokens"`
	OutputTokens        int    `json:"output_tokens"`
	CacheCreationTokens int    `json:"cache_creation_input_tokens,omitempty"`
	CacheReadTokens     int    `json:"cache_read_input_tokens,omitempty"`
	// CacheCreation1hTokens is the 1-hour share of CacheCreationTokens.
	CacheCreation1hTokens int `json:"cache_creation_1h_input_tokens,omitempty"`
}

// UnsupportedServerTools returns an error when req sets ServerTools or a
// ContainerID (the container those tools run in). Adapters
// without server tools call it so a request never silently loses a tool the
// caller depends on (and pays for).
func UnsupportedServerTools(provider string, req Request) error {
	if len(req.ServerTools) > 0 {
		return fmt.Errorf("llm: provider %q does not support server tools (%s)", provider, req.ServerTools[0].Name)
	}
	if req.ContainerID != "" {
		return fmt.Errorf("llm: provider %q does not support ContainerID (an Anthropic server-tool container)", provider)
	}
	return nil
}

// Prompt-cache lifetimes for the CacheTTL fields (Anthropic). The 1-hour
// cache costs more to write (2x input instead of 1.25x); the API requires
// 1-hour entries to come before 5-minute ones in a request.
const (
	CacheTTL5m = "5m"
	CacheTTL1h = "1h"
)

// StopDetails is Anthropic's stop_details, set when a request was declined
// (StopReason StopRefusal). Category names the policy area ("cyber", "bio",
// "reasoning_extraction", …; empty when the API sends null) and is
// informational: branch on StopReason, not on StopDetails. RecommendedModel
// is a model to retry on when the fallback could not run. The fallback-credit
// fields are set with the fallback-credit beta. Raw is the object as received,
// for fields not modeled here.
type StopDetails struct {
	Type                    string          `json:"type,omitempty"`
	Category                string          `json:"category,omitempty"`
	Explanation             string          `json:"explanation,omitempty"`
	RecommendedModel        string          `json:"recommended_model,omitempty"`
	FallbackCreditToken     string          `json:"fallback_credit_token,omitempty"`
	FallbackHasPrefillClaim bool            `json:"fallback_has_prefill_claim,omitempty"`
	Raw                     json.RawMessage `json:"raw,omitempty"`
}

// StopDetailsFromRaw decodes a stop_details object. It returns nil for an
// absent or null value, and for anything that is not a JSON object. Fields
// whose types do not match are left empty; Raw keeps them.
func StopDetailsFromRaw(raw json.RawMessage) *StopDetails {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || trimmed[0] != '{' || !json.Valid(trimmed) {
		return nil
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(trimmed, &fields); err != nil {
		return nil
	}
	d := &StopDetails{Raw: compactRaw(trimmed)}
	for key, dst := range map[string]*string{
		"type": &d.Type, "category": &d.Category, "explanation": &d.Explanation,
		"recommended_model": &d.RecommendedModel, "fallback_credit_token": &d.FallbackCreditToken,
	} {
		_ = json.Unmarshal(fields[key], dst)
	}
	_ = json.Unmarshal(fields["fallback_has_prefill_claim"], &d.FallbackHasPrefillClaim)
	return d
}
