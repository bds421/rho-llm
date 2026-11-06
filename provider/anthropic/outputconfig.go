package anthropic

import (
	"fmt"
	"slices"

	llm "github.com/bds421/rho-llm"
)

// outputFormat maps Request.ResponseFormat onto output_config.format.
// Anthropic constrains output only to a JSON schema; plain JSON mode has no
// equivalent, so it is refused rather than sent as free text.
func outputFormat(rf *llm.ResponseFormat) (*anthropicOutputFormat, error) {
	if rf == nil || rf.Type == "" {
		return nil, nil
	}
	switch rf.Type {
	case llm.ResponseFormatJSONSchema:
		if len(rf.Schema) == 0 {
			return nil, fmt.Errorf("anthropic: ResponseFormat json_schema needs a Schema (output_config.format has no schema-less JSON mode)")
		}
		return &anthropicOutputFormat{Type: llm.ResponseFormatJSONSchema, Schema: rf.Schema}, nil
	case llm.ResponseFormatJSONObject:
		return nil, fmt.Errorf("anthropic: ResponseFormat json_object is not supported; output_config.format takes a json_schema, so pass a Schema")
	}
	return nil, fmt.Errorf("anthropic: unsupported ResponseFormat type %q", rf.Type)
}

// thinkingDisplay maps ReasoningSummary onto adaptive thinking's display.
// Unset keeps "summarized" (readable thinking text, the adapter's long-standing
// behaviour); ReasoningSummaryOmitted sends "omitted"; ReasoningSummaryAuto
// leaves display off so the API default applies ("omitted" on Claude 4.7 and later, "summarized" on 4.6). Display
// changes only what is returned: thinking is billed the same either way.
func thinkingDisplay(summary llm.ReasoningSummary) string {
	switch summary {
	case llm.ReasoningSummaryAuto:
		return ""
	case llm.ReasoningSummaryOmitted:
		return "omitted"
	}
	return "summarized"
}

// budgetEffort lists the effort levels the budget-thinking models accept;
// Sonnet 4.5 and Haiku 4.5 reject effort altogether. Other models not in
// adaptiveThinkingModels are unreviewed and pass through.
var budgetEffort = map[string][]llm.Effort{
	"claude-opus-4-6":            {llm.EffortLow, llm.EffortMedium, llm.EffortHigh, llm.EffortMax},
	"claude-sonnet-4-6":          {llm.EffortLow, llm.EffortMedium, llm.EffortHigh, llm.EffortMax},
	"claude-opus-4-5":            {llm.EffortLow, llm.EffortMedium, llm.EffortHigh},
	"claude-opus-4-5-20251101":   {llm.EffortLow, llm.EffortMedium, llm.EffortHigh},
	"claude-sonnet-4-5":          nil,
	"claude-sonnet-4-5-20250929": nil,
	"claude-haiku-4-5":           nil,
	"claude-haiku-4-5-20251001":  nil,
}

// requestEffort returns output_config.effort for req: the effort a
// ThinkingLevel sets on an adaptive model, or Request.Effort. Both set must
// agree.
func requestEffort(req llm.Request, model string, adaptive bool) (string, error) {
	if !req.Effort.Valid() {
		return "", fmt.Errorf("anthropic: unknown Effort %q", req.Effort)
	}
	level := ""
	if adaptive && req.ThinkingLevel != llm.ThinkingNone {
		level = string(req.ThinkingLevel)
		if req.ThinkingLevel == llm.ThinkingMinimal {
			level = string(llm.EffortLow)
		}
	}
	if req.Effort == "" {
		return level, nil
	}
	if level != "" && level != string(req.Effort) {
		return "", fmt.Errorf("anthropic: ThinkingLevel %q sets effort %q, which conflicts with Effort %q; set one", req.ThinkingLevel, level, req.Effort)
	}
	if allowed, known := budgetEffort[model]; known && !slices.Contains(allowed, req.Effort) {
		return "", fmt.Errorf("anthropic: %s does not accept effort %q", model, req.Effort)
	}
	return string(req.Effort), nil
}

func aboveHigh(effort string) bool {
	return effort == string(llm.EffortXHigh) || effort == string(llm.EffortMax)
}
