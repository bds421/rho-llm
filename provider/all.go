// Package provider registers all built-in LLM provider adapters.
// Import this package (typically as a blank import) to make all providers
// available to llm.NewClient:
//
//	import _ "github.com/bds421/rho-llm/provider"
package provider

import (
	_ "github.com/bds421/rho-llm/provider/anthropic"
	_ "github.com/bds421/rho-llm/provider/gemini"
	_ "github.com/bds421/rho-llm/provider/openaicompat"
	_ "github.com/bds421/rho-llm/provider/openairesponses"
)
