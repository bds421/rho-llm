// Package provider registers all built-in LLM provider adapters.
// Import this package (typically as a blank import) to make all providers
// available to llm.NewClient:
//
//	import _ "gitlab2024.bds421-cloud.com/bds421/rho/llm/provider"
package provider

import (
	_ "gitlab2024.bds421-cloud.com/bds421/rho/llm/provider/anthropic"
	_ "gitlab2024.bds421-cloud.com/bds421/rho/llm/provider/gemini"
	_ "gitlab2024.bds421-cloud.com/bds421/rho/llm/provider/openaicompat"
)
