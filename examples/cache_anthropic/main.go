// Example: Anthropic Prompt Caching
//
// Demonstrates using Anthropic's cache_control to cache large system prompts,
// content blocks, and tool definitions. Unlike Gemini, Anthropic caching is
// inline — no external cache creation step needed. Mark content as cacheable
// and Anthropic handles it automatically.
//
// Usage:
//
//	go run examples/cache_anthropic/main.go
package main

import (
	"context"
	_ "embed"
	"fmt"
	"os"
	"time"

	"github.com/joho/godotenv"
	"github.com/bds421/rho-llm"
	_ "github.com/bds421/rho-llm/provider" // register all provider adapters
)

//go:embed system_prompt.txt
var systemPrompt string

func main() {
	_ = godotenv.Load(".env")

	ctx := context.Background()

	client, err := llm.NewClient(llm.Config{
		Provider:  "anthropic",
		Model:     "claude-sonnet-4-20250514",
		APIKey:    os.Getenv("ANTHROPIC_API_KEY"),
		MaxTokens: 4096,
		Timeout:   60 * time.Second,
	})
	if err != nil {
		panic(err)
	}
	defer client.Close()

	// --- 1. System prompt caching ---
	//
	// Setting SystemCacheControl: true caches the entire system prompt.
	// On the first request, Anthropic writes it to cache (you pay cache_creation tokens).
	// On subsequent requests within the TTL (~5 min), the cached version is used
	// and you pay the cheaper cache_read rate (~90% discount).
	//
	// Best for: large, stable system prompts reused across many requests.

	fmt.Printf("System prompt size: ~%d characters\n", len(systemPrompt))

	fmt.Println("=== System Prompt Caching ===")
	fmt.Println("Sending first request (cache write)...")

	req := llm.Request{
		System: systemPrompt,
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "Review this Go snippet:\nfunc add(a, b int) int { return a + b }"),
		},
		SystemCacheControl: true, // cache the system prompt
	}

	resp, err := client.Complete(ctx, req)
	if err != nil {
		panic(err)
	}
	printResponse("First call", resp)

	// Second call with the same system prompt — should hit cache
	fmt.Println("\nSending second request (cache read)...")
	req.Messages = []llm.Message{
		llm.NewTextMessage(llm.RoleUser, "Review this Go snippet:\nfunc div(a, b int) int { return a / b }"),
	}

	resp, err = client.Complete(ctx, req)
	if err != nil {
		panic(err)
	}
	printResponse("Second call", resp)

	// --- 2. Content block caching ---
	//
	// Mark individual content blocks as cacheable with CacheControl: true.
	// Useful for caching large user-provided context (documents, code files)
	// while keeping the follow-up questions dynamic.

	fmt.Println("\n=== Content Block Caching ===")

	// In production, this would be a large document or codebase.
	// Here we reuse the system prompt as example content to cache.
	largeContext := systemPrompt

	req = llm.Request{
		Messages: []llm.Message{
			{
				Role: llm.RoleUser,
				Content: []llm.ContentPart{
					{
						Type:         llm.ContentText,
						Text:         largeContext,
						CacheControl: true, // cache this large content block
					},
					{
						Type: llm.ContentText,
						Text: "What patterns do you see in this code?",
					},
				},
			},
		},
	}

	resp, err = client.Complete(ctx, req)
	if err != nil {
		panic(err)
	}
	printResponse("Content block cache", resp)

	// --- 3. Tool definition caching ---
	//
	// Mark tool definitions as cacheable with CacheControl: true.
	// Useful when you have many tools with large schemas that don't change
	// between requests.

	fmt.Println("\n=== Tool Definition Caching ===")

	req = llm.Request{
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "What's the weather in Berlin and the current stock price of AAPL?"),
		},
		Tools: []llm.Tool{
			{
				Name:        "get_weather",
				Description: "Get current weather for a city",
				InputSchema: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"city": map[string]interface{}{
							"type":        "string",
							"description": "City name",
						},
					},
					"required": []string{"city"},
				},
				CacheControl: true, // cache the tool definitions
			},
			{
				Name:        "get_stock_price",
				Description: "Get current stock price for a ticker symbol",
				InputSchema: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"ticker": map[string]interface{}{
							"type":        "string",
							"description": "Stock ticker symbol, e.g. AAPL",
						},
					},
					"required": []string{"ticker"},
				},
			},
		},
	}

	resp, err = client.Complete(ctx, req)
	if err != nil {
		panic(err)
	}
	printResponse("Tool cache", resp)
}

func printResponse(label string, resp *llm.Response) {
	fmt.Printf("\n[%s]\n", label)
	fmt.Printf("  Response: %.80s...\n", resp.Content)
	fmt.Printf("  Input tokens:           %d\n", resp.InputTokens)
	fmt.Printf("  Output tokens:          %d\n", resp.OutputTokens)
	fmt.Printf("  Cache creation tokens:  %d\n", resp.CacheCreationTokens)
	fmt.Printf("  Cache read tokens:      %d\n", resp.CacheReadTokens)
}
