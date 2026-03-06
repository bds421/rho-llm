// Example: Gemini Context Caching (end-to-end)
//
// Demonstrates the full lifecycle of Gemini's explicit context caching:
//  1. Create a cache with a large system instruction via the REST API
//  2. Use the cached content in multiple LLM requests
//  3. Clean up (delete) the cache
//
// Unlike Anthropic's inline caching, Gemini requires an external cache
// creation step. This example handles that via direct REST calls, then
// passes the cache name into the rho/llm SDK via Request.CachedContent.
//
// Minimum cacheable size: 32,768 tokens (Flash) or 4,096 tokens (Pro).
//
// Usage:
//
//	go run examples/cache_gemini/main.go
package main

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/joho/godotenv"
	"gitlab2024.bds421-cloud.com/bds421/rho/llm"
	_ "gitlab2024.bds421-cloud.com/bds421/rho/llm/provider" // register all provider adapters
)

const (
	cacheAPIBase = "https://generativelanguage.googleapis.com/v1beta/cachedContents"
	modelName    = "gemini-2.5-pro"
)

//go:embed system_prompt.txt
var systemPrompt string

func main() {
	_ = godotenv.Load(".env")

	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		fmt.Println("GEMINI_API_KEY not set in .env")
		os.Exit(1)
	}

	ctx := context.Background()

	// =========================================================================
	// Step 1: Load the system instruction to cache
	// =========================================================================
	//
	// The system prompt is embedded from system_prompt.txt at compile time.
	// It contains a comprehensive code review prompt with a Go review
	// knowledge base — the kind of large, reusable context that benefits
	// from caching. Gemini Pro requires at least 4,096 tokens (~16K chars).

	fmt.Printf("System instruction size: ~%d characters\n", len(systemPrompt))

	// =========================================================================
	// Step 2: Create the cache via the Gemini REST API
	// =========================================================================

	fmt.Println("\nCreating cache...")
	cacheName, err := createCache(apiKey, systemPrompt)
	if err != nil {
		fmt.Printf("Failed to create cache: %v\n", err)
		fmt.Println("\nNote: The content may be too small to cache. Gemini requires")
		fmt.Println("at least 32,768 tokens for Flash or 4,096 tokens for Pro models.")
		os.Exit(1)
	}
	fmt.Printf("Cache created: %s\n", cacheName)

	// Ensure cleanup on exit
	defer func() {
		fmt.Println("\nDeleting cache...")
		if err := deleteCache(apiKey, cacheName); err != nil {
			fmt.Printf("Warning: failed to delete cache: %v\n", err)
		} else {
			fmt.Println("Cache deleted.")
		}
	}()

	// =========================================================================
	// Step 3: Use the cache with the rho/llm SDK
	// =========================================================================

	client, err := llm.NewClient(llm.Config{
		Provider: "gemini",
		Model:    modelName,
		APIKey:   apiKey,
		Timeout:  30 * time.Second,
	})
	if err != nil {
		panic(err)
	}
	defer client.Close()

	// First request — references the cached system instruction
	fmt.Println("\n=== Request 1 (using cache) ===")
	resp, err := client.Complete(ctx, llm.Request{
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "Review this file:\n\n// File: math.go\npackage mathutil\n\nfunc div(a, b int) int { return a / b }\n"),
		},
		CachedContent: cacheName, // reference the pre-created cache
	})
	if err != nil {
		fmt.Printf("Request 1 failed: %v\n", err)
	} else {
		printResponse("Request 1", resp)
	}

	// Second request — same cache, different question
	fmt.Println("\n=== Request 2 (same cache, different question) ===")
	resp, err = client.Complete(ctx, llm.Request{
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "Review this file:\n\n// File: fileutil.go\npackage fileutil\n\nimport \"os\"\n\nfunc readFile(path string) string {\n\tdata, _ := os.ReadFile(path)\n\treturn string(data)\n}\n"),
		},
		CachedContent: cacheName, // same cache reused
	})
	if err != nil {
		fmt.Printf("Request 2 failed: %v\n", err)
	} else {
		printResponse("Request 2", resp)
	}
}

func printResponse(label string, resp *llm.Response) {
	fmt.Printf("\n[%s]\n", label)
	fmt.Printf("  Response: %.100s...\n", resp.Content)
	fmt.Printf("  Input tokens:      %d\n", resp.InputTokens)
	fmt.Printf("  Output tokens:     %d\n", resp.OutputTokens)
	fmt.Printf("  Cache read tokens: %d\n", resp.CacheReadTokens)
}

// =============================================================================
// Gemini Caching REST API helpers
// =============================================================================

// createCache creates a CachedContent resource with the given system instruction.
// Returns the cache resource name (e.g. "cachedContents/abc123").
func createCache(apiKey, systemInstruction string) (string, error) {
	payload := map[string]interface{}{
		"model": "models/" + modelName,
		"systemInstruction": map[string]interface{}{
			"parts": []map[string]string{
				{"text": systemInstruction},
			},
		},
		"ttl": "300s", // 5 minutes — enough for this demo
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("marshal: %w", err)
	}

	url := fmt.Sprintf("%s?key=%s", cacheAPIBase, apiKey)
	resp, err := http.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("POST: %w", err)
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("status %d: %s", resp.StatusCode, string(respBody))
	}

	var result struct {
		Name string `json:"name"`
	}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return "", fmt.Errorf("decode: %w", err)
	}

	return result.Name, nil
}

// deleteCache deletes a CachedContent resource.
func deleteCache(apiKey, cacheName string) error {
	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/%s?key=%s", cacheName, apiKey)
	req, err := http.NewRequest("DELETE", url, nil)
	if err != nil {
		return err
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("status %d: %s", resp.StatusCode, string(body))
	}
	return nil
}
