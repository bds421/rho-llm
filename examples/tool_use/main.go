package main

import (
	"context"
	"fmt"
	"os"
	"time"

	"gitlab2024.bds421-cloud.com/bds421/rho/llm"
	_ "gitlab2024.bds421-cloud.com/bds421/rho/llm/provider" // register all provider adapters
)

// A mocked local function we want the LLM to call
func executeMyTool(toolName string, input any) string {
	if toolName == "get_weather" {
		// In a real app, parse input map[string]interface{} and call weather API
		return "It is 22°C and sunny in " + fmt.Sprintf("%v", input)
	}
	return "unknown tool"
}

func main() {
	ctx := context.Background()

	cfg := llm.Config{
		Provider: "gemini",
		Model:    "flash",
		APIKey:   os.Getenv("GEMINI_API_KEY"),
		Timeout:  30 * time.Second,
	}

	client, err := llm.NewClient(cfg)
	if err != nil {
		panic(err)
	}
	defer client.Close()

	// 1. Define tools and send the initial request
	req := llm.Request{
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "What's the weather in Berlin?"),
		},
		Tools: []llm.Tool{{
			Name:        "get_weather",
			Description: "Get the current weather for a city",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"location": map[string]interface{}{
						"type":        "string",
						"description": "City name, e.g. 'Berlin'",
					},
				},
				"required": []string{"location"},
			},
		}},
	}

	fmt.Println("Sending request...")
	resp, err := client.Complete(ctx, req)
	if err != nil {
		panic(err)
	}

	// 2. The Agentic Loop: Handle tool calls sequentially until the model stops returning "tool_use"
	for resp.StopReason == "tool_use" {
		fmt.Printf("Model wants to use tools: %d calls requested\n", len(resp.ToolCalls))

		var results []llm.Message
		for _, tc := range resp.ToolCalls {
			fmt.Printf("-> Executing %s with args %v...\n", tc.Name, tc.Input)
			output := executeMyTool(tc.Name, tc.Input)

			// Wrap the result — the tool use ID links it back to the original request
			results = append(results, llm.NewToolResultMessage(tc.ID, output, false))
		}

		// Append the full assistant response (text + tool_use blocks)
		req.Messages = append(req.Messages, llm.NewAssistantMessage(resp))

		// Append the tool results we just generated
		req.Messages = append(req.Messages, results...)

		fmt.Println("Sending tool results back to the model...")
		resp, err = client.Complete(ctx, req)
		if err != nil {
			panic(err)
		}
	}

	// 3. Final text output
	fmt.Printf("\nFinal Response:\n%s\n", resp.Content)
}
