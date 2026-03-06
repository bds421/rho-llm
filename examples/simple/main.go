package main

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/joho/godotenv"
	"gitlab2024.bds421-cloud.com/bds421/rho/llm"
	_ "gitlab2024.bds421-cloud.com/bds421/rho/llm/provider" // register all provider adapters
)

func main() {
	// Load .env from the working directory (run as: go run examples/simple/main.go from llm/)
	_ = godotenv.Load(".env")

	ctx := context.Background()

	// 1. Configure the LLM client
	cfg := llm.Config{
		Provider: "gemini",
		Model:    "gemini-3.1-flash-lite-preview",
		APIKey:   os.Getenv("GEMINI_API_KEY"),
		Timeout:  30 * time.Second,
	}

	// 2. Initialize the client
	client, err := llm.NewClient(cfg)
	if err != nil {
		panic(err)
	}
	defer client.Close()

	// 3. Create a simple text request
	req := llm.Request{
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "Tell me a short joke about a programmer."),
		},
	}

	fmt.Println("Sending request to LLM...")

	// 4. Send the request and wait for the response
	resp, err := client.Complete(ctx, req)
	if err != nil {
		panic(err)
	}

	// 5. Print the response content
	fmt.Printf("\nResponse:\n%s\n", resp.Content)
}
