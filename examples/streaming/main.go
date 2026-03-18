package main

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/bds421/rho-llm"
	_ "github.com/bds421/rho-llm/provider" // register all provider adapters
)

func main() {
	ctx := context.Background()

	cfg := llm.Config{
		Provider: "anthropic",
		Model:    "claude-sonnet-4-6",
		APIKey:   os.Getenv("ANTHROPIC_API_KEY"),
		Timeout:  30 * time.Second,
	}

	client, err := llm.NewClient(cfg)
	if err != nil {
		panic(err)
	}
	defer client.Close()

	req := llm.Request{
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "Write a short haiku about streaming data."),
		},
	}

	fmt.Println("Streaming response:")
	fmt.Println("-------------------")

	// The stream returns a Go 1.23 standard iterator
	for event, err := range client.Stream(ctx, req) {
		if err != nil {
			fmt.Printf("\nStream error: %v\n", err)
			break
		}

		switch event.Type {
		case llm.EventContent:
			// Print each chunk of text as it arrives
			fmt.Print(event.Text)
		case llm.EventDone:
			// The API has finished and returned usage metadata
			fmt.Printf("\n\n(Done: %s, Input: %d, Output: %d)\n",
				event.StopReason, event.InputTokens, event.OutputTokens)
		}
	}
}
