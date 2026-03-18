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
		Provider: "openai",
		Model:    "gpt-4o",
		Timeout:  30 * time.Second,
	}

	// 1. Configure the failover pool
	// We can provide multiple keys. Some keys might even point to entirely different proxies.
	// The pool will gracefully fall back if one is rate-limited (429) or dead (5xx).
	keys := []string{
		os.Getenv("OPENAI_PRIMARY_KEY"),
		os.Getenv("OPENAI_BACKUP_KEY"),
		fmt.Sprintf("%s|https://my-azure-proxy.internal/v1", os.Getenv("AZURE_OPENAI_KEY")), // Custom base URL!
	}

	// 2. Initialize the pooled client
	client, err := llm.NewClientWithKeys(cfg, keys)
	if err != nil {
		panic(err)
	}
	defer client.Close()

	// 3. Send a request
	req := llm.Request{
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "What is the capital of France?"),
		},
	}

	fmt.Println("Sending request... (if the primary key fails, it will seamlessly failover to the backups)")
	resp, err := client.Complete(ctx, req)
	if err != nil {
		fmt.Printf("All profiles failed. Error: %v\n", err)
		return
	}

	fmt.Printf("\nSuccess!\n%s\n", resp.Content)
}
