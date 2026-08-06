package openaibatch

import (
	"net/http"
	"testing"

	llm "github.com/bds421/rho-llm"
)

func TestNewAppliesPerDeploymentProxyPolicy(t *testing.T) {
	t.Run("reviewed public proxy", func(t *testing.T) {
		client, err := New(llm.Config{
			Provider: "openai", APIKey: "test", BaseURL: "https://api.openai.com/v1",
			ProxyURL: "http://reviewed-proxy.example:3128",
		})
		if err != nil {
			t.Fatal(err)
		}
		defer client.Close()
		transport := client.httpClient.Transport.(*http.Transport)
		req, err := http.NewRequest(http.MethodGet, "https://api.openai.com/v1/files", nil)
		if err != nil {
			t.Fatal(err)
		}
		proxy, err := transport.Proxy(req)
		if err != nil {
			t.Fatal(err)
		}
		if proxy == nil || proxy.String() != "http://reviewed-proxy.example:3128" {
			t.Fatalf("proxy = %v", proxy)
		}
	})

	t.Run("private endpoint bypasses ambient proxy", func(t *testing.T) {
		t.Setenv("HTTPS_PROXY", "http://ambient.invalid:8080")
		client, err := New(llm.Config{
			Provider: "openai", APIKey: "test", BaseURL: "https://private-model.internal/v1",
			DisableProxy: true,
		})
		if err != nil {
			t.Fatal(err)
		}
		defer client.Close()
		transport := client.httpClient.Transport.(*http.Transport)
		if transport.Proxy != nil {
			t.Fatal("private batch client still consults an ambient proxy policy")
		}
	})
}
