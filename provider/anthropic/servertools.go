package anthropic

import (
	"fmt"

	"github.com/bds421/rho-llm"
)

// serverToolKeys are the wire fields ServerTool sets itself; Params may not
// repeat them.
var serverToolKeys = map[string]bool{
	"type": true, "name": true, "max_uses": true, "allowed_domains": true,
	"blocked_domains": true, "user_location": true, "allowed_callers": true,
	"cache_control": true,
}

// serverToolWire encodes a ServerTool as an Anthropic tool object. The object
// is a map, which encoding/json writes with sorted keys, so the same
// ServerTool always encodes to the same bytes (prompt-cache stable).
func serverToolWire(t llm.ServerTool) (map[string]any, error) {
	if t.Type == "" || t.Name == "" {
		return nil, fmt.Errorf("anthropic: server tool needs Type and Name (got type %q, name %q)", t.Type, t.Name)
	}
	if t.MaxUses < 0 {
		return nil, fmt.Errorf("anthropic: server tool %q: MaxUses must not be negative", t.Name)
	}
	if len(t.AllowedDomains) > 0 && len(t.BlockedDomains) > 0 {
		return nil, fmt.Errorf("anthropic: server tool %q: set AllowedDomains or BlockedDomains, not both", t.Name)
	}
	tool := map[string]any{"type": t.Type, "name": t.Name}
	if t.MaxUses > 0 {
		tool["max_uses"] = t.MaxUses
	}
	if err := setDomains(tool, "allowed_domains", t.Name, t.AllowedDomains); err != nil {
		return nil, err
	}
	if err := setDomains(tool, "blocked_domains", t.Name, t.BlockedDomains); err != nil {
		return nil, err
	}
	if loc := t.UserLocation; loc != nil {
		if loc.City == "" && loc.Region == "" && loc.Country == "" && loc.Timezone == "" {
			return nil, fmt.Errorf("anthropic: server tool %q: UserLocation needs at least one of city, region, country, timezone", t.Name)
		}
		wire := map[string]any{"type": "approximate"}
		if loc.Type != "" {
			wire["type"] = loc.Type
		}
		for key, value := range map[string]string{"city": loc.City, "region": loc.Region, "country": loc.Country, "timezone": loc.Timezone} {
			if value != "" {
				wire[key] = value
			}
		}
		tool["user_location"] = wire
	}
	if len(t.AllowedCallers) > 0 {
		tool["allowed_callers"] = append([]string(nil), t.AllowedCallers...)
	}
	if t.CacheControl {
		cc, err := cacheControl(t.CacheTTL)
		if err != nil {
			return nil, err
		}
		tool["cache_control"] = cc
	}
	for key, value := range t.Params {
		if serverToolKeys[key] {
			return nil, fmt.Errorf("anthropic: server tool %q: Params must not set %q (use the typed field)", t.Name, key)
		}
		tool[key] = value
	}
	return tool, nil
}

func setDomains(tool map[string]any, key, name string, domains []string) error {
	if len(domains) == 0 {
		return nil
	}
	for _, d := range domains {
		if d == "" {
			return fmt.Errorf("anthropic: server tool %q: empty entry in %s", name, key)
		}
	}
	tool[key] = append([]string(nil), domains...)
	return nil
}
