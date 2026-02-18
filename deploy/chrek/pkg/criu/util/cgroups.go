package util

import (
	"fmt"
	"strings"

	criurpc "github.com/checkpoint-restore/go-criu/v8/rpc"
)

// ParseManageCgroupsMode normalizes and validates the CRIU cgroup mode setting.
func ParseManageCgroupsMode(raw string) (criurpc.CriuCgMode, string, error) {
	mode := strings.ToLower(strings.TrimSpace(raw))
	switch mode {
	case "ignore":
		return criurpc.CriuCgMode_IGNORE, "ignore", nil
	case "soft":
		return criurpc.CriuCgMode_SOFT, mode, nil
	case "full":
		return criurpc.CriuCgMode_FULL, mode, nil
	case "strict":
		return criurpc.CriuCgMode_STRICT, mode, nil
	default:
		return criurpc.CriuCgMode_IGNORE, "", fmt.Errorf("invalid manageCgroupsMode %q", raw)
	}
}
