package util

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/go-logr/logr"
)

func LogRestoreErrors(checkpointPath, workDir string, log logr.Logger) {
	candidates := make([]string, 0, 2)
	if workDir != "" {
		candidates = append(candidates, filepath.Join(workDir, "restore.log"))
	}
	candidates = append(candidates, filepath.Join(checkpointPath, "restore.log"))

	for _, logPath := range candidates {
		if _, err := os.Stat(logPath); err != nil {
			continue
		}
		log.Info("CRIU restore log found", "path", logPath)
		LogRestoreSummary(logPath, log)
		return
	}
}

func LogRestoreSummary(path string, log logr.Logger) {
	data, err := os.ReadFile(path)
	if err != nil {
		log.Error(err, "Failed to read CRIU restore log", "path", path)
		return
	}

	lines := strings.Split(string(data), "\n")
	keyLines := make([]string, 0, 64)
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}
		lower := strings.ToLower(trimmed)
		if strings.Contains(lower, "error") ||
			strings.Contains(lower, "warn") ||
			strings.Contains(lower, "fail") ||
			strings.Contains(lower, "cuda") ||
			strings.Contains(lower, "iptables") ||
			strings.Contains(lower, "restore finished successfully") ||
			strings.Contains(lower, "tasks resumed") {
			keyLines = append(keyLines, trimmed)
			if len(keyLines) == 80 {
				break
			}
		}
	}
	if len(keyLines) > 0 {
		log.Info("CRIU restore key lines", "path", path, "lines", strings.Join(keyLines, "\n"))
	}

	tail := make([]string, 0, 40)
	for i := len(lines) - 1; i >= 0 && len(tail) < 40; i-- {
		trimmed := strings.TrimSpace(lines[i])
		if trimmed == "" {
			continue
		}
		tail = append(tail, trimmed)
	}
	for i, j := 0, len(tail)-1; i < j; i, j = i+1, j-1 {
		tail[i], tail[j] = tail[j], tail[i]
	}
	if len(tail) > 0 {
		log.Info("CRIU restore tail", "path", path, "lines", strings.Join(tail, "\n"))
	}
}

func LogCgroupRestoreErrors(checkpointPath, workDir string, log logr.Logger) {
	candidates := make([]string, 0, 2)
	if workDir != "" {
		candidates = append(candidates, filepath.Join(workDir, "restore.log"))
	}
	candidates = append(candidates, filepath.Join(checkpointPath, "restore.log"))

	for _, logPath := range candidates {
		data, err := os.ReadFile(logPath)
		if err != nil {
			continue
		}

		lines := strings.Split(string(data), "\n")
		cgroupErrors := make([]string, 0, 20)
		for _, line := range lines {
			trimmed := strings.TrimSpace(line)
			if trimmed == "" {
				continue
			}
			lower := strings.ToLower(trimmed)
			if !strings.Contains(lower, "cgroup") {
				continue
			}
			if strings.Contains(lower, "error") || strings.Contains(lower, "failed") || strings.Contains(lower, "fail") {
				cgroupErrors = append(cgroupErrors, trimmed)
			}
			if len(cgroupErrors) >= 20 {
				break
			}
		}
		if len(cgroupErrors) > 0 {
			log.Error(
				fmt.Errorf("detected %d cgroup restore errors", len(cgroupErrors)),
				"CRIU cgroup restore errors",
				"path", logPath,
				"lines", strings.Join(cgroupErrors, "\n"),
			)
		}
		return
	}
}
