package nsrunner

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/go-logr/logr"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/criu"
)

func ValidateRestoredProcess(procRoot string, pid int, restoreLogPath string, log logr.Logger) error {
	if err := validateProcessState(procRoot, pid); err != nil {
		log.Error(err, "Restored process failed immediate post-restore validation",
			"restored_pid", pid,
			"proc_root", procRoot,
		)
		logProcessDiagnostics(procRoot, pid, restoreLogPath, log)
		return fmt.Errorf("restored process failed post-restore validation: %w", err)
	}
	return nil
}

func validateProcessState(procRoot string, pid int) error {
	if pid <= 0 {
		return fmt.Errorf("invalid restored PID %d", pid)
	}

	statusPath := filepath.Join(procRoot, strconv.Itoa(pid), "status")
	data, err := os.ReadFile(statusPath)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("process %d exited", pid)
		}
		return fmt.Errorf("failed to inspect process %d: %w", pid, err)
	}

	for _, line := range strings.Split(string(data), "\n") {
		if !strings.HasPrefix(line, "State:") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			return fmt.Errorf("state not found in %s", statusPath)
		}
		if fields[1] == "Z" {
			return fmt.Errorf("process %d became zombie", pid)
		}
		return nil
	}

	return fmt.Errorf("state not found in %s", statusPath)
}

func logProcessDiagnostics(procRoot string, pid int, restoreLogPath string, log logr.Logger) {
	entry := log.WithValues("restored_pid", pid, "proc_root", procRoot)

	statusPath := filepath.Join(procRoot, strconv.Itoa(pid), "status")
	if data, err := os.ReadFile(statusPath); err == nil {
		entry.Error(fmt.Errorf("%s", strings.TrimSpace(string(data))), "Restored process status", "path", statusPath)
	} else {
		entry.Error(err, "Failed to read restored process status", "path", statusPath)
	}

	cmdlinePath := filepath.Join(procRoot, strconv.Itoa(pid), "cmdline")
	if data, err := os.ReadFile(cmdlinePath); err == nil {
		cmdline := strings.TrimSpace(strings.ReplaceAll(string(data), "\x00", " "))
		if cmdline == "" {
			cmdline = "<empty>"
		}
		entry.Info("Restored process cmdline", "cmdline", cmdline)
	} else {
		entry.Error(err, "Failed to read restored process cmdline", "path", cmdlinePath)
	}

	statPath := filepath.Join(procRoot, strconv.Itoa(pid), "stat")
	if data, err := os.ReadFile(statPath); err == nil {
		raw, parseErr := parseProcExitCodeRaw(string(data))
		if parseErr != nil {
			entry.Error(parseErr, "Failed to parse /proc stat exit code", "path", statPath)
		} else {
			exitStatus, termSignal, coreDumped := decodeProcExitCode(raw)
			entry.Info("Decoded restored process exit code",
				"exit_code_raw", raw,
				"exit_status", exitStatus,
				"term_signal", termSignal,
				"core_dumped", coreDumped,
			)
		}
	}

	childrenPath := filepath.Join(procRoot, "1", "task", "1", "children")
	if data, err := os.ReadFile(childrenPath); err == nil {
		entry.Info("PID 1 children in restored namespace", "children", strings.TrimSpace(string(data)))
	}

	criu.LogRestoreSummary(restoreLogPath, entry)
}

func parseProcExitCodeRaw(statLine string) (int, error) {
	statLine = strings.TrimSpace(statLine)
	if statLine == "" {
		return 0, fmt.Errorf("empty stat line")
	}
	paren := strings.LastIndex(statLine, ")")
	if paren < 0 || paren+2 > len(statLine) {
		return 0, fmt.Errorf("malformed stat line")
	}
	fields := strings.Fields(statLine[paren+2:])
	if len(fields) == 0 {
		return 0, fmt.Errorf("malformed stat fields")
	}
	raw, err := strconv.Atoi(fields[len(fields)-1])
	if err != nil {
		return 0, err
	}
	return raw, nil
}

func decodeProcExitCode(raw int) (exitStatus int, termSignal int, coreDumped bool) {
	exitStatus = (raw >> 8) & 0xff
	termSignal = raw & 0x7f
	coreDumped = (raw & 0x80) != 0
	return exitStatus, termSignal, coreDumped
}
