package nsrunner

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/go-logr/logr"
)

const maxOutputLines = 400

func Run(ctx context.Context, args []string, log logr.Logger) (int, int, string, error) {
	cmd := exec.CommandContext(ctx, "nsenter", args...)
	log.V(1).Info("Executing nsenter + ns-restore-runner", "cmd", cmd.String())

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return 0, 0, "", fmt.Errorf("failed to open ns-restore-runner stdout pipe: %w", err)
	}
	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		return 0, 0, "", fmt.Errorf("failed to open ns-restore-runner stderr pipe: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return 0, 0, "", fmt.Errorf("failed to start nsenter + ns-restore-runner: %w", err)
	}

	var (
		mu              sync.Mutex
		capturedLines   []string
		restoredPID     int
		restoredHostPID int
	)

	appendCapturedLine := func(line string) {
		mu.Lock()
		defer mu.Unlock()
		if len(capturedLines) >= maxOutputLines {
			capturedLines = append(capturedLines[1:], line)
			return
		}
		capturedLines = append(capturedLines, line)
	}

	setMarker := func(key, rawValue string) error {
		value, parseErr := strconv.Atoi(rawValue)
		if parseErr != nil {
			return fmt.Errorf("failed to parse %s from ns-restore-runner output: %w", key, parseErr)
		}
		mu.Lock()
		defer mu.Unlock()
		switch key {
		case "RESTORED_PID":
			restoredPID = value
		case "RESTORED_HOST_PID":
			restoredHostPID = value
		}
		return nil
	}

	consumePipe := func(reader io.Reader, stream string) error {
		scanner := bufio.NewScanner(reader)
		scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" {
				continue
			}
			appendCapturedLine(line)

			if strings.HasPrefix(line, "RESTORED_PID=") {
				if err := setMarker("RESTORED_PID", strings.TrimPrefix(line, "RESTORED_PID=")); err != nil {
					return err
				}
				continue
			}
			if strings.HasPrefix(line, "RESTORED_HOST_PID=") {
				if err := setMarker("RESTORED_HOST_PID", strings.TrimPrefix(line, "RESTORED_HOST_PID=")); err != nil {
					return err
				}
				continue
			}

			logNSRestoreRunnerLine(line, stream)
		}
		if scanErr := scanner.Err(); scanErr != nil {
			return fmt.Errorf("failed to read ns-restore-runner %s: %w", stream, scanErr)
		}
		return nil
	}

	pipeErrCh := make(chan error, 2)
	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		pipeErrCh <- consumePipe(stdoutPipe, "stdout")
	}()
	go func() {
		defer wg.Done()
		pipeErrCh <- consumePipe(stderrPipe, "stderr")
	}()

	waitErr := cmd.Wait()
	wg.Wait()
	close(pipeErrCh)

	for pipeErr := range pipeErrCh {
		if pipeErr != nil {
			return 0, 0, "", pipeErr
		}
	}

	output := "<no output>"
	mu.Lock()
	if len(capturedLines) > 0 {
		output = strings.Join(capturedLines, "\n")
	}
	parsedRestoredPID := restoredPID
	parsedRestoredHostPID := restoredHostPID
	mu.Unlock()

	if waitErr != nil {
		return 0, 0, output, waitErr
	}

	if parsedRestoredPID > 0 {
		return parsedRestoredPID, parsedRestoredHostPID, output, nil
	}

	return 0, 0, output, fmt.Errorf("ns-restore-runner did not output RESTORED_PID")
}

func IsUnsupportedFlagError(output string, flags []string) bool {
	if !strings.Contains(output, "flag provided but not defined") {
		return false
	}
	for _, flag := range flags {
		name := strings.TrimLeft(flag, "-")
		if name != "" && strings.Contains(output, name) {
			return true
		}
	}
	return false
}

func logNSRestoreRunnerLine(line, stream string) {
	level, message, fields, ok := parseRunnerLogLine(line)
	if !ok {
		ts := time.Now().UTC().Format(time.RFC3339Nano)
		kv := map[string]interface{}{"source": "ns-restore-runner", "stream": stream}
		encoded, err := json.Marshal(kv)
		if err != nil {
			encoded = []byte(`{"source":"ns-restore-runner"}`)
		}
		fmt.Fprintf(os.Stdout, "%s\tINFO\tnsrunner\tnsrunner/run.go:0\t%s\t%s\n", ts, line, string(encoded))
		return
	}

	helperTime := ""
	if fields != nil {
		if raw, exists := fields["helper_time"]; exists {
			if s, ok := raw.(string); ok {
				helperTime = strings.TrimSpace(s)
			}
			delete(fields, "helper_time")
		}
	} else {
		fields = map[string]interface{}{}
	}
	if helperTime == "" {
		helperTime = time.Now().UTC().Format(time.RFC3339Nano)
	}

	fields["source"] = "ns-restore-runner"
	if stream == "stderr" {
		fields["stream"] = stream
	}

	encoded, err := json.Marshal(fields)
	if err != nil {
		encoded = []byte(`{"source":"ns-restore-runner"}`)
	}
	fmt.Fprintf(
		os.Stdout,
		"%s\t%s\tnsrunner\tnsrunner/run.go:0\t%s\t%s\n",
		helperTime,
		zapLevel(level),
		message,
		string(encoded),
	)
}

func zapLevel(level string) string {
	switch strings.ToLower(strings.TrimSpace(level)) {
	case "trace", "debug":
		return "DEBUG"
	case "warn", "warning":
		return "WARN"
	case "error", "fatal", "panic", "dpanic":
		return "ERROR"
	default:
		return "INFO"
	}
}

func parseRunnerLogLine(line string) (string, string, map[string]interface{}, bool) {
	if level, msg, fields, ok := parseZapDevLine(line); ok {
		return level, msg, fields, true
	}

	if !strings.Contains(line, " level=") || !strings.Contains(line, " msg=") {
		return "info", "", nil, false
	}

	level := "info"
	msg := ""
	fields := map[string]interface{}{}

	tokens := splitLogfmtTokens(line)
	for _, token := range tokens {
		parts := strings.SplitN(token, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key := parts[0]
		value := strings.Trim(parts[1], "\"")
		value = strings.ReplaceAll(value, `\"`, `"`)

		switch key {
		case "time":
			if strings.TrimSpace(value) != "" {
				fields["helper_time"] = strings.TrimSpace(value)
			}
		case "level":
			level = strings.ToLower(strings.TrimSpace(value))
		case "msg":
			msg = value
		default:
			fields[key] = value
		}
	}

	if msg == "" {
		return "info", "", nil, false
	}
	return level, msg, fields, true
}

func parseZapDevLine(line string) (string, string, map[string]interface{}, bool) {
	parts := strings.Split(line, "\t")
	if len(parts) < 4 {
		return "", "", nil, false
	}

	level := strings.TrimSpace(parts[1])
	levelLower := strings.ToLower(level)
	switch levelLower {
	case "debug", "info", "warn", "warning", "error", "dpanic", "panic", "fatal":
	default:
		return "", "", nil, false
	}

	msg := strings.TrimSpace(parts[3])
	if msg == "" {
		return "", "", nil, false
	}

	var fields map[string]interface{}
	if len(parts) >= 5 {
		jsonStr := strings.TrimSpace(parts[4])
		if strings.HasPrefix(jsonStr, "{") {
			parsed := map[string]interface{}{}
			if err := json.Unmarshal([]byte(jsonStr), &parsed); err == nil {
				fields = parsed
			}
		}
	}
	if strings.TrimSpace(parts[0]) != "" {
		if fields == nil {
			fields = map[string]interface{}{}
		}
		fields["helper_time"] = strings.TrimSpace(parts[0])
	}

	return levelLower, msg, fields, true
}

func splitLogfmtTokens(line string) []string {
	var tokens []string
	var current strings.Builder
	inQuotes := false
	escaped := false

	for _, ch := range line {
		switch {
		case escaped:
			current.WriteRune(ch)
			escaped = false
		case ch == '\\' && inQuotes:
			current.WriteRune(ch)
			escaped = true
		case ch == '"':
			current.WriteRune(ch)
			inQuotes = !inQuotes
		case ch == ' ' && !inQuotes:
			if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}
		default:
			current.WriteRune(ch)
		}
	}

	if current.Len() > 0 {
		tokens = append(tokens, current.String())
	}
	return tokens
}
