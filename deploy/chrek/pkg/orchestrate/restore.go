package orchestrate

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"golang.org/x/sys/unix"

	"github.com/go-logr/logr"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/config"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/containerd"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/cuda"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/filesystem"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/manifest"
)

const (
	// CRIUHelperBinary is the path to the criu-helper binary in the placeholder image.
	CRIUHelperBinary = "/usr/local/bin/criu-helper"

	// RestoreLogFilename is the CRIU restore log filename.
	RestoreLogFilename = "restore.log"

	maxCRIUHelperOutputLines = 400
)

// RestorerConfig holds configuration for the external restore orchestrator.
type RestorerConfig struct {
	CheckpointBasePath string
	CRIUHelperPath     string
	CRIUSettings       *config.CRIUSettings
}

// Restorer orchestrates external restore operations from the DaemonSet.
type Restorer struct {
	cfg             RestorerConfig
	discoveryClient *containerd.DiscoveryClient
	log             logr.Logger
}

// NewRestorer creates a new external restore orchestrator.
func NewRestorer(cfg RestorerConfig, discoveryClient *containerd.DiscoveryClient, log logr.Logger) *Restorer {
	if cfg.CRIUHelperPath == "" {
		cfg.CRIUHelperPath = CRIUHelperBinary
	}
	return &Restorer{
		cfg:             cfg,
		discoveryClient: discoveryClient,
		log:             log,
	}
}

// RestoreRequest holds the parameters for a restore operation.
type RestoreRequest struct {
	CheckpointHash string
	PodName        string
	PodNamespace   string
	ContainerName  string
}

// RestoreResult contains the result of a restore operation.
type RestoreResult struct {
	RestoredPID     int
	RestoredHostPID int
	CompletedSteps  []string
}

// Restore performs external restore for the given request.
func (r *Restorer) Restore(ctx context.Context, req RestoreRequest) (*RestoreResult, error) {
	restoreStart := time.Now()
	r.log.Info("=== Starting external restore ===",
		"checkpoint_hash", req.CheckpointHash,
		"pod", req.PodName,
		"namespace", req.PodNamespace,
		"container", req.ContainerName,
	)

	checkpointPath := filepath.Join(r.cfg.CheckpointBasePath, req.CheckpointHash)

	m, err := manifest.Read(checkpointPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read checkpoint manifest: %w", err)
	}

	containerName := req.ContainerName
	if containerName == "" {
		containerName = "main"
	}

	placeholderPID, _, err := r.discoveryClient.ResolveContainerByPod(ctx, req.PodName, req.PodNamespace, containerName)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve placeholder container: %w", err)
	}
	r.log.Info("Resolved placeholder container", "pid", placeholderPID)

	cudaDeviceMap := ""
	if m.ExternalRestore != nil && m.ExternalRestore.CUDA != nil && len(m.ExternalRestore.CUDA.PIDs) > 0 {
		if len(m.ExternalRestore.CUDA.SourceGPUUUIDs) == 0 {
			return nil, fmt.Errorf("missing source GPU UUIDs in checkpoint manifest")
		}
		targetGPUUUIDs, err := cuda.GetPodGPUUUIDsWithRetry(ctx, req.PodName, req.PodNamespace, containerName, r.log)
		if err != nil {
			return nil, fmt.Errorf("failed to get target GPU UUIDs: %w", err)
		}
		if len(targetGPUUUIDs) == 0 {
			return nil, fmt.Errorf("missing target GPU UUIDs for %s/%s container %s", req.PodNamespace, req.PodName, containerName)
		}
		cudaDeviceMap, err = cuda.BuildDeviceMap(m.ExternalRestore.CUDA.SourceGPUUUIDs, targetGPUUUIDs)
		if err != nil {
			return nil, fmt.Errorf("failed to build CUDA device map: %w", err)
		}
	}

	var completedSteps []string

	// Step 1: Apply rootfs diff
	targetRoot := fmt.Sprintf("%s/%d/root", config.HostProcPath, placeholderPID)
	if err := filesystem.ApplyRootfsDiff(checkpointPath, targetRoot, r.log); err != nil {
		return nil, fmt.Errorf("rootfs diff failed: %w", err)
	}
	if err := filesystem.ApplyDeletedFiles(checkpointPath, targetRoot, r.log); err != nil {
		r.log.Error(err, "Failed to apply deleted files")
	}
	completedSteps = append(completedSteps, "rootfs")

	// Step 2: Restore /dev/shm
	if err := filesystem.RestoreDevShm(checkpointPath, targetRoot, r.log); err != nil {
		r.log.Error(err, "Failed to restore /dev/shm")
	}

	// Step 2.5: Ensure /dev/net/tun exists in placeholder rootfs
	tunPath := filepath.Join(targetRoot, "dev/net/tun")
	if _, statErr := os.Stat(tunPath); os.IsNotExist(statErr) {
		if err := os.MkdirAll(filepath.Dir(tunPath), 0755); err != nil {
			r.log.Error(err, "Failed to create /dev/net dir in placeholder")
		} else if err := syscall.Mknod(tunPath, syscall.S_IFCHR|0666, int(unix.Mkdev(10, 200))); err != nil {
			r.log.Error(err, "Failed to create /dev/net/tun in placeholder")
		} else {
			r.log.Info("Created /dev/net/tun in placeholder rootfs")
		}
	}

	// Step 3: Create link_remap stubs
	if err := filesystem.CreateLinkRemapStubs(checkpointPath, targetRoot, r.log); err != nil {
		r.log.Error(err, "Failed to create link_remap stubs")
	}

	// Step 4: Execute nsenter + criu-helper
	restoredPID, restoredHostPID, err := r.executeCRIURestore(ctx, placeholderPID, checkpointPath, m, cudaDeviceMap)
	if err != nil {
		return nil, fmt.Errorf("CRIU restore failed: %w", err)
	}
	completedSteps = append(completedSteps, "criu")
	r.log.Info("CRIU restore completed", "restored_pid", restoredPID, "restored_host_pid", restoredHostPID)

	if cudaDeviceMap != "" {
		completedSteps = append(completedSteps, "cuda")
	}

	// Step 5: Validate restored process
	procRoot := filepath.Join(targetRoot, "proc")
	restoreLogPath := filepath.Join(targetRoot, "var", "criu-work", RestoreLogFilename)
	if err := validateRestoredProcessState(procRoot, restoredPID); err != nil {
		r.log.Error(err, "Restored process failed immediate post-restore validation",
			"restored_pid", restoredPID,
			"proc_root", procRoot,
		)
		logRestoredProcessDiagnostics(procRoot, restoredPID, restoreLogPath, r.log)
		return nil, fmt.Errorf("restored process failed post-restore validation: %w", err)
	}

	totalDuration := time.Since(restoreStart)
	r.log.Info("=== External restore completed ===",
		"total_duration", totalDuration,
		"restored_pid", restoredPID,
		"restored_host_pid", restoredHostPID,
		"steps", completedSteps,
	)

	return &RestoreResult{
		RestoredPID:     restoredPID,
		RestoredHostPID: restoredHostPID,
		CompletedSteps:  completedSteps,
	}, nil
}

func (r *Restorer) executeCRIURestore(ctx context.Context, placeholderPID int, checkpointPath string, m *manifest.CheckpointManifest, cudaDeviceMap string) (int, int, error) {
	pidStr := strconv.Itoa(placeholderPID)

	baseArgs := []string{
		"-t", pidStr,
		"-m", "-n", "-p", "-i", "-u",
		"--", r.cfg.CRIUHelperPath,
		"--checkpoint-path", checkpointPath,
	}

	if m.CRIUDump.CRIU.WorkDir != "" {
		baseArgs = append(baseArgs, "--work-dir", m.CRIUDump.CRIU.WorkDir)
	}
	if cudaDeviceMap != "" {
		baseArgs = append(baseArgs, "--cuda-device-map", cudaDeviceMap)
	}

	restoreFlags := []string{}

	// Pass restore-specific CRIU options from ConfigMap
	if r.cfg.CRIUSettings != nil {
		if r.cfg.CRIUSettings.RstSibling {
			restoreFlags = append(restoreFlags, "--rst-sibling")
		}
		if r.cfg.CRIUSettings.MntnsCompatMode {
			restoreFlags = append(restoreFlags, "--mntns-compat-mode")
		}
		if r.cfg.CRIUSettings.EvasiveDevices {
			restoreFlags = append(restoreFlags, "--evasive-devices")
		}
		if r.cfg.CRIUSettings.ForceIrmap {
			restoreFlags = append(restoreFlags, "--force-irmap")
		}
	}

	args := append(append([]string{}, baseArgs...), restoreFlags...)
	restoredPID, restoredHostPID, output, err := r.runCRIUHelper(ctx, args)
	if err != nil && len(restoreFlags) > 0 && isUnsupportedCRIUHelperFlag(output, restoreFlags) {
		r.log.Info("Retrying restore without unsupported optional criu-helper flags", "flags", strings.Join(restoreFlags, " "))
		restoredPID, restoredHostPID, output, err = r.runCRIUHelper(ctx, baseArgs)
	}
	if err != nil {
		return 0, 0, fmt.Errorf("nsenter + criu-helper failed: %w\noutput: %s", err, output)
	}
	return restoredPID, restoredHostPID, nil
}

func (r *Restorer) runCRIUHelper(ctx context.Context, args []string) (int, int, string, error) {
	cmd := exec.CommandContext(ctx, "nsenter", args...)
	r.log.V(1).Info("Executing nsenter + criu-helper", "cmd", cmd.String())

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return 0, 0, "", fmt.Errorf("failed to open criu-helper stdout pipe: %w", err)
	}
	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		return 0, 0, "", fmt.Errorf("failed to open criu-helper stderr pipe: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return 0, 0, "", fmt.Errorf("failed to start nsenter + criu-helper: %w", err)
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
		if len(capturedLines) >= maxCRIUHelperOutputLines {
			capturedLines = append(capturedLines[1:], line)
			return
		}
		capturedLines = append(capturedLines, line)
	}

	setMarker := func(key, rawValue string) error {
		value, parseErr := strconv.Atoi(rawValue)
		if parseErr != nil {
			return fmt.Errorf("failed to parse %s from criu-helper output: %w", key, parseErr)
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

			if stream == "stderr" {
				r.log.Info(line, "source", "criu-helper", "stream", "stderr")
				continue
			}
			r.logCRIUHelperLine(line)
		}
		if scanErr := scanner.Err(); scanErr != nil {
			return fmt.Errorf("failed to read criu-helper %s: %w", stream, scanErr)
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

	return 0, 0, output, fmt.Errorf("criu-helper did not output RESTORED_PID")
}

func isUnsupportedCRIUHelperFlag(output string, flags []string) bool {
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

func (r *Restorer) logCRIUHelperLine(line string) {
	level, message, fields, ok := parseHelperLogLine(line)
	if !ok {
		r.log.Info(line, "source", "criu-helper")
		return
	}

	keysAndValues := make([]interface{}, 0, len(fields)*2+2)
	keysAndValues = append(keysAndValues, "source", "criu-helper")
	for key, value := range fields {
		keysAndValues = append(keysAndValues, key, value)
	}

	switch level {
	case "trace", "debug":
		r.log.V(1).Info(message, keysAndValues...)
	case "warn", "warning":
		r.log.Info(message, keysAndValues...)
	case "error", "fatal", "panic":
		r.log.Error(fmt.Errorf("%s", message), "criu-helper message", keysAndValues...)
	default:
		r.log.Info(message, keysAndValues...)
	}
}

func parseHelperLogLine(line string) (string, string, map[string]interface{}, bool) {
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

func validateRestoredProcessState(procRoot string, pid int) error {
	if pid <= 0 {
		return fmt.Errorf("invalid restored PID %d", pid)
	}

	state, err := readProcessState(procRoot, pid)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("process %d exited", pid)
		}
		return fmt.Errorf("failed to inspect process %d: %w", pid, err)
	}
	if state == "Z" {
		return fmt.Errorf("process %d became zombie", pid)
	}
	return nil
}

func readProcessState(procRoot string, pid int) (string, error) {
	statusPath := filepath.Join(procRoot, strconv.Itoa(pid), "status")
	data, err := os.ReadFile(statusPath)
	if err != nil {
		return "", err
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "State:") {
			fields := strings.Fields(line)
			if len(fields) > 1 {
				return fields[1], nil
			}
			break
		}
	}
	return "", fmt.Errorf("state not found in %s", statusPath)
}

func logRestoredProcessDiagnostics(procRoot string, pid int, restoreLogPath string, log logr.Logger) {
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

	logCRIURestoreSummary(restoreLogPath, entry)
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

func logCRIURestoreSummary(path string, log logr.Logger) {
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
