// restore.go orchestrates external restore from the DaemonSet.
// All operations happen externally: rootfs replay via /host/proc/<PID>/root,
// CRIU restore via nsenter + criu-helper, CUDA restore via cuda-checkpoint.
package externalrestore

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"syscall"

	"golang.org/x/sys/unix"
	"strconv"
	"strings"
	"time"

	"github.com/go-logr/logr"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/checkpoint"
)

const (
	// CRIUHelperBinary is the path to the criu-helper binary in the placeholder image.
	CRIUHelperBinary = "/usr/local/bin/criu-helper"

	// RestoreLogFilename is the CRIU restore log filename.
	RestoreLogFilename = "restore.log"

	// Max number of helper output lines to retain for error reporting.
	maxCRIUHelperOutputLines = 400
)

// RestorerConfig holds configuration for the external restore orchestrator.
type RestorerConfig struct {
	CheckpointBasePath string // Base path for checkpoint storage (PVC mount)
	CRIUHelperPath     string // Path to criu-helper binary (default: CRIUHelperBinary)
}

// Restorer orchestrates external restore operations from the DaemonSet.
type Restorer struct {
	cfg             RestorerConfig
	discoveryClient *checkpoint.DiscoveryClient
	log             logr.Logger
}

// NewRestorer creates a new external restore orchestrator.
func NewRestorer(cfg RestorerConfig, discoveryClient *checkpoint.DiscoveryClient, log logr.Logger) *Restorer {
	if cfg.CRIUHelperPath == "" {
		cfg.CRIUHelperPath = CRIUHelperBinary
	}
	return &Restorer{
		cfg:             cfg,
		discoveryClient: discoveryClient,
		log:             log,
	}
}

// Restore performs external restore for the given request.
func (r *Restorer) Restore(ctx context.Context, req RestoreAPIRequest) (*RestoreAPIResponse, error) {
	restoreStart := time.Now()
	r.log.Info("=== Starting external restore ===",
		"checkpoint_hash", req.CheckpointHash,
		"pod", req.PodName,
		"namespace", req.PodNamespace,
		"container", req.ContainerName,
	)

	checkpointPath := filepath.Join(r.cfg.CheckpointBasePath, req.CheckpointHash)

	// Load checkpoint manifest
	manifest, err := checkpoint.ReadCheckpointManifest(checkpointPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read checkpoint manifest: %w", err)
	}

	// Resolve the placeholder container to get its PID
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
	if manifest.ExternalRestore != nil && manifest.ExternalRestore.CUDA != nil && len(manifest.ExternalRestore.CUDA.PIDs) > 0 {
		if len(manifest.ExternalRestore.CUDA.SourceGPUUUIDs) == 0 {
			return nil, fmt.Errorf("missing source GPU UUIDs in checkpoint manifest")
		}
		targetGPUUUIDs, err := checkpoint.GetPodGPUUUIDsWithRetry(ctx, req.PodName, req.PodNamespace, containerName, r.log)
		if err != nil {
			return nil, fmt.Errorf("failed to get target GPU UUIDs: %w", err)
		}
		if len(targetGPUUUIDs) == 0 {
			return nil, fmt.Errorf("missing target GPU UUIDs for %s/%s container %s", req.PodNamespace, req.PodName, containerName)
		}
		cudaDeviceMap, err = checkpoint.BuildCUDADeviceMap(manifest.ExternalRestore.CUDA.SourceGPUUUIDs, targetGPUUUIDs)
		if err != nil {
			return nil, fmt.Errorf("failed to build CUDA device map: %w", err)
		}
	}

	var completedSteps []string

	// Step 1: Apply rootfs diff into placeholder rootfs via /host/proc/<PID>/root
	targetRoot := fmt.Sprintf("%s/%d/root", checkpoint.HostProcPath, placeholderPID)
	if err := applyRootfsDiff(checkpointPath, targetRoot, r.log); err != nil {
		return nil, fmt.Errorf("rootfs diff failed: %w", err)
	}
	if err := applyDeletedFiles(checkpointPath, targetRoot, r.log); err != nil {
		r.log.Error(err, "Failed to apply deleted files")
	}
	completedSteps = append(completedSteps, "rootfs")

	// Step 2: Restore /dev/shm into placeholder
	if err := restoreDevShm(checkpointPath, targetRoot, r.log); err != nil {
		r.log.Error(err, "Failed to restore /dev/shm")
	}

	// Step 2.5: Ensure /dev/net/tun exists in placeholder rootfs.
	// CRIU needs this character device (major 10, minor 200) to restore TUN/TAP
	// network devices. The unprivileged placeholder container doesn't have it,
	// but the DaemonSet (running as root) can create it via /host/proc/<PID>/root.
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

	// Step 3: Create link_remap stubs in placeholder rootfs
	if err := createLinkRemapStubs(checkpointPath, targetRoot, r.log); err != nil {
		r.log.Error(err, "Failed to create link_remap stubs")
	}

	// Step 4: Execute nsenter + criu-helper
	restoredPID, restoredHostPID, err := r.executeCRIURestore(ctx, placeholderPID, checkpointPath, manifest, cudaDeviceMap)
	if err != nil {
		return nil, fmt.Errorf("CRIU restore failed: %w", err)
	}
	completedSteps = append(completedSteps, "criu")
	r.log.Info("CRIU restore completed",
		"restored_pid", restoredPID,
		"restored_host_pid", restoredHostPID,
	)

	// Step 5: CUDA restore runs inside criu-helper after CRIU restore.
	if cudaDeviceMap != "" {
		completedSteps = append(completedSteps, "cuda")
	}

	// Step 6: Fail closed if the restored process already exited before we return.
	// Ongoing health after this point is delegated to Kubernetes probes.
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

	return &RestoreAPIResponse{
		Success:         true,
		RestoredPID:     restoredPID,
		RestoredHostPID: restoredHostPID,
		CompletedSteps:  completedSteps,
	}, nil
}

// executeCRIURestore runs criu-helper inside the placeholder's namespaces via nsenter.
func (r *Restorer) executeCRIURestore(ctx context.Context, placeholderPID int, checkpointPath string, manifest *checkpoint.CheckpointManifest, cudaDeviceMap string) (int, int, error) {
	pidStr := strconv.Itoa(placeholderPID)

	// Build nsenter command for the namespaces CRIU restore needs.
	// We intentionally do not enter the cgroup namespace when cgroup management is ignored.
	args := []string{
		"-t", pidStr,
		"-m",
		"-n",
		"-p",
		"-i",
		"-u",
		"--", r.cfg.CRIUHelperPath,
		"--checkpoint-path", checkpointPath,
	}

	// Pass CRIU settings from manifest
	if manifest.CRIUDump.CRIU.WorkDir != "" {
		args = append(args, "--work-dir", manifest.CRIUDump.CRIU.WorkDir)
	}
	if cudaDeviceMap != "" {
		args = append(args, "--cuda-device-map", cudaDeviceMap)
	}

	cmd := exec.CommandContext(ctx, "nsenter", args...)
	r.log.V(1).Info("Executing nsenter + criu-helper", "cmd", cmd.String())

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return 0, 0, fmt.Errorf("failed to open criu-helper stdout pipe: %w", err)
	}
	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		return 0, 0, fmt.Errorf("failed to open criu-helper stderr pipe: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return 0, 0, fmt.Errorf("failed to start nsenter + criu-helper: %w", err)
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
			return 0, 0, pipeErr
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
		return 0, 0, fmt.Errorf("nsenter + criu-helper failed: %w\noutput: %s", waitErr, output)
	}

	if parsedRestoredPID > 0 {
		return parsedRestoredPID, parsedRestoredHostPID, nil
	}

	return 0, 0, fmt.Errorf("criu-helper did not output RESTORED_PID; output: %s", output)
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
	// Try zap development console format first (tab-delimited):
	//   TIMESTAMP\tLEVEL\tLOGGER\tMESSAGE\t{json fields}
	if level, msg, fields, ok := parseZapDevLine(line); ok {
		return level, msg, fields, true
	}

	// Fall back to logfmt (key=value pairs).
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
			// Drop nested helper timestamp; outer log line already has one.
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

// parseZapDevLine parses zap's development console format:
//
//	TIMESTAMP\tLEVEL\tLOGGER\tMESSAGE[\t{json fields}]
//
// Returns level, message, parsed JSON fields, and whether parsing succeeded.
func parseZapDevLine(line string) (string, string, map[string]interface{}, bool) {
	parts := strings.Split(line, "\t")
	// Need at least: timestamp, level, logger, message
	if len(parts) < 4 {
		return "", "", nil, false
	}

	level := strings.TrimSpace(parts[1])
	levelLower := strings.ToLower(level)
	switch levelLower {
	case "debug", "info", "warn", "warning", "error", "dpanic", "panic", "fatal":
		// valid zap level
	default:
		return "", "", nil, false
	}

	msg := strings.TrimSpace(parts[3])
	if msg == "" {
		return "", "", nil, false
	}

	// Parse optional JSON structured fields from the 5th tab-separated segment
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

// applyRootfsDiff extracts rootfs-diff.tar into the target root.
func applyRootfsDiff(checkpointPath, targetRoot string, log logr.Logger) error {
	rootfsDiffPath := filepath.Join(checkpointPath, checkpoint.RootfsDiffFilename)
	if _, err := os.Stat(rootfsDiffPath); os.IsNotExist(err) {
		log.V(1).Info("No rootfs-diff.tar, skipping")
		return nil
	}

	log.Info("Applying rootfs diff", "target", targetRoot)
	cmd := exec.Command("tar", "--keep-old-files", "-C", targetRoot, "-xf", rootfsDiffPath)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	if err != nil {
		// tar exit codes 1-2 with --keep-old-files are non-fatal:
		// 1 = "file changed as we read it" or general warnings
		// 2 = "Cannot open: File exists" for read-only files
		// Both are expected when extracting over an existing rootfs.
		if exitErr, ok := err.(*exec.ExitError); ok && exitErr.ExitCode() <= 2 {
			log.V(1).Info("Rootfs diff applied (some files skipped)", "exit_code", exitErr.ExitCode())
			return nil
		}
		return fmt.Errorf("tar extract failed: %w", err)
	}
	return nil
}

// applyDeletedFiles removes files marked as deleted in the checkpoint.
func applyDeletedFiles(checkpointPath, targetRoot string, log logr.Logger) error {
	deletedFilesPath := filepath.Join(checkpointPath, checkpoint.DeletedFilesFilename)
	data, err := os.ReadFile(deletedFilesPath)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("failed to read deleted files: %w", err)
	}

	var deletedFiles []string
	if err := json.Unmarshal(data, &deletedFiles); err != nil {
		return fmt.Errorf("failed to parse deleted files: %w", err)
	}

	count := 0
	for _, f := range deletedFiles {
		if f == "" {
			continue
		}
		target := filepath.Join(targetRoot, f)
		if _, err := os.Stat(target); os.IsNotExist(err) {
			continue
		}
		if err := os.RemoveAll(target); err != nil {
			log.V(1).Info("Could not delete file", "path", target, "error", err)
			continue
		}
		count++
	}
	log.Info("Deleted files applied", "count", count)
	return nil
}

// restoreDevShm restores /dev/shm files into the target root's /dev/shm.
func restoreDevShm(checkpointPath, targetRoot string, log logr.Logger) error {
	srcDir := filepath.Join(checkpointPath, checkpoint.DevShmDirName)
	entries, err := os.ReadDir(srcDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("failed to read dev-shm dir: %w", err)
	}

	destDir := filepath.Join(targetRoot, "dev", "shm")
	if err := os.MkdirAll(destDir, 0777); err != nil {
		return fmt.Errorf("failed to create target /dev/shm: %w", err)
	}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		srcPath := filepath.Join(srcDir, entry.Name())
		destPath := filepath.Join(destDir, entry.Name())

		info, err := entry.Info()
		if err != nil {
			continue
		}
		uid, gid := -1, -1
		if stat, ok := info.Sys().(*syscall.Stat_t); ok {
			uid = int(stat.Uid)
			gid = int(stat.Gid)
		}

		src, err := os.Open(srcPath)
		if err != nil {
			continue
		}

		mode := info.Mode()
		if mode == 0 {
			mode = 0666
		}
		dst, err := os.OpenFile(destPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode)
		if err != nil {
			src.Close()
			continue
		}

		if _, err := io.Copy(dst, src); err != nil {
			src.Close()
			dst.Close()
			continue
		}
		if uid >= 0 && gid >= 0 {
			if err := dst.Chown(uid, gid); err != nil {
				src.Close()
				dst.Close()
				continue
			}
		}
		src.Close()
		dst.Close()
	}

	log.V(1).Info("Restored /dev/shm files", "count", len(entries))
	return nil
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
