// criu-helper is a self-contained binary that performs CRIU restore inside container
// namespaces. It is invoked by the DaemonSet via:
//
//	nsenter -t <PID> -m -n -p -i -- /usr/local/bin/criu-helper --checkpoint-path <path>
//
// It runs inside the placeholder container's mount/net/PID/IPC namespaces and:
//  1. Remounts /proc/sys read-write (CRIU needs to write sysctl)
//  2. Opens checkpoint images directory and net NS file
//  3. Uses AddInheritFd for proper FD passing to CRIU's swrk child
//  4. Generates ExtMountMaps from /proc/1/mountinfo
//  5. Creates link_remap stubs for cross-node restore
//  6. Calls go-criu Restore()
//  7. Remounts /proc/sys read-only
//  8. Prints RESTORED_PID=<N> to stdout
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/go-logr/logr"
	"github.com/go-logr/zapr"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	criu "github.com/checkpoint-restore/go-criu/v8"
	"github.com/checkpoint-restore/go-criu/v8/crit"
	"github.com/checkpoint-restore/go-criu/v8/crit/images/fdinfo"
	criurpc "github.com/checkpoint-restore/go-criu/v8/rpc"
	"google.golang.org/protobuf/proto"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/common"
)

const (
	netNsPath           = "/proc/1/ns/net"
	mountInfoPath       = "/proc/1/mountinfo"
	restoreLogFile      = "restore.log"
	procStdoutPath      = "/proc/1/fd/1"
	procStderrPath      = "/proc/1/fd/2"
	criuBinaryPath      = "/usr/local/sbin/criu"
	cudaCheckpointBin = "/usr/local/sbin/cuda-checkpoint"
	cudaActionRestore = "restore"
	cudaActionUnlock  = "unlock"
	cudaDiscoverTick  = 1 * time.Second
)

// restoreOptions holds restore-specific CRIU options passed via CLI flags from the DaemonSet agent.
type restoreOptions struct {
	RstSibling      bool
	MntnsCompatMode bool
	EvasiveDevices  bool
	ForceIrmap      bool
}

func main() {
	rootLog := configureLogging()

	checkpointPath := flag.String("checkpoint-path", "", "Path to checkpoint directory")
	workDir := flag.String("work-dir", "", "CRIU work directory")
	cudaDeviceMap := flag.String("cuda-device-map", "", "CUDA device map for cuda-checkpoint restore")
	rstSibling := flag.Bool("rst-sibling", false, "Restore process as sibling (required for go-criu swrk mode)")
	mntnsCompatMode := flag.Bool("mntns-compat-mode", false, "Enable mount namespace compatibility mode")
	evasiveDevices := flag.Bool("evasive-devices", false, "Use any device path if original is inaccessible")
	forceIrmap := flag.Bool("force-irmap", false, "Force resolving inotify/fsnotify watch names")
	flag.Parse()

	log := rootLog.WithName("criu-helper")

	if *checkpointPath == "" {
		fatal(log, nil, "--checkpoint-path is required")
	}

	restoreOpts := restoreOptions{
		RstSibling:      *rstSibling,
		MntnsCompatMode: *mntnsCompatMode,
		EvasiveDevices:  *evasiveDevices,
		ForceIrmap:      *forceIrmap,
	}

	if err := run(*checkpointPath, *workDir, *cudaDeviceMap, restoreOpts, log); err != nil {
		fatal(log, err, "CRIU restore failed")
	}
}

func configureLogging() logr.Logger {
	level := strings.TrimSpace(strings.ToLower(os.Getenv("CHREK_LOG_LEVEL")))
	if level == "" {
		level = "info"
	}

	zapLevel := zapcore.InfoLevel
	parseErr := error(nil)
	switch level {
	case "trace", "debug":
		zapLevel = zapcore.DebugLevel
	case "info":
		zapLevel = zapcore.InfoLevel
	case "warn", "warning":
		zapLevel = zapcore.WarnLevel
	case "error":
		zapLevel = zapcore.ErrorLevel
	default:
		parseErr = fmt.Errorf("invalid level %q", level)
	}

	zapCfg := zap.Config{
		Level:            zap.NewAtomicLevelAt(zapLevel),
		Development:      true,
		Encoding:         "console",
		EncoderConfig:    zap.NewDevelopmentEncoderConfig(),
		OutputPaths:      []string{"stdout"},
		ErrorOutputPaths: []string{"stderr"},
	}
	zapCfg.EncoderConfig.EncodeTime = zapcore.RFC3339NanoTimeEncoder
	zapLog, err := zapCfg.Build()
	if err != nil {
		zapLog, _ = zap.NewDevelopment()
	}

	log := zapr.NewLogger(zapLog)
	if parseErr != nil {
		log.WithName("setup").Info("Invalid CHREK_LOG_LEVEL, falling back to info", "value", level, "error", parseErr)
	}
	return log
}

func fatal(log logr.Logger, err error, msg string, keysAndValues ...interface{}) {
	if err != nil {
		log.Error(err, msg, keysAndValues...)
	} else {
		log.Info(msg, keysAndValues...)
	}
	os.Exit(1)
}

func run(checkpointPath, workDir, cudaDeviceMap string, opts restoreOptions, log logr.Logger) error {
	restoreStart := time.Now()
	log.Info("Starting criu-helper restore workflow",
		"checkpoint_path", checkpointPath,
		"work_dir", workDir,
		"has_cuda_map", cudaDeviceMap != "",
	)

	// Load checkpoint manifest for CRIU settings and mount plan
	manifest, err := checkpoint.ReadCheckpointManifest(checkpointPath)
	if err != nil {
		return fmt.Errorf("failed to read manifest: %w", err)
	}
	log.Info("Loaded checkpoint manifest",
		"ext_mounts", len(manifest.CRIUDump.ExtMnt),
		"criu_log_level", manifest.CRIUDump.CRIU.LogLevel,
		"manage_cgroups_mode", manifest.CRIUDump.CRIU.ManageCgroupsMode,
		"checkpoint_has_cuda", manifest.ExternalRestore != nil && manifest.ExternalRestore.CUDA != nil,
		"checkpoint_cuda_pids", len(getCheckpointCUDAPIDs(manifest)),
		"checkpoint_cuda_gpus", len(getCheckpointSourceGPUUUIDs(manifest)),
		"checkpoint_link_remap", manifest.CRIUDump.CRIU.LinkRemap,
	)

	// Remount /proc/sys rw — CRIU needs to write sysctl values
	log.Info("Remounting /proc/sys read-write", "path", "/proc/sys")
	if err := remountProcSys(true, log); err != nil {
		log.Error(err, "Failed to remount /proc/sys rw (restore may still work)")
	}
	defer remountProcSys(false, log) //nolint:errcheck

	manageCgroupsMode := strings.ToLower(strings.TrimSpace(manifest.CRIUDump.CRIU.ManageCgroupsMode))
	if manageCgroupsMode != "ignore" {
		// Remount cgroup2 rw only for non-ignore cgroup management modes.
		log.Info("Remounting /sys/fs/cgroup read-write", "path", "/sys/fs/cgroup")
		if err := remountCgroupFS(true, log); err != nil {
			log.Error(err, "Failed to remount /sys/fs/cgroup rw (restore may fail with cgroup management enabled)")
		}
		defer remountCgroupFS(false, log) //nolint:errcheck
	} else {
		log.Info("Skipping /sys/fs/cgroup remount (manage cgroups mode is ignore)")
	}

	// Open checkpoint images directory with CLOEXEC cleared for CRIU
	imageDir, imageDirFD, err := common.OpenPathForCRIU(checkpointPath)
	if err != nil {
		return fmt.Errorf("failed to open image directory: %w", err)
	}
	defer imageDir.Close()
	log.Info("Opened checkpoint images directory for CRIU", "images_dir_fd", imageDirFD)

	// Open work directory if specified
	var workDirFile *os.File
	var workDirFD int32 = -1
	if workDir != "" {
		if err := os.MkdirAll(workDir, 0755); err != nil {
			log.Error(err, "Failed to create work directory")
		} else {
			f, fd, err := common.OpenPathForCRIU(workDir)
			if err == nil {
				workDirFile = f
				workDirFD = fd
				defer workDirFile.Close()
				log.Info("Opened CRIU work directory", "work_dir", workDir, "work_dir_fd", workDirFD)
			} else {
				log.Error(err, "Failed to open CRIU work directory", "work_dir", workDir)
			}
		}
	}

	// Generate external mount maps from /proc/1/mountinfo (inside container namespace)
	extMounts, err := generateExtMountMaps(manifest)
	if err != nil {
		return fmt.Errorf("failed to generate ext mount maps: %w", err)
	}
	log.Info("Generated external mount map set",
		"ext_mount_count", len(extMounts),
		"ext_mount_sample", extMountMapSample(extMounts, 8),
	)

	// Build CRIU restore options
	criuOpts := buildRestoreOptions(manifest, imageDirFD, workDirFD, extMounts, opts)
	log.Info("Constructed CRIU restore options",
		"images_dir_fd", criuOpts.GetImagesDirFd(),
		"work_dir_fd", criuOpts.GetWorkDirFd(),
		"log_level", criuOpts.GetLogLevel(),
		"log_file", criuOpts.GetLogFile(),
		"root", criuOpts.GetRoot(),
		"timeout", criuOpts.GetTimeout(),
		"rst_sibling", criuOpts.GetRstSibling(),
		"manage_cgroups", criuOpts.GetManageCgroups(),
		"manage_cgroups_mode", criuOpts.GetManageCgroupsMode().String(),
		"tcp_close", criuOpts.GetTcpClose(),
		"file_locks", criuOpts.GetFileLocks(),
		"ext_unix_sk", criuOpts.GetExtUnixSk(),
		"link_remap", criuOpts.GetLinkRemap(),
		"force_irmap", criuOpts.GetForceIrmap(),
		"evasive_devices", criuOpts.GetEvasiveDevices(),
		"ext_mount_count", len(criuOpts.ExtMnt),
		"mntns_compat_mode", criuOpts.GetMntnsCompatMode(),
	)

	// Reuse criu.conf from checkpoint if it exists
	criuConfPath := filepath.Join(checkpointPath, checkpoint.CheckpointCRIUConfFilename)
	if _, err := os.Stat(criuConfPath); err == nil {
		criuOpts.ConfigFile = proto.String(criuConfPath)
		log.Info("Using checkpointed CRIU config file", "config_file", criuConfPath)
	} else {
		log.Info("No checkpointed CRIU config file, using generated options", "config_file", criuConfPath)
	}

	// Create CRIU client and set up inherited FDs using AddInheritFd
	c := criu.MakeCriu()
	if _, err := os.Stat(criuBinaryPath); err != nil {
		return fmt.Errorf("criu binary not found at %s: %w", criuBinaryPath, err)
	}
	c.SetCriuPath(criuBinaryPath)
	log.Info("Configured CRIU binary", "criu_binary", criuBinaryPath)

	// Open network namespace and register via AddInheritFd for proper FD passing
	netNsFile, err := os.Open(netNsPath)
	if err != nil {
		return fmt.Errorf("failed to open net NS at %s: %w", netNsPath, err)
	}
	defer netNsFile.Close()
	c.AddInheritFd("extNetNs", netNsFile)
	log.Info("Registered inherited network namespace FD", "netns_path", netNsPath, "netns_fd", netNsFile.Fd())

	inheritedStdioFiles, err := registerStdioInheritFDs(c, checkpointPath, log)
	if err != nil {
		log.Error(err, "Failed to configure inherited stdio resources")
	} else {
		defer closeFiles(inheritedStdioFiles)
	}

	// Execute CRIU restore
	notify := &restoreNotify{log: log}
	log.Info("Executing go-criu Restore call")
	if err := c.Restore(criuOpts, notify); err != nil {
		log.Error(err, "go-criu Restore returned error")
		logCRIUErrors(checkpointPath, workDir, log)
		return fmt.Errorf("CRIU restore failed: %w", err)
	}

	log.Info("CRIU restore completed", "pid", notify.restoredPID, "duration", time.Since(restoreStart))

	if err := restoreCUDA(context.Background(), manifest, int(notify.restoredPID), cudaDeviceMap, log); err != nil {
		log.Error(err, "CUDA restore sequence failed")
		return err
	}
	log.Info("CUDA restore sequence completed")

	// Print the restored PID so the DaemonSet can parse it
	fmt.Printf("RESTORED_PID=%d\n", notify.restoredPID)
	if restoredHostPID, err := readRestoredHostPID(int(notify.restoredPID)); err == nil {
		fmt.Printf("RESTORED_HOST_PID=%d\n", restoredHostPID)
		log.Info("Resolved restored host PID from NSpid", "restored_host_pid", restoredHostPID)
	} else {
		log.Error(err, "Failed to resolve restored host PID from NSpid")
	}
	log.Info("criu-helper completed successfully", "restored_pid", notify.restoredPID)
	return nil
}

func readRestoredHostPID(restoredPID int) (int, error) {
	if restoredPID <= 0 {
		return 0, fmt.Errorf("invalid restored PID %d", restoredPID)
	}
	statusPath := fmt.Sprintf("/proc/%d/status", restoredPID)
	data, err := os.ReadFile(statusPath)
	if err != nil {
		return 0, fmt.Errorf("failed to read %s: %w", statusPath, err)
	}
	for _, line := range strings.Split(string(data), "\n") {
		if !strings.HasPrefix(line, "NSpid:") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			return 0, fmt.Errorf("malformed NSpid line in %s", statusPath)
		}
		return strconv.Atoi(fields[1])
	}
	return 0, fmt.Errorf("NSpid not found in %s", statusPath)
}

func restoreCUDA(ctx context.Context, manifest *checkpoint.CheckpointManifest, restoredPID int, deviceMap string, log logr.Logger) error {
	if manifest.ExternalRestore == nil || manifest.ExternalRestore.CUDA == nil || len(manifest.ExternalRestore.CUDA.PIDs) == 0 {
		log.Info("Checkpoint does not contain CUDA metadata, skipping cuda-checkpoint restore")
		return nil
	}
	if deviceMap == "" {
		log.Info("Missing CUDA device map for checkpoint with CUDA metadata", "checkpoint_cuda_pids", len(manifest.ExternalRestore.CUDA.PIDs))
		return fmt.Errorf("missing --cuda-device-map for checkpoint with CUDA state")
	}
	if _, err := os.Stat(cudaCheckpointBin); err != nil {
		log.Error(err, "cuda-checkpoint binary is missing", "cuda_binary", cudaCheckpointBin)
		return fmt.Errorf("cuda-checkpoint not found at %s: %w", cudaCheckpointBin, err)
	}
	log.Info("Starting CUDA restore sequence",
		"restored_pid", restoredPID,
		"checkpoint_cuda_pids", len(manifest.ExternalRestore.CUDA.PIDs),
		"checkpoint_cuda_pids_sample", truncateInts(manifest.ExternalRestore.CUDA.PIDs, 16),
		"device_map", deviceMap,
	)

	attempt := 0
	for {
		attempt++
		candidates := processTreePIDs(restoredPID)
		cudaPIDs := make([]int, 0, len(candidates))
		for _, pid := range candidates {
			ok, err := isCUDAProcess(ctx, pid)
			if err != nil {
				log.V(1).Info("CUDA state probe failed", "pid", pid, "error", err)
				continue
			}
			if ok {
				cudaPIDs = append(cudaPIDs, pid)
			}
		}

		log.Info("CUDA restore PID discovery attempt",
			"attempt", attempt,
			"restored_pid", restoredPID,
			"candidates", len(candidates),
			"cuda_pids", len(cudaPIDs),
			"sample_pids", truncateInts(candidates, 16),
			"sample_cuda", truncateInts(cudaPIDs, 16),
		)

		if len(cudaPIDs) > 0 {
			for _, pid := range cudaPIDs {
				log.Info("Running cuda-checkpoint restore", "pid", pid, "device_map", deviceMap)
				if err := runCudaCheckpoint(ctx, pid, cudaActionRestore, deviceMap, log); err != nil {
					return fmt.Errorf("cuda restore failed for PID %d: %w", pid, err)
				}
				log.Info("Running cuda-checkpoint unlock", "pid", pid)
				if err := runCudaCheckpoint(ctx, pid, cudaActionUnlock, "", log); err != nil {
					return fmt.Errorf("cuda unlock failed for PID %d: %w", pid, err)
				}
			}
			log.Info("CUDA restore completed", "cuda_pids", len(cudaPIDs), "device_map", deviceMap, "restored_pid", restoredPID)
			return nil
		}

		select {
		case <-ctx.Done():
			return fmt.Errorf("checkpoint captured %d CUDA PIDs but none found in restored process tree rooted at PID %d: %w", len(manifest.ExternalRestore.CUDA.PIDs), restoredPID, ctx.Err())
		case <-time.After(cudaDiscoverTick):
		}
	}
}

func processTreePIDs(rootPID int) []int {
	if rootPID <= 0 {
		return nil
	}

	queue := []int{rootPID}
	seen := map[int]struct{}{}
	all := make([]int, 0, 16)

	for len(queue) > 0 {
		pid := queue[0]
		queue = queue[1:]
		if pid <= 0 {
			continue
		}
		if _, ok := seen[pid]; ok {
			continue
		}
		seen[pid] = struct{}{}
		if _, err := os.Stat(fmt.Sprintf("/proc/%d", pid)); err != nil {
			continue
		}
		all = append(all, pid)

		children, err := os.ReadFile(fmt.Sprintf("/proc/%d/task/%d/children", pid, pid))
		if err != nil {
			continue
		}
		for _, child := range strings.Fields(string(children)) {
			childPID, err := strconv.Atoi(child)
			if err != nil {
				continue
			}
			queue = append(queue, childPID)
		}
	}

	return all
}

func isCUDAProcess(ctx context.Context, pid int) (bool, error) {
	cmd := exec.CommandContext(ctx, cudaCheckpointBin, "--get-state", "--pid", strconv.Itoa(pid))
	if err := cmd.Run(); err != nil {
		if ctx.Err() != nil {
			return false, ctx.Err()
		}
		return false, nil
	}
	return true, nil
}

func runCudaCheckpoint(ctx context.Context, pid int, action string, deviceMap string, log logr.Logger) error {
	args := []string{"--action", action, "--pid", strconv.Itoa(pid)}
	if action == cudaActionRestore && deviceMap != "" {
		args = append(args, "--device-map", deviceMap)
	}
	cmd := exec.CommandContext(ctx, cudaCheckpointBin, args...)
	start := time.Now()
	output, err := cmd.CombinedOutput()
	duration := time.Since(start)
	out := strings.TrimSpace(string(output))
	if err != nil {
		return fmt.Errorf("cuda-checkpoint %v failed for pid %d after %s: %w (output: %s)", args, pid, duration, err, out)
	}
	log.Info("cuda-checkpoint command succeeded",
		"pid", pid,
		"action", action,
		"duration", duration,
		"output", out,
	)
	return nil
}

func registerStdioInheritFDs(c *criu.Criu, checkpointPath string, log logr.Logger) ([]*os.File, error) {
	stdoutResources, stderrResources, err := discoverStdioInheritResources(checkpointPath)
	if err != nil {
		return nil, err
	}
	if len(stdoutResources) == 0 && len(stderrResources) == 0 {
		log.Info("No stdio resources found for inherit-fd wiring")
		return nil, nil
	}

	log.Info("Discovered stdio inherit-fd resources",
		"stdout_resources", stdoutResources,
		"stderr_resources", stderrResources,
	)

	openFiles := make([]*os.File, 0, 2)

	if len(stdoutResources) > 0 {
		stdoutFile, err := os.OpenFile(procStdoutPath, os.O_WRONLY, 0)
		if err != nil {
			closeFiles(openFiles)
			return nil, fmt.Errorf("failed to open %s: %w", procStdoutPath, err)
		}
		openFiles = append(openFiles, stdoutFile)
		for _, resource := range stdoutResources {
			c.AddInheritFd(resource, stdoutFile)
		}
		log.Info("Registered inherited stdout resources", "path", procStdoutPath, "resources", stdoutResources)
	}

	if len(stderrResources) > 0 {
		stderrFile, err := os.OpenFile(procStderrPath, os.O_WRONLY, 0)
		if err != nil {
			closeFiles(openFiles)
			return nil, fmt.Errorf("failed to open %s: %w", procStderrPath, err)
		}
		openFiles = append(openFiles, stderrFile)
		for _, resource := range stderrResources {
			c.AddInheritFd(resource, stderrFile)
		}
		log.Info("Registered inherited stderr resources", "path", procStderrPath, "resources", stderrResources)
	}

	return openFiles, nil
}

func discoverStdioInheritResources(checkpointPath string) ([]string, []string, error) {
	resourcesByFileID, err := loadInheritResourcesByFileID(checkpointPath)
	if err != nil {
		return nil, nil, err
	}

	fdinfoPaths, err := filepath.Glob(filepath.Join(checkpointPath, "fdinfo-*.img"))
	if err != nil {
		return nil, nil, fmt.Errorf("failed to list fdinfo images: %w", err)
	}
	if len(fdinfoPaths) == 0 {
		return nil, nil, fmt.Errorf("no fdinfo images found in %s", checkpointPath)
	}
	sort.Strings(fdinfoPaths)

	stdoutSet := map[string]struct{}{}
	stderrSet := map[string]struct{}{}

	for _, fdinfoPath := range fdinfoPaths {
		fdinfoFile, err := os.Open(fdinfoPath)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to open %s: %w", fdinfoPath, err)
		}

		img, decodeErr := crit.New(fdinfoFile, nil, "", false, false).Decode(&fdinfo.FdinfoEntry{})
		closeErr := fdinfoFile.Close()
		if decodeErr != nil {
			return nil, nil, fmt.Errorf("failed to decode %s: %w", fdinfoPath, decodeErr)
		}
		if closeErr != nil {
			return nil, nil, fmt.Errorf("failed to close %s: %w", fdinfoPath, closeErr)
		}

		for _, entry := range img.Entries {
			fdEntry, ok := entry.Message.(*fdinfo.FdinfoEntry)
			if !ok {
				continue
			}

			resource := resourcesByFileID[fdEntry.GetId()]
			if resource == "" {
				continue
			}

			switch fdEntry.GetFd() {
			case 1:
				stdoutSet[resource] = struct{}{}
			case 2:
				stderrSet[resource] = struct{}{}
			}
		}
	}

	return sortedSetValues(stdoutSet), sortedSetValues(stderrSet), nil
}

func loadInheritResourcesByFileID(checkpointPath string) (map[uint32]string, error) {
	filesPath := filepath.Join(checkpointPath, "files.img")
	filesImage, err := os.Open(filesPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open %s: %w", filesPath, err)
	}

	img, decodeErr := crit.New(filesImage, nil, "", false, false).Decode(&fdinfo.FileEntry{})
	closeErr := filesImage.Close()
	if decodeErr != nil {
		return nil, fmt.Errorf("failed to decode %s: %w", filesPath, decodeErr)
	}
	if closeErr != nil {
		return nil, fmt.Errorf("failed to close %s: %w", filesPath, closeErr)
	}

	resources := make(map[uint32]string, len(img.Entries))
	for _, entry := range img.Entries {
		fileEntry, ok := entry.Message.(*fdinfo.FileEntry)
		if !ok {
			continue
		}

		resource := fileEntryInheritResource(fileEntry)
		if resource == "" {
			continue
		}
		resources[fileEntry.GetId()] = resource
	}

	return resources, nil
}

func fileEntryInheritResource(fileEntry *fdinfo.FileEntry) string {
	if fileEntry == nil {
		return ""
	}
	if pipeEntry := fileEntry.GetPipe(); pipeEntry != nil {
		// CRIU inherit-fd keys for pipe endpoints use pipe:[inode] syntax.
		return fmt.Sprintf("pipe:[%d]", pipeEntry.GetPipeId())
	}
	if socketEntry := fileEntry.GetUsk(); socketEntry != nil {
		return fmt.Sprintf("socket[%d]", socketEntry.GetIno())
	}
	if regEntry := fileEntry.GetReg(); regEntry != nil && regEntry.GetName() != "" {
		return regEntry.GetName()
	}
	return ""
}

func sortedSetValues(values map[string]struct{}) []string {
	if len(values) == 0 {
		return nil
	}
	result := make([]string, 0, len(values))
	for value := range values {
		result = append(result, value)
	}
	sort.Strings(result)
	return result
}

func closeFiles(files []*os.File) {
	for _, file := range files {
		if file == nil {
			continue
		}
		_ = file.Close()
	}
}

func truncateInts(values []int, limit int) []int {
	if len(values) <= limit {
		return append([]int(nil), values...)
	}
	return append([]int(nil), values[:limit]...)
}

// generateExtMountMaps builds CRIU ext-mount maps by replaying the dump-time plan.
func generateExtMountMaps(manifest *checkpoint.CheckpointManifest) ([]*criurpc.ExtMountMap, error) {
	if len(manifest.CRIUDump.ExtMnt) == 0 {
		return nil, fmt.Errorf("checkpoint manifest is missing criuDump.extMnt")
	}

	maps := []*criurpc.ExtMountMap{{
		Key: proto.String("/"),
		Val: proto.String("."),
	}}
	added := map[string]struct{}{"/": {}}

	for _, mount := range manifest.CRIUDump.ExtMnt {
		if mount.Key == "" || mount.Key == "/" {
			continue
		}
		if _, exists := added[mount.Key]; exists {
			continue
		}
		val := mount.Val
		if val == "" {
			val = mount.Key
		}
		maps = append(maps, &criurpc.ExtMountMap{
			Key: proto.String(mount.Key),
			Val: proto.String(val),
		})
		added[mount.Key] = struct{}{}
	}

	return maps, nil
}

// buildRestoreOptions creates CRIU options for restore from the checkpoint manifest
// and restore-specific options passed via CLI flags from the DaemonSet agent.
func buildRestoreOptions(manifest *checkpoint.CheckpointManifest, imageDirFD, workDirFD int32, extMounts []*criurpc.ExtMountMap, restoreOpts restoreOptions) *criurpc.CriuOpts {
	settings := manifest.CRIUDump.CRIU

	var cgMode criurpc.CriuCgMode
	switch settings.ManageCgroupsMode {
	case "soft":
		cgMode = criurpc.CriuCgMode_SOFT
	case "full":
		cgMode = criurpc.CriuCgMode_FULL
	case "strict":
		cgMode = criurpc.CriuCgMode_STRICT
	default:
		cgMode = criurpc.CriuCgMode_IGNORE
	}

	criuOpts := &criurpc.CriuOpts{
		ImagesDirFd: proto.Int32(imageDirFD),
		LogLevel:    proto.Int32(settings.LogLevel),
		LogFile:     proto.String(restoreLogFile),
		Root:        proto.String("/"),

		// Restore-specific options (from ConfigMap via CLI flags)
		RstSibling:      proto.Bool(restoreOpts.RstSibling),
		MntnsCompatMode: proto.Bool(restoreOpts.MntnsCompatMode),
		EvasiveDevices:  proto.Bool(restoreOpts.EvasiveDevices),
		ForceIrmap:      proto.Bool(restoreOpts.ForceIrmap),

		// Options from saved checkpoint
		ShellJob:          proto.Bool(settings.ShellJob),
		TcpClose:          proto.Bool(settings.TcpClose),
		FileLocks:         proto.Bool(settings.FileLocks),
		ExtUnixSk:         proto.Bool(settings.ExtUnixSk),
		LinkRemap:         proto.Bool(settings.LinkRemap),
		ManageCgroups:     proto.Bool(true),
		ManageCgroupsMode: &cgMode,

		// External mounts
		ExtMnt: extMounts,
	}

	if workDirFD >= 0 {
		criuOpts.WorkDirFd = proto.Int32(workDirFD)
	}
	return criuOpts
}

// remountProcSys remounts /proc/sys as rw (true) or ro (false).
func remountProcSys(rw bool, log logr.Logger) error {
	flags := uintptr(syscall.MS_REMOUNT | syscall.MS_BIND)
	if !rw {
		flags |= syscall.MS_RDONLY
	}
	mode := "ro"
	if rw {
		mode = "rw"
	}
	if err := syscall.Mount("", "/proc/sys", "", flags, ""); err != nil {
		return fmt.Errorf("failed to remount /proc/sys %s: %w", mode, err)
	}
	log.V(1).Info("Remounted /proc/sys", "mode", mode)
	return nil
}

// remountCgroupFS remounts /sys/fs/cgroup as rw (true) or ro (false).
func remountCgroupFS(rw bool, log logr.Logger) error {
	flags := uintptr(syscall.MS_REMOUNT)
	if !rw {
		flags |= syscall.MS_RDONLY
	}
	mode := "ro"
	if rw {
		mode = "rw"
	}
	if err := syscall.Mount("", "/sys/fs/cgroup", "", flags, ""); err != nil {
		return fmt.Errorf("failed to remount /sys/fs/cgroup %s: %w", mode, err)
	}
	log.V(1).Info("Remounted /sys/fs/cgroup", "mode", mode)
	return nil
}

// logCRIUErrors reads and logs the CRIU restore log file.
func logCRIUErrors(checkpointPath, workDir string, log logr.Logger) {
	candidates := make([]string, 0, 2)
	if workDir != "" {
		candidates = append(candidates, filepath.Join(workDir, restoreLogFile))
	}
	candidates = append(candidates, filepath.Join(checkpointPath, restoreLogFile))

	for _, logPath := range candidates {
		data, err := os.ReadFile(logPath)
		if err != nil {
			continue
		}
		log.Info("=== CRIU RESTORE LOG (FULL) ===", "log_path", logPath, "log_bytes", len(data))
		log.Info("=== CRIU RESTORE LOG ===")
		for _, line := range strings.Split(string(data), "\n") {
			if line != "" {
				log.Info(line)
			}
		}
		log.Info("=== END CRIU RESTORE LOG ===")
		return
	}

	log.Info("Failed to read CRIU restore log from candidate paths", "checked_paths", candidates)
}

func getCheckpointCUDAPIDs(manifest *checkpoint.CheckpointManifest) []int {
	if manifest.ExternalRestore == nil || manifest.ExternalRestore.CUDA == nil {
		return nil
	}
	return manifest.ExternalRestore.CUDA.PIDs
}

func getCheckpointSourceGPUUUIDs(manifest *checkpoint.CheckpointManifest) []string {
	if manifest.ExternalRestore == nil || manifest.ExternalRestore.CUDA == nil {
		return nil
	}
	return manifest.ExternalRestore.CUDA.SourceGPUUUIDs
}

func extMountMapSample(extMounts []*criurpc.ExtMountMap, limit int) []string {
	if len(extMounts) == 0 {
		return nil
	}
	if limit <= 0 {
		limit = 1
	}
	size := len(extMounts)
	if size > limit {
		size = limit
	}
	sample := make([]string, 0, size)
	for i := 0; i < size; i++ {
		key := ""
		val := ""
		if extMounts[i].Key != nil {
			key = extMounts[i].GetKey()
		}
		if extMounts[i].Val != nil {
			val = extMounts[i].GetVal()
		}
		sample = append(sample, fmt.Sprintf("%s=>%s", key, val))
	}
	return sample
}

// restoreNotify captures the restored PID from CRIU callbacks.
type restoreNotify struct {
	criu.NoNotify
	restoredPID int32
	log         logr.Logger
}

func (n *restoreNotify) PreRestore() error {
	n.log.V(1).Info("CRIU pre-restore")
	return nil
}

func (n *restoreNotify) PostRestore(pid int32) error {
	n.restoredPID = pid
	n.log.Info("CRIU post-restore: process restored", "pid", pid)
	return nil
}
