// ns-restore-runner is a self-contained binary that performs CRIU restore inside container
// namespaces. It is invoked by the DaemonSet via:
//
//	nsenter -t <PID> -m -n -p -i -- /usr/local/bin/ns-restore-runner --checkpoint-path <path>
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

	criulib "github.com/checkpoint-restore/go-criu/v8"
	"github.com/checkpoint-restore/go-criu/v8/crit"
	"github.com/checkpoint-restore/go-criu/v8/crit/images/fdinfo"
	criurpc "github.com/checkpoint-restore/go-criu/v8/rpc"
	"google.golang.org/protobuf/proto"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/config"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/criu"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/cuda"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/manifest"
)

const (
	netNsPath      = "/proc/1/ns/net"
	restoreLogFile = "restore.log"
	procStdoutPath = "/proc/1/fd/1"
	procStderrPath = "/proc/1/fd/2"
	criuBinaryPath = "/usr/local/sbin/criu"
)

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

	log := rootLog.WithName("ns-restore-runner")

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

type restoreOptions struct {
	RstSibling      bool
	MntnsCompatMode bool
	EvasiveDevices  bool
	ForceIrmap      bool
}

func run(checkpointPath, workDir, cudaDeviceMap string, opts restoreOptions, log logr.Logger) error {
	restoreStart := time.Now()
	log.Info("Starting ns-restore-runner restore workflow",
		"checkpoint_path", checkpointPath,
		"work_dir", workDir,
		"has_cuda_map", cudaDeviceMap != "",
	)

	m, err := manifest.Read(checkpointPath)
	if err != nil {
		return fmt.Errorf("failed to read manifest: %w", err)
	}
	log.Info("Loaded checkpoint manifest",
		"ext_mounts", len(m.CRIUDump.ExtMnt),
		"criu_log_level", m.CRIUDump.CRIU.LogLevel,
		"manage_cgroups_mode", m.CRIUDump.CRIU.ManageCgroupsMode,
		"checkpoint_has_cuda", m.ExternalRestore != nil && m.ExternalRestore.CUDA != nil,
	)

	// Remount /proc/sys rw — CRIU needs to write sysctl values
	log.Info("Remounting /proc/sys read-write")
	if err := remountProcSys(true, log); err != nil {
		log.Error(err, "Failed to remount /proc/sys rw (restore may still work)")
	}
	defer remountProcSys(false, log) //nolint:errcheck

	manageCgroupsMode := strings.ToLower(strings.TrimSpace(m.CRIUDump.CRIU.ManageCgroupsMode))
	if manageCgroupsMode != "ignore" {
		log.Info("Remounting /sys/fs/cgroup read-write")
		if err := remountCgroupFS(true, log); err != nil {
			log.Error(err, "Failed to remount /sys/fs/cgroup rw")
		}
		defer remountCgroupFS(false, log) //nolint:errcheck
	}

	imageDir, imageDirFD, err := criu.OpenPathForCRIU(checkpointPath)
	if err != nil {
		return fmt.Errorf("failed to open image directory: %w", err)
	}
	defer imageDir.Close()

	var workDirFile *os.File
	var workDirFD int32 = -1
	if workDir != "" {
		if err := os.MkdirAll(workDir, 0755); err != nil {
			log.Error(err, "Failed to create work directory")
		} else {
			f, fd, err := criu.OpenPathForCRIU(workDir)
			if err == nil {
				workDirFile = f
				workDirFD = fd
				defer workDirFile.Close()
			} else {
				log.Error(err, "Failed to open CRIU work directory")
			}
		}
	}

	extMounts, err := generateExtMountMaps(m)
	if err != nil {
		return fmt.Errorf("failed to generate ext mount maps: %w", err)
	}
	log.Info("Generated external mount map set", "ext_mount_count", len(extMounts))

	criuOpts := buildRestoreOptions(m, imageDirFD, workDirFD, extMounts, opts)

	// Reuse criu.conf from checkpoint if it exists
	criuConfPath := filepath.Join(checkpointPath, config.CheckpointCRIUConfFilename)
	if _, err := os.Stat(criuConfPath); err == nil {
		criuOpts.ConfigFile = proto.String(criuConfPath)
		log.Info("Using checkpointed CRIU config file", "config_file", criuConfPath)
	}

	c := criulib.MakeCriu()
	if _, err := os.Stat(criuBinaryPath); err != nil {
		return fmt.Errorf("criu binary not found at %s: %w", criuBinaryPath, err)
	}
	c.SetCriuPath(criuBinaryPath)

	netNsFile, err := os.Open(netNsPath)
	if err != nil {
		return fmt.Errorf("failed to open net NS at %s: %w", netNsPath, err)
	}
	defer netNsFile.Close()
	c.AddInheritFd("extNetNs", netNsFile)

	inheritedStdioFiles, err := registerStdioInheritFDs(c, checkpointPath, log)
	if err != nil {
		log.Error(err, "Failed to configure inherited stdio resources")
	} else {
		defer closeFiles(inheritedStdioFiles)
	}

	notify := &restoreNotify{log: log}
	log.Info("Executing go-criu Restore call")
	if err := c.Restore(criuOpts, notify); err != nil {
		log.Error(err, "go-criu Restore returned error")
		logCRIUErrors(checkpointPath, workDir, log)
		return fmt.Errorf("CRIU restore failed: %w", err)
	}

	log.Info("CRIU restore completed", "pid", notify.restoredPID, "duration", time.Since(restoreStart))

	if err := cuda.RestoreInNamespace(context.Background(), m, int(notify.restoredPID), cudaDeviceMap, log); err != nil {
		log.Error(err, "CUDA restore sequence failed")
		return err
	}

	fmt.Printf("RESTORED_PID=%d\n", notify.restoredPID)
	if restoredHostPID, err := readRestoredHostPID(int(notify.restoredPID)); err == nil {
		fmt.Printf("RESTORED_HOST_PID=%d\n", restoredHostPID)
	} else {
		log.Error(err, "Failed to resolve restored host PID from NSpid")
	}
	log.Info("ns-restore-runner completed successfully", "restored_pid", notify.restoredPID)
	return nil
}

// --- CRIU restore options ---

func buildRestoreOptions(m *manifest.CheckpointManifest, imageDirFD, workDirFD int32, extMounts []*criurpc.ExtMountMap, restoreOpts restoreOptions) *criurpc.CriuOpts {
	settings := m.CRIUDump.CRIU

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

		ExtMnt: extMounts,
	}

	if workDirFD >= 0 {
		criuOpts.WorkDirFd = proto.Int32(workDirFD)
	}
	return criuOpts
}

func generateExtMountMaps(m *manifest.CheckpointManifest) ([]*criurpc.ExtMountMap, error) {
	if len(m.CRIUDump.ExtMnt) == 0 {
		return nil, fmt.Errorf("checkpoint manifest is missing criuDump.extMnt")
	}

	maps := []*criurpc.ExtMountMap{{
		Key: proto.String("/"),
		Val: proto.String("."),
	}}
	added := map[string]struct{}{"/": {}}

	for _, mount := range m.CRIUDump.ExtMnt {
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

// --- stdio fd inheritance ---

func registerStdioInheritFDs(c *criulib.Criu, checkpointPath string, log logr.Logger) ([]*os.File, error) {
	stdoutResources, stderrResources, err := discoverStdioInheritResources(checkpointPath)
	if err != nil {
		return nil, err
	}
	if len(stdoutResources) == 0 && len(stderrResources) == 0 {
		log.Info("No stdio resources found for inherit-fd wiring")
		return nil, nil
	}

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
		if file != nil {
			file.Close()
		}
	}
}

// --- utility helpers ---

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
		log.Info("=== CRIU RESTORE LOG ===")
		for _, line := range strings.Split(string(data), "\n") {
			if line != "" {
				log.Info(line)
			}
		}
		log.Info("=== END CRIU RESTORE LOG ===")
		return
	}
}

type restoreNotify struct {
	criulib.NoNotify
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
