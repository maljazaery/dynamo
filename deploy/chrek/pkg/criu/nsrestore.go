package criu

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"

	criulib "github.com/checkpoint-restore/go-criu/v8"
	criurpc "github.com/checkpoint-restore/go-criu/v8/rpc"
	"github.com/go-logr/logr"
	"google.golang.org/protobuf/proto"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/config"
	criuutil "github.com/ai-dynamo/dynamo/deploy/chrek/pkg/criu/util"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/cuda"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/manifest"
)

const (
	netNsPath      = "/proc/1/ns/net"
	restoreLogFile = "restore.log"
	procStdoutPath = "/proc/1/fd/1"
	procStderrPath = "/proc/1/fd/2"
)

// RestoreOptions holds configuration for an in-namespace CRIU restore.
type RestoreOptions struct {
	CheckpointPath string
	CUDADeviceMap  string
}

// RestoreInNamespace performs a CRIU restore from inside the target container's namespaces.
// It reads all CRIU settings from the checkpoint manifest.
func RestoreInNamespace(ctx context.Context, opts RestoreOptions, log logr.Logger) (int, int, error) {
	restoreStart := time.Now()
	log.Info("Starting nsrestore restore workflow",
		"checkpoint_path", opts.CheckpointPath,
		"has_cuda_map", opts.CUDADeviceMap != "",
	)

	m, err := manifest.Read(opts.CheckpointPath)
	if err != nil {
		return 0, 0, fmt.Errorf("failed to read manifest: %w", err)
	}
	settings := m.CRIUDump.CRIU
	log.Info("Loaded checkpoint manifest",
		"ext_mounts", len(m.CRIUDump.ExtMnt),
		"criu_log_level", settings.LogLevel,
		"manage_cgroups_mode", settings.ManageCgroupsMode,
		"checkpoint_has_cuda", !m.CUDA.IsEmpty(),
	)

	log.Info("Remounting /proc/sys read-write")
	if err := remountProcSys(true, log); err != nil {
		log.Error(err, "Failed to remount /proc/sys rw (restore may still work)")
	}
	defer remountProcSys(false, log) //nolint:errcheck

	manageCgroupsMode := strings.ToLower(strings.TrimSpace(settings.ManageCgroupsMode))
	if manageCgroupsMode != "ignore" {
		log.Info("Remounting /sys/fs/cgroup read-write")
		if err := remountCgroupFS(true, log); err != nil {
			log.Error(err, "Failed to remount /sys/fs/cgroup rw")
		}
		defer remountCgroupFS(false, log) //nolint:errcheck
	}

	imageDir, imageDirFD, err := criuutil.OpenPathForCRIU(opts.CheckpointPath)
	if err != nil {
		return 0, 0, fmt.Errorf("failed to open image directory: %w", err)
	}
	defer imageDir.Close()

	var workDirFile *os.File
	var workDirFD int32 = -1
	if settings.WorkDir != "" {
		if err := os.MkdirAll(settings.WorkDir, 0755); err != nil {
			log.Error(err, "Failed to create work directory")
		} else {
			f, fd, err := criuutil.OpenPathForCRIU(settings.WorkDir)
			if err == nil {
				workDirFile = f
				workDirFD = fd
				defer workDirFile.Close()
			} else {
				log.Error(err, "Failed to open CRIU work directory")
			}
		}
	}

	extMounts, err := generateRestoreExtMountMaps(m)
	if err != nil {
		return 0, 0, fmt.Errorf("failed to generate ext mount maps: %w", err)
	}
	log.Info("Generated external mount map set", "ext_mount_count", len(extMounts))

	criuOpts := buildRestoreOpts(m, imageDirFD, workDirFD, extMounts)

	criuConfPath := filepath.Join(opts.CheckpointPath, config.CheckpointCRIUConfFilename)
	if _, err := os.Stat(criuConfPath); err == nil {
		criuOpts.ConfigFile = proto.String(criuConfPath)
		log.Info("Using checkpointed CRIU config file", "config_file", criuConfPath)
	}

	c := criulib.MakeCriu()
	if _, err := os.Stat(settings.BinaryPath); err != nil {
		return 0, 0, fmt.Errorf("criu binary not found at %s: %w", settings.BinaryPath, err)
	}
	c.SetCriuPath(settings.BinaryPath)

	netNsFile, err := os.Open(netNsPath)
	if err != nil {
		return 0, 0, fmt.Errorf("failed to open net NS at %s: %w", netNsPath, err)
	}
	defer netNsFile.Close()
	c.AddInheritFd("extNetNs", netNsFile)

	inheritedStdioFiles, err := registerStdioInheritFDs(c, opts.CheckpointPath, log)
	if err != nil {
		log.Error(err, "Failed to configure inherited stdio resources")
	} else {
		defer closeFiles(inheritedStdioFiles)
	}

	notify := &restoreNotify{log: log}
	log.Info("Executing go-criu Restore call")
	if err := c.Restore(criuOpts, notify); err != nil {
		log.Error(err, "go-criu Restore returned error")
		criuutil.LogRestoreErrors(opts.CheckpointPath, settings.WorkDir, log)
		return 0, 0, fmt.Errorf("CRIU restore failed: %w", err)
	}
	log.Info("CRIU restore completed", "pid", notify.restoredPID, "duration", time.Since(restoreStart))

	if err := cuda.Restore(ctx, m, int(notify.restoredPID), opts.CUDADeviceMap, log); err != nil {
		log.Error(err, "CUDA restore sequence failed")
		return 0, 0, err
	}

	restoredHostPID, err := readRestoredHostPID(int(notify.restoredPID))
	if err != nil {
		log.Error(err, "Failed to resolve restored host PID from NSpid")
	}

	log.Info("nsrestore completed successfully", "restored_pid", notify.restoredPID)
	return int(notify.restoredPID), restoredHostPID, nil
}

func buildRestoreOpts(m *manifest.CheckpointManifest, imageDirFD, workDirFD int32, extMounts []*criurpc.ExtMountMap) *criurpc.CriuOpts {
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

		RstSibling:      proto.Bool(settings.RstSibling),
		MntnsCompatMode: proto.Bool(settings.MntnsCompatMode),
		EvasiveDevices:  proto.Bool(settings.EvasiveDevices),
		ForceIrmap:      proto.Bool(settings.ForceIrmap),

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

func generateRestoreExtMountMaps(m *manifest.CheckpointManifest) ([]*criurpc.ExtMountMap, error) {
	if len(m.CRIUDump.ExtMnt) == 0 {
		return nil, fmt.Errorf("checkpoint manifest is missing criuDump.extMnt")
	}

	maps := []*criurpc.ExtMountMap{{
		Key: proto.String("/"),
		Val: proto.String("."),
	}}
	added := map[string]struct{}{"/": {}}

	keys := make([]string, 0, len(m.CRIUDump.ExtMnt))
	for key := range m.CRIUDump.ExtMnt {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	for _, key := range keys {
		if key == "" || key == "/" {
			continue
		}
		if _, exists := added[key]; exists {
			continue
		}
		val := m.CRIUDump.ExtMnt[key]
		if val == "" {
			val = key
		}
		maps = append(maps, &criurpc.ExtMountMap{
			Key: proto.String(key),
			Val: proto.String(val),
		})
		added[key] = struct{}{}
	}

	return maps, nil
}

func registerStdioInheritFDs(c *criulib.Criu, checkpointPath string, log logr.Logger) ([]*os.File, error) {
	stdoutResources, stderrResources, err := DiscoverStdioInheritResources(checkpointPath)
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

func closeFiles(files []*os.File) {
	for _, file := range files {
		if file != nil {
			file.Close()
		}
	}
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
