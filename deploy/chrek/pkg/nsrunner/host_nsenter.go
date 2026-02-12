package nsrunner

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"

	"github.com/go-logr/logr"
	"golang.org/x/sys/unix"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/config"
)

type HostRestoreOptions struct {
	PlaceholderPID  int
	RunnerPath      string
	CheckpointPath  string
	WorkDir         string
	CUDADeviceMap   string
	RestoreSettings *config.CRIUSettings
}

func NSEnterCallFromHost(ctx context.Context, opts HostRestoreOptions, log logr.Logger) (int, int, error) {
	pidStr := strconv.Itoa(opts.PlaceholderPID)

	baseArgs := []string{
		"-t", pidStr,
		"-m", "-n", "-p", "-i", "-u",
		"--", opts.RunnerPath,
		"--checkpoint-path", opts.CheckpointPath,
	}

	if opts.WorkDir != "" {
		baseArgs = append(baseArgs, "--work-dir", opts.WorkDir)
	}
	if opts.CUDADeviceMap != "" {
		baseArgs = append(baseArgs, "--cuda-device-map", opts.CUDADeviceMap)
	}

	restoreFlags := []string{}
	if opts.RestoreSettings != nil {
		if opts.RestoreSettings.RstSibling {
			restoreFlags = append(restoreFlags, "--rst-sibling")
		}
		if opts.RestoreSettings.MntnsCompatMode {
			restoreFlags = append(restoreFlags, "--mntns-compat-mode")
		}
		if opts.RestoreSettings.EvasiveDevices {
			restoreFlags = append(restoreFlags, "--evasive-devices")
		}
		if opts.RestoreSettings.ForceIrmap {
			restoreFlags = append(restoreFlags, "--force-irmap")
		}
	}

	args := append(append([]string{}, baseArgs...), restoreFlags...)
	restoredPID, restoredHostPID, output, err := Run(ctx, args, log)
	if err != nil && len(restoreFlags) > 0 && IsUnsupportedFlagError(output, restoreFlags) {
		log.Info("Retrying restore without unsupported optional ns-restore-runner flags", "flags", strings.Join(restoreFlags, " "))
		restoredPID, restoredHostPID, output, err = Run(ctx, baseArgs, log)
	}
	if err != nil {
		return 0, 0, fmt.Errorf("nsenter + ns-restore-runner failed: %w\noutput: %s", err, output)
	}

	return restoredPID, restoredHostPID, nil
}

func EnsureDevNetTunInTargetRoot(targetRoot string, log logr.Logger) {
	tunPath := filepath.Join(targetRoot, "dev/net/tun")
	if _, statErr := os.Stat(tunPath); !os.IsNotExist(statErr) {
		return
	}
	if err := os.MkdirAll(filepath.Dir(tunPath), 0755); err != nil {
		log.Error(err, "Failed to create /dev/net dir in placeholder")
		return
	}
	if err := syscall.Mknod(tunPath, syscall.S_IFCHR|0666, int(unix.Mkdev(10, 200))); err != nil {
		log.Error(err, "Failed to create /dev/net/tun in placeholder")
		return
	}
	log.Info("Created /dev/net/tun in placeholder rootfs")
}
