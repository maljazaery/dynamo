package cuda

import (
	"context"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/go-logr/logr"
)

const (
	actionLock       = "lock"
	actionCheckpoint = "checkpoint"
	actionRestore    = "restore"
	actionUnlock     = "unlock"
)

// CUDA checkpoint shims

// Lock invokes cuda-checkpoint lock for one PID.
func Lock(ctx context.Context, pid int, log logr.Logger) error {
	return runAction(ctx, pid, actionLock, "", log)
}

// Checkpoint invokes cuda-checkpoint checkpoint for one PID.
func Checkpoint(ctx context.Context, pid int, log logr.Logger) error {
	return runAction(ctx, pid, actionCheckpoint, "", log)
}

// RestoreProcess invokes cuda-checkpoint restore for one PID and logs timing/output.
func RestoreProcess(ctx context.Context, pid int, deviceMap string, log logr.Logger) error {
	return runAction(ctx, pid, actionRestore, deviceMap, log)
}

// Unlock invokes cuda-checkpoint unlock for one PID and logs timing/output.
func Unlock(ctx context.Context, pid int, log logr.Logger) error {
	return runAction(ctx, pid, actionUnlock, "", log)
}

func runAction(ctx context.Context, pid int, action, deviceMap string, log logr.Logger) error {
	args := []string{"--action", action, "--pid", strconv.Itoa(pid)}
	if action == actionRestore && deviceMap != "" {
		args = append(args, "--device-map", deviceMap)
	}
	cmd := exec.CommandContext(ctx, cudaCheckpointBinary, args...)
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

// Utility functions

// UnlockProcesses unlocks a list of CUDA PIDs.
func UnlockProcesses(ctx context.Context, pids []int, log logr.Logger) {
	for _, pid := range pids {
		if err := Unlock(ctx, pid, log); err != nil {
			log.Error(err, "Failed to unlock CUDA process", "pid", pid)
		}
	}
}

// RestoreProcesses restores a list of CUDA PIDs.
func RestoreProcesses(ctx context.Context, pids []int, deviceMap string, log logr.Logger) error {
	for _, pid := range pids {
		log.Info("Running cuda-checkpoint restore", "pid", pid, "device_map", deviceMap)
		if err := RestoreProcess(ctx, pid, deviceMap, log); err != nil {
			return fmt.Errorf("cuda restore failed for PID %d: %w", pid, err)
		}
	}
	return nil
}

// FilterCUDAProcesses returns the subset of candidate PIDs that report CUDA state.
func FilterCUDAProcesses(ctx context.Context, allPIDs []int, log logr.Logger) []int {
	cudaPIDs := make([]int, 0, len(allPIDs))
	for _, pid := range allPIDs {
		if pid <= 0 {
			continue
		}

		cmd := exec.CommandContext(ctx, cudaCheckpointBinary, "--get-state", "--pid", strconv.Itoa(pid))
		err := cmd.Run()
		if err != nil {
			if ctx.Err() != nil {
				break
			}
			log.V(1).Info("CUDA state probe failed", "pid", pid, "error", err)
			continue
		}
		cudaPIDs = append(cudaPIDs, pid)
	}
	return cudaPIDs
}

// BuildDeviceMap creates a cuda-checkpoint --device-map value from source and target GPU UUID lists.
func BuildDeviceMap(sourceUUIDs, targetUUIDs []string) (string, error) {
	if len(sourceUUIDs) != len(targetUUIDs) {
		return "", fmt.Errorf("GPU count mismatch: source has %d, target has %d", len(sourceUUIDs), len(targetUUIDs))
	}
	if len(sourceUUIDs) == 0 {
		return "", fmt.Errorf("GPU UUID list is empty")
	}
	pairs := make([]string, len(sourceUUIDs))
	for i := range sourceUUIDs {
		pairs[i] = sourceUUIDs[i] + "=" + targetUUIDs[i]
	}
	return strings.Join(pairs, ","), nil
}
