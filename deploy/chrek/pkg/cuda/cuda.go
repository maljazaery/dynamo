// Package cuda provides CUDA checkpoint and restore operations using
// the external cuda-checkpoint binary. Both DaemonSet-side (cgroup-based)
// and ns-restore-runner-side (process-tree-based) operations are included.
package cuda

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/go-logr/logr"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/manifest"
)

const (
	cudaCheckpointBinary = "/usr/local/sbin/cuda-checkpoint"
	podResourcesSocket   = "/var/lib/kubelet/pod-resources/kubelet.sock"
	nvidiaGPUResource    = "nvidia.com/gpu"
	hostCgroupPath       = "/sys/fs/cgroup"
	cudaDiscoverTick     = 1 * time.Second
	podGPUDiscoverTick   = 1 * time.Second
)

// CheckpointRequest holds per-request identifiers needed for CUDA checkpoint.
type CheckpointRequest struct {
	PodName       string
	PodNamespace  string
	ContainerName string
}

// PrepareCheckpoint locks and checkpoints CUDA state for all CUDA PIDs in the container's cgroup.
// Updates the manifest with CUDA metadata for restore.
func PrepareCheckpoint(ctx context.Context, req CheckpointRequest, sourcePID int, m *manifest.CheckpointManifest, log logr.Logger) error {
	cgroupPath, err := getContainerCgroupPath(sourcePID)
	if err != nil {
		return fmt.Errorf("failed to get source cgroup path: %w", err)
	}

	cgroupPIDs, err := getCgroupPIDs(cgroupPath)
	if err != nil {
		return fmt.Errorf("failed to list source cgroup PIDs: %w", err)
	}

	cudaPIDs := FilterCUDAProcesses(ctx, cgroupPIDs, log)
	if len(cudaPIDs) == 0 {
		log.V(1).Info("No CUDA PIDs detected in source cgroup, skipping external CUDA preparation")
		return nil
	}

	locked := make([]int, 0, len(cudaPIDs))
	for _, pid := range cudaPIDs {
		if err := Lock(ctx, pid, log); err != nil {
			UnlockProcesses(context.Background(), locked, log)
			return fmt.Errorf("cuda lock failed for PID %d: %w", pid, err)
		}
		locked = append(locked, pid)
	}

	for _, pid := range cudaPIDs {
		if err := Checkpoint(ctx, pid, log); err != nil {
			UnlockProcesses(context.Background(), locked, log)
			return fmt.Errorf("cuda checkpoint failed for PID %d: %w", pid, err)
		}
	}

	sourceGPUUUIDs, err := GetPodGPUUUIDsWithRetry(ctx, req.PodName, req.PodNamespace, req.ContainerName, log)
	if err != nil {
		return fmt.Errorf("failed to discover source GPU UUIDs: %w", err)
	}
	if len(sourceGPUUUIDs) == 0 {
		UnlockProcesses(context.Background(), locked, log)
		return fmt.Errorf("no source GPU UUIDs found for %s/%s container %s", req.PodNamespace, req.PodName, req.ContainerName)
	}
	m.CUDA = manifest.NewCUDAManifest(cudaPIDs, sourceGPUUUIDs)

	log.Info("Prepared external CUDA checkpoint metadata",
		"cuda_pids", len(cudaPIDs),
		"source_gpu_uuids", len(sourceGPUUUIDs),
	)
	return nil
}

// UnlockFromManifest unlocks CUDA processes that were locked during checkpoint.
func UnlockFromManifest(m *manifest.CheckpointManifest, log logr.Logger) {
	if m == nil || m.CUDA.IsEmpty() {
		return
	}
	UnlockProcesses(context.Background(), m.CUDA.PIDs, log)
}

// Restore performs CUDA restore from inside the container namespace
// (called by ns-restore-runner). Uses process tree walking instead of cgroup discovery.
func Restore(ctx context.Context, m *manifest.CheckpointManifest, restoredPID int, deviceMap string, log logr.Logger) error {
	if m.CUDA.IsEmpty() {
		log.Info("Checkpoint does not contain CUDA metadata, skipping cuda-checkpoint restore")
		return nil
	}
	if deviceMap == "" {
		return fmt.Errorf("missing --cuda-device-map for checkpoint with CUDA state")
	}
	if _, err := os.Stat(cudaCheckpointBinary); err != nil {
		return fmt.Errorf("cuda-checkpoint not found at %s: %w", cudaCheckpointBinary, err)
	}
	log.Info("Starting CUDA restore sequence",
		"restored_pid", restoredPID,
		"checkpoint_cuda_pids", len(m.CUDA.PIDs),
		"device_map", deviceMap,
	)

	attempt := 0
	for {
		attempt++
		candidates := ProcessTreePIDs(restoredPID)
		cudaPIDs := FilterCUDAProcesses(ctx, candidates, log)

		log.Info("CUDA restore PID discovery attempt",
			"attempt", attempt,
			"restored_pid", restoredPID,
			"candidates", len(candidates),
			"cuda_pids", len(cudaPIDs),
		)

		if len(cudaPIDs) > 0 {
			log.Info("Running cuda-checkpoint unlock for candidate CUDA PIDs", "cuda_pids", len(cudaPIDs))
			UnlockProcesses(ctx, cudaPIDs, log)
			if err := RestoreProcesses(ctx, cudaPIDs, deviceMap, log); err != nil {
				return err
			}
			log.Info("CUDA restore completed", "cuda_pids", len(cudaPIDs), "device_map", deviceMap)
			return nil
		}

		select {
		case <-ctx.Done():
			return fmt.Errorf("checkpoint captured %d CUDA PIDs but none found in restored process tree rooted at PID %d: %w", len(m.CUDA.PIDs), restoredPID, ctx.Err())
		case <-time.After(cudaDiscoverTick):
		}
	}
}
