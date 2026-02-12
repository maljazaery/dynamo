// Package cuda provides CUDA checkpoint and restore operations using
// the external cuda-checkpoint binary. Both DaemonSet-side (cgroup-based)
// and ns-restore-runner-side (process-tree-based) operations are included.
package cuda

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/go-logr/logr"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	podresourcesv1 "k8s.io/kubelet/pkg/apis/podresources/v1"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/manifest"
)

type checkpointAction string

const (
	cudaCheckpointBinary = "/usr/local/sbin/cuda-checkpoint"
	podResourcesSocket   = "/var/lib/kubelet/pod-resources/kubelet.sock"
	nvidiaGPUResource    = "nvidia.com/gpu"
	hostCgroupPath       = "/sys/fs/cgroup"
	cudaDiscoverTick     = 1 * time.Second
	podGPUDiscoverTick   = 1 * time.Second

	actionLock       checkpointAction = "lock"
	actionCheckpoint checkpointAction = "checkpoint"
	actionRestore    checkpointAction = "restore"
	actionUnlock     checkpointAction = "unlock"
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

	cudaPIDs := findCUDAPIDs(cgroupPIDs)
	if len(cudaPIDs) == 0 {
		log.V(1).Info("No CUDA PIDs detected in source cgroup, skipping external CUDA preparation")
		return nil
	}

	locked := make([]int, 0, len(cudaPIDs))
	for _, pid := range cudaPIDs {
		if err := runCudaCheckpoint(ctx, pid, actionLock, ""); err != nil {
			UnlockProcesses(context.Background(), locked, log)
			return fmt.Errorf("cuda lock failed for PID %d: %w", pid, err)
		}
		locked = append(locked, pid)
	}

	for _, pid := range cudaPIDs {
		if err := runCudaCheckpoint(ctx, pid, actionCheckpoint, ""); err != nil {
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

	m.CUDARestore = &manifest.CUDARestoreManifest{
		PIDs:           cudaPIDs,
		SourceGPUUUIDs: sourceGPUUUIDs,
		Locked:         true,
	}

	log.Info("Prepared external CUDA checkpoint metadata",
		"cuda_pids", len(cudaPIDs),
		"source_gpu_uuids", len(sourceGPUUUIDs),
	)
	return nil
}

// UnlockFromManifest unlocks CUDA processes that were locked during checkpoint.
func UnlockFromManifest(m *manifest.CheckpointManifest, log logr.Logger) {
	if m == nil || m.CUDARestore == nil {
		return
	}
	if !m.CUDARestore.Locked {
		return
	}
	UnlockProcesses(context.Background(), m.CUDARestore.PIDs, log)
}

// UnlockProcesses unlocks a list of CUDA PIDs.
func UnlockProcesses(ctx context.Context, pids []int, log logr.Logger) {
	for _, pid := range pids {
		if err := runCudaCheckpoint(ctx, pid, actionUnlock, ""); err != nil {
			log.Error(err, "Failed to unlock CUDA process", "pid", pid)
		}
	}
}

// Restore performs CUDA restore from inside the container namespace
// (called by ns-restore-runner). Uses process tree walking instead of cgroup discovery.
func Restore(ctx context.Context, m *manifest.CheckpointManifest, restoredPID int, deviceMap string, log logr.Logger) error {
	if m.CUDARestore == nil || len(m.CUDARestore.PIDs) == 0 {
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
		"checkpoint_cuda_pids", len(m.CUDARestore.PIDs),
		"device_map", deviceMap,
	)

	attempt := 0
	for {
		attempt++
		candidates := ProcessTreePIDs(restoredPID)
		cudaPIDs := make([]int, 0, len(candidates))
		for _, pid := range candidates {
			ok, err := IsCUDAProcess(ctx, pid)
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
		)

		if len(cudaPIDs) > 0 {
			for _, pid := range cudaPIDs {
				log.Info("Running cuda-checkpoint restore", "pid", pid, "device_map", deviceMap)
				if err := RunAction(ctx, pid, string(actionRestore), deviceMap, log); err != nil {
					return fmt.Errorf("cuda restore failed for PID %d: %w", pid, err)
				}
				log.Info("Running cuda-checkpoint unlock", "pid", pid)
				if err := RunAction(ctx, pid, string(actionUnlock), "", log); err != nil {
					return fmt.Errorf("cuda unlock failed for PID %d: %w", pid, err)
				}
			}
			log.Info("CUDA restore completed", "cuda_pids", len(cudaPIDs), "device_map", deviceMap)
			return nil
		}

		select {
		case <-ctx.Done():
			return fmt.Errorf("checkpoint captured %d CUDA PIDs but none found in restored process tree rooted at PID %d: %w", len(m.CUDARestore.PIDs), restoredPID, ctx.Err())
		case <-time.After(cudaDiscoverTick):
		}
	}
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

// IsCUDAProcess checks if a PID has CUDA state via cuda-checkpoint --get-state.
func IsCUDAProcess(ctx context.Context, pid int) (bool, error) {
	if pid <= 0 {
		return false, nil
	}

	cmd := exec.CommandContext(ctx, cudaCheckpointBinary, "--get-state", "--pid", strconv.Itoa(pid))
	if err := cmd.Run(); err != nil {
		if ctx.Err() != nil {
			return false, ctx.Err()
		}
		return false, nil
	}
	return true, nil
}

// RunAction runs a cuda-checkpoint action (restore/unlock) with logging.
// Used by ns-restore-runner which needs per-action log output.
func RunAction(ctx context.Context, pid int, action string, deviceMap string, log logr.Logger) error {
	args := []string{"--action", action, "--pid", strconv.Itoa(pid)}
	if action == string(actionRestore) && deviceMap != "" {
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

// ProcessTreePIDs walks the process tree rooted at rootPID and returns all PIDs.
// Used by ns-restore-runner for in-namespace CUDA PID discovery.
func ProcessTreePIDs(rootPID int) []int {
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

// GetPodGPUUUIDsWithRetry resolves GPU UUIDs for a pod/container from kubelet pod-resources API.
func GetPodGPUUUIDsWithRetry(ctx context.Context, podName, podNamespace, containerName string, log logr.Logger) ([]string, error) {
	ticker := time.NewTicker(podGPUDiscoverTick)
	defer ticker.Stop()

	attempt := 0
	var lastErr error
	for {
		attempt++
		uuids, err := getPodGPUUUIDs(ctx, podName, podNamespace, containerName)
		if err == nil && len(uuids) > 0 {
			return uuids, nil
		}
		if err != nil {
			lastErr = err
		}

		uuidCount := 0
		if uuids != nil {
			uuidCount = len(uuids)
		}
		log.V(1).Info("Waiting for pod GPU UUIDs in pod-resources",
			"attempt", attempt,
			"pod", podName,
			"namespace", podNamespace,
			"container", containerName,
			"uuid_count", uuidCount,
		)

		select {
		case <-ctx.Done():
			if lastErr != nil {
				return nil, lastErr
			}
			return nil, nil
		case <-ticker.C:
		}
	}
}

// --- internal helpers ---

func runCudaCheckpoint(ctx context.Context, pid int, action checkpointAction, deviceMap string) error {
	args := []string{"--action", string(action), "--pid", strconv.Itoa(pid)}
	if action == actionRestore && deviceMap != "" {
		args = append(args, "--device-map", deviceMap)
	}

	cmd := exec.CommandContext(ctx, cudaCheckpointBinary, args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("cuda-checkpoint %v failed for pid %d: %w (output: %s)", args, pid, err, strings.TrimSpace(string(output)))
	}
	return nil
}

func getPodGPUUUIDs(ctx context.Context, podName, podNamespace, containerName string) ([]string, error) {
	if podName == "" || podNamespace == "" {
		return nil, nil
	}

	conn, err := grpc.DialContext(
		ctx,
		"unix://"+podResourcesSocket,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
	)
	if err != nil {
		return nil, err
	}
	defer conn.Close()

	client := podresourcesv1.NewPodResourcesListerClient(conn)
	resp, err := client.List(ctx, &podresourcesv1.ListPodResourcesRequest{})
	if err != nil {
		return nil, err
	}

	for _, pod := range resp.GetPodResources() {
		if pod.GetName() != podName || pod.GetNamespace() != podNamespace {
			continue
		}
		for _, container := range pod.GetContainers() {
			if containerName != "" && container.GetName() != containerName {
				continue
			}
			for _, device := range container.GetDevices() {
				if device.GetResourceName() == nvidiaGPUResource {
					return device.GetDeviceIds(), nil
				}
			}
		}
	}

	return nil, nil
}

func getContainerCgroupPath(pid int) (string, error) {
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/cgroup", pid))
	if err != nil {
		return "", err
	}
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		if strings.HasPrefix(line, "0::") {
			return strings.TrimPrefix(line, "0::"), nil
		}
	}
	return "", fmt.Errorf("no cgroup v2 path found for pid %d", pid)
}

func getCgroupPIDs(cgroupPath string) ([]int, error) {
	procsPath := filepath.Join(hostCgroupPath, cgroupPath, "cgroup.procs")
	data, err := os.ReadFile(procsPath)
	if err != nil {
		return nil, err
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	pids := make([]int, 0, len(lines))
	for _, line := range lines {
		if line == "" {
			continue
		}
		pid, err := strconv.Atoi(line)
		if err != nil {
			continue
		}
		pids = append(pids, pid)
	}
	return pids, nil
}

func findCUDAPIDs(pids []int) []int {
	cudaPIDs := make([]int, 0, len(pids))
	for _, pid := range pids {
		mapsPath := fmt.Sprintf("/proc/%d/maps", pid)
		data, err := os.ReadFile(mapsPath)
		if err != nil {
			continue
		}
		if strings.Contains(string(data), "libnvidia") {
			cudaPIDs = append(cudaPIDs, pid)
		}
	}
	return cudaPIDs
}
