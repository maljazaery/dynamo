// cuda_external.go implements external CUDA checkpoint + restore orchestration.
// It replaces CRIU CUDA plugin hooks by running cuda-checkpoint from the DaemonSet.
package checkpoint

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
)

type cudaCheckpointAction string

const (
	cudaCheckpointBinary = "/usr/local/sbin/cuda-checkpoint"
	podResourcesSocket  = "/var/lib/kubelet/pod-resources/kubelet.sock"
	nvidiaGPUResource   = "nvidia.com/gpu"
	hostCgroupPath      = "/sys/fs/cgroup"
	cudaDiscoverTick    = 1 * time.Second
	podGPUDiscoverTick  = 1 * time.Second

	cudaActionLock       cudaCheckpointAction = "lock"
	cudaActionCheckpoint cudaCheckpointAction = "checkpoint"
	cudaActionRestore    cudaCheckpointAction = "restore"
	cudaActionUnlock     cudaCheckpointAction = "unlock"
)

func prepareExternalCUDA(ctx context.Context, req CheckpointRequest, sourcePID int, manifest *CheckpointManifest, log logr.Logger) error {
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
		if err := runCudaCheckpoint(ctx, pid, cudaActionLock, ""); err != nil {
			unlockCUDAProcesses(context.Background(), locked, log)
			return fmt.Errorf("cuda lock failed for PID %d: %w", pid, err)
		}
		locked = append(locked, pid)
	}

	for _, pid := range cudaPIDs {
		if err := runCudaCheckpoint(ctx, pid, cudaActionCheckpoint, ""); err != nil {
			unlockCUDAProcesses(context.Background(), locked, log)
			return fmt.Errorf("cuda checkpoint failed for PID %d: %w", pid, err)
		}
	}

	sourceGPUUUIDs, err := getPodGPUUUIDsWithRetry(ctx, req.PodName, req.PodNamespace, req.ContainerName, log)
	if err != nil {
		return fmt.Errorf("failed to discover source GPU UUIDs: %w", err)
	}
	if len(sourceGPUUUIDs) == 0 {
		unlockCUDAProcesses(context.Background(), locked, log)
		return fmt.Errorf("no source GPU UUIDs found for %s/%s container %s", req.PodNamespace, req.PodName, req.ContainerName)
	}

	manifest.ExternalRestore = &ExternalRestoreConfig{
		CUDA: &CUDARestoreData{
			PIDs:           cudaPIDs,
			SourceGPUUUIDs: sourceGPUUUIDs,
			Locked:         true,
		},
	}

	log.Info("Prepared external CUDA checkpoint metadata",
		"cuda_pids", len(cudaPIDs),
		"source_gpu_uuids", len(sourceGPUUUIDs),
		"container_cg_path", cgroupPath,
		"source_pod", req.PodName,
		"source_namespace", req.PodNamespace,
		"source_container", req.ContainerName,
	)
	return nil
}

func unlockExternalCUDA(manifest *CheckpointManifest, log logr.Logger) {
	if manifest == nil || manifest.ExternalRestore == nil || manifest.ExternalRestore.CUDA == nil {
		return
	}
	if !manifest.ExternalRestore.CUDA.Locked {
		return
	}
	unlockCUDAProcesses(context.Background(), manifest.ExternalRestore.CUDA.PIDs, log)
}

func unlockCUDAProcesses(ctx context.Context, pids []int, log logr.Logger) {
	for _, pid := range pids {
		if err := runCudaCheckpoint(ctx, pid, cudaActionUnlock, ""); err != nil {
			log.Error(err, "Failed to unlock CUDA process", "pid", pid)
		}
	}
}

func RestoreExternalCUDA(ctx context.Context, manifest *CheckpointManifest, podName, podNamespace, containerName string, containerPID int, restoredPID int, log logr.Logger) error {
	if manifest.ExternalRestore == nil || manifest.ExternalRestore.CUDA == nil {
		log.V(1).Info("No CUDA restore data in manifest, skipping")
		return nil
	}

	cudaData := manifest.ExternalRestore.CUDA
	if len(cudaData.PIDs) == 0 {
		log.V(1).Info("No CUDA PIDs recorded, skipping")
		return nil
	}

	targetUUIDs, err := getPodGPUUUIDsWithRetry(ctx, podName, podNamespace, containerName, log)
	if err != nil {
		return fmt.Errorf("failed to get target GPU UUIDs: %w", err)
	}
	if len(cudaData.SourceGPUUUIDs) == 0 {
		return fmt.Errorf("missing source GPU UUIDs in checkpoint manifest")
	}
	if len(targetUUIDs) == 0 {
		return fmt.Errorf("missing target GPU UUIDs for %s/%s container %s", podNamespace, podName, containerName)
	}
	deviceMap, err := buildCUDADeviceMap(cudaData.SourceGPUUUIDs, targetUUIDs)
	if err != nil {
		return fmt.Errorf("failed to build CUDA device map: %w", err)
	}

	cgroupPath, err := getContainerCgroupPath(containerPID)
	if err != nil {
		return fmt.Errorf("failed to get container cgroup path: %w", err)
	}
	cudaPIDs, err := findRestoredCUDAPIDs(ctx, cgroupPath, restoredPID, log)
	if err != nil {
		return fmt.Errorf("failed to discover restored CUDA PIDs: %w", err)
	}
	if len(cudaPIDs) == 0 {
		return fmt.Errorf("checkpoint captured %d CUDA PIDs but none found in restored cgroup %s", len(cudaData.PIDs), cgroupPath)
	}

	for _, pid := range cudaPIDs {
		if err := runCudaCheckpoint(ctx, pid, cudaActionRestore, deviceMap); err != nil {
			return fmt.Errorf("cuda restore failed for PID %d: %w", pid, err)
		}
		if err := runCudaCheckpoint(ctx, pid, cudaActionUnlock, ""); err != nil {
			return fmt.Errorf("cuda unlock failed for PID %d: %w", pid, err)
		}
	}

	log.Info("CUDA restore completed",
		"cuda_pids", len(cudaPIDs),
		"device_map", deviceMap,
		"containerID", containerPID,
		"restoredPID", restoredPID,
	)
	return nil
}

func findRestoredCUDAPIDs(ctx context.Context, cgroupPath string, restoredPID int, log logr.Logger) ([]int, error) {
	ticker := time.NewTicker(cudaDiscoverTick)
	defer ticker.Stop()

	attempt := 0
	for {
		attempt++
		cgroupPIDs, err := getCgroupPIDs(cgroupPath)
		if err != nil {
			return nil, fmt.Errorf("failed to list restored cgroup PIDs from %s: %w", cgroupPath, err)
		}

		candidates := make([]int, 0, len(cgroupPIDs)+1)
		seen := make(map[int]struct{}, len(cgroupPIDs)+1)
		if restoredPID > 0 {
			candidates = append(candidates, restoredPID)
			seen[restoredPID] = struct{}{}
		}
		for _, pid := range cgroupPIDs {
			if _, ok := seen[pid]; ok {
				continue
			}
			candidates = append(candidates, pid)
			seen[pid] = struct{}{}
		}

		cudaPIDs := make([]int, 0, len(candidates))
		for _, pid := range candidates {
			ok, err := isCUDAProcess(ctx, pid)
			if err != nil {
				log.V(1).Info("Failed CUDA state probe", "pid", pid, "error", err)
				continue
			}
			if ok {
				cudaPIDs = append(cudaPIDs, pid)
			}
		}

		log.Info("CUDA restore PID discovery attempt",
			"attempt", attempt,
			"cgroup_path", cgroupPath,
			"cgroup_pids", len(cgroupPIDs),
			"candidates", len(candidates),
			"cuda_pids", len(cudaPIDs),
			"sample_pids", truncateIntSlice(cgroupPIDs, 16),
			"sample_cuda", truncateIntSlice(cudaPIDs, 16),
		)
		if len(cudaPIDs) > 0 {
			return cudaPIDs, nil
		}

		select {
		case <-ctx.Done():
			return nil, nil
		case <-ticker.C:
		}
	}
}

func isCUDAProcess(ctx context.Context, pid int) (bool, error) {
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

func truncateIntSlice(values []int, limit int) []int {
	if len(values) <= limit {
		return append([]int(nil), values...)
	}
	return append([]int(nil), values[:limit]...)
}

func runCudaCheckpoint(ctx context.Context, pid int, action cudaCheckpointAction, deviceMap string) error {
	args := []string{"--action", string(action), "--pid", strconv.Itoa(pid)}
	if action == cudaActionRestore && deviceMap != "" {
		args = append(args, "--device-map", deviceMap)
	}

	cmd := exec.CommandContext(ctx, cudaCheckpointBinary, args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("cuda-checkpoint %v failed for pid %d: %w (output: %s)", args, pid, err, strings.TrimSpace(string(output)))
	}
	return nil
}

func buildCUDADeviceMap(sourceUUIDs, targetUUIDs []string) (string, error) {
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

// BuildCUDADeviceMap creates a cuda-checkpoint --device-map value from source and target GPU UUID lists.
func BuildCUDADeviceMap(sourceUUIDs, targetUUIDs []string) (string, error) {
	return buildCUDADeviceMap(sourceUUIDs, targetUUIDs)
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

func getPodGPUUUIDsWithRetry(ctx context.Context, podName, podNamespace, containerName string, log logr.Logger) ([]string, error) {
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

// GetPodGPUUUIDsWithRetry resolves GPU UUIDs for a pod/container from kubelet pod-resources API.
func GetPodGPUUUIDsWithRetry(ctx context.Context, podName, podNamespace, containerName string, log logr.Logger) ([]string, error) {
	return getPodGPUUUIDsWithRetry(ctx, podName, podNamespace, containerName, log)
}
