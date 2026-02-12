package orchestrate

import (
	"context"
	"fmt"
	"path/filepath"
	"time"

	"github.com/go-logr/logr"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/config"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/containerd"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/cuda"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/filesystem"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/manifest"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/nsrunner"
)

const (
	// NSRestoreRunnerBinary is the path to the ns-restore-runner binary in the placeholder image.
	NSRestoreRunnerBinary = "/usr/local/bin/ns-restore-runner"

	// RestoreLogFilename is the CRIU restore log filename.
	RestoreLogFilename = "restore.log"
)

// RestorerConfig holds configuration for the external restore orchestrator.
type RestorerConfig struct {
	CheckpointBasePath  string
	NSRestoreRunnerPath string
	CRIUSettings        *config.CRIUSettings
}

// Restorer orchestrates external restore operations from the DaemonSet.
type Restorer struct {
	cfg             RestorerConfig
	discoveryClient *containerd.DiscoveryClient
	log             logr.Logger
}

// NewRestorer creates a new external restore orchestrator.
func NewRestorer(cfg RestorerConfig, discoveryClient *containerd.DiscoveryClient, log logr.Logger) *Restorer {
	if cfg.NSRestoreRunnerPath == "" {
		cfg.NSRestoreRunnerPath = NSRestoreRunnerBinary
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
	if !m.CUDA.IsEmpty() {
		if len(m.CUDA.SourceGPUUUIDs) == 0 {
			return nil, fmt.Errorf("missing source GPU UUIDs in checkpoint manifest")
		}
		targetGPUUUIDs, err := cuda.GetPodGPUUUIDsWithRetry(ctx, req.PodName, req.PodNamespace, containerName, r.log)
		if err != nil {
			return nil, fmt.Errorf("failed to get target GPU UUIDs: %w", err)
		}
		if len(targetGPUUUIDs) == 0 {
			return nil, fmt.Errorf("missing target GPU UUIDs for %s/%s container %s", req.PodNamespace, req.PodName, containerName)
		}
		cudaDeviceMap, err = cuda.BuildDeviceMap(m.CUDA.SourceGPUUUIDs, targetGPUUUIDs)
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
	nsrunner.EnsureDevNetTunInTargetRoot(targetRoot, r.log)

	// Step 3: Create link_remap stubs
	if err := filesystem.CreateLinkRemapStubs(checkpointPath, targetRoot, r.log); err != nil {
		r.log.Error(err, "Failed to create link_remap stubs")
	}

	// Step 4: Execute nsenter + ns-restore-runner
	restoredPID, restoredHostPID, err := nsrunner.NSEnterCallFromHost(ctx, nsrunner.HostRestoreOptions{
		PlaceholderPID:  placeholderPID,
		RunnerPath:      r.cfg.NSRestoreRunnerPath,
		CheckpointPath:  checkpointPath,
		WorkDir:         m.CRIUDump.CRIU.WorkDir,
		CUDADeviceMap:   cudaDeviceMap,
		RestoreSettings: r.cfg.CRIUSettings,
	}, r.log)
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
	if err := nsrunner.ValidateRestoredProcess(procRoot, restoredPID, restoreLogPath, r.log); err != nil {
		return nil, err
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
