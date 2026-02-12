// Package orchestrate provides the top-level checkpoint and restore orchestrators.
// These wire together the lib packages (criu, cuda, filesystem, etc.) into
// multi-step workflows.
package orchestrate

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	criurpc "github.com/checkpoint-restore/go-criu/v8/rpc"
	"github.com/go-logr/logr"
	specs "github.com/opencontainers/runtime-spec/specs-go"
	"google.golang.org/protobuf/proto"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/config"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/containerd"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/criu"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/cuda"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/filesystem"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/manifest"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/mounts"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/namespace"
)

// CheckpointRequest holds per-checkpoint identifiers for a checkpoint operation.
type CheckpointRequest struct {
	ContainerID    string
	ContainerName  string
	CheckpointHash string
	CheckpointDir  string
	NodeName       string
	PodName        string
	PodNamespace   string
}

// CheckpointOutcome contains the result of a checkpoint operation.
type CheckpointOutcome struct {
	CheckpointHash string
	CheckpointDir  string
	Data           *manifest.CheckpointManifest
}

// containerSnapshot holds runtime/container info needed for checkpointing.
type containerSnapshot struct {
	PID        int
	RootFS     string
	UpperDir   string
	OCISpec    *specs.Spec
	MountInfo  []mounts.Info
	Namespaces map[namespace.Type]*namespace.NamespaceInfo
}

// Checkpointer performs CRIU checkpoint operations.
type Checkpointer struct {
	discoveryClient *containerd.DiscoveryClient
	log             logr.Logger
}

// NewCheckpointer creates a new checkpointer.
func NewCheckpointer(discoveryClient *containerd.DiscoveryClient, log logr.Logger) *Checkpointer {
	return &Checkpointer{
		discoveryClient: discoveryClient,
		log:             log,
	}
}

// Checkpoint performs a CRIU dump of a container.
// The operation has three phases: introspect, configure, capture.
//
// The checkpoint directory is staged under tmp/<hash> during the operation.
// On success, it is atomically renamed to <hash> at the base path root.
func (c *Checkpointer) Checkpoint(ctx context.Context, req CheckpointRequest, spec *config.CheckpointSpec) (*CheckpointOutcome, error) {
	if spec == nil {
		return nil, fmt.Errorf("checkpoint spec is required")
	}
	checkpointStart := time.Now()
	c.log.Info("=== Starting checkpoint operation ===")

	finalDir := filepath.Join(req.CheckpointDir, req.CheckpointHash)
	tmpDir := filepath.Join(req.CheckpointDir, config.TmpCheckpointDir, req.CheckpointHash)
	if err := os.MkdirAll(tmpDir, 0700); err != nil {
		return nil, fmt.Errorf("failed to create checkpoint directory: %w", err)
	}

	// Open image directory FD for CRIU
	imageDir, imageDirFD, err := criu.OpenPathForCRIU(tmpDir)
	if err != nil {
		return nil, fmt.Errorf("failed to open image directory: %w", err)
	}
	defer imageDir.Close()

	// Phase 1: Introspect container state
	state, err := c.introspect(ctx, req.ContainerID)
	if err != nil {
		return nil, err
	}

	// Phase 2: Configure CRIU options and build checkpoint manifest
	criuOpts, data, err := c.configure(ctx, state, req, spec, tmpDir, imageDirFD)
	if err != nil {
		return nil, err
	}
	defer cuda.UnlockFromManifest(data, c.log)

	// Phase 3: Capture — CRIU dump, /dev/shm, rootfs diff
	criuDumpDuration, err := c.capture(criuOpts, data, state, tmpDir)
	if err != nil {
		return nil, err
	}

	// Atomic rename: tmp/<hash> -> <hash> signals checkpoint completeness
	if err := os.Rename(tmpDir, finalDir); err != nil {
		return nil, fmt.Errorf("failed to finalize checkpoint directory: %w", err)
	}

	totalDuration := time.Since(checkpointStart)
	c.log.Info("=== Checkpoint operation completed ===",
		"total_duration", totalDuration,
		"criu_dump_duration", criuDumpDuration,
	)

	return &CheckpointOutcome{
		CheckpointHash: req.CheckpointHash,
		CheckpointDir:  finalDir,
		Data:           data,
	}, nil
}

func (c *Checkpointer) introspect(ctx context.Context, containerID string) (*containerSnapshot, error) {
	pid, ociSpec, err := c.discoveryClient.ResolveContainer(ctx, containerID)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve container: %w", err)
	}

	rootFS, err := filesystem.GetRootFS(pid)
	if err != nil {
		return nil, fmt.Errorf("failed to get rootfs: %w", err)
	}
	upperDir, err := filesystem.GetOverlayUpperDir(pid)
	if err != nil {
		return nil, fmt.Errorf("failed to get overlay upperdir: %w", err)
	}
	mountInfo, err := mounts.ReadFromHostProc(pid)
	if err != nil {
		return nil, fmt.Errorf("failed to parse mountinfo: %w", err)
	}
	namespaces, err := namespace.GetAll(pid)
	if err != nil {
		return nil, fmt.Errorf("failed to get namespaces: %w", err)
	}

	return &containerSnapshot{
		PID:        pid,
		RootFS:     rootFS,
		UpperDir:   upperDir,
		OCISpec:    ociSpec,
		MountInfo:  mountInfo,
		Namespaces: namespaces,
	}, nil
}

func (c *Checkpointer) configure(
	ctx context.Context,
	state *containerSnapshot,
	req CheckpointRequest,
	spec *config.CheckpointSpec,
	checkpointDir string,
	imageDirFD int32,
) (*criurpc.CriuOpts, *manifest.CheckpointManifest, error) {
	criuOpts, err := criu.BuildDumpOptions(
		&spec.CRIU,
		state.PID,
		imageDirFD,
		state.RootFS,
		state.MountInfo,
		state.OCISpec,
		state.Namespaces,
		c.log,
	)
	if err != nil {
		return nil, nil, err
	}

	// Write CRIU config file (for options unavailable via RPC)
	configPath := filepath.Join(checkpointDir, config.CheckpointCRIUConfFilename)
	if err := os.WriteFile(configPath, []byte(criu.GenerateCRIUConfContent(&spec.CRIU)), 0644); err != nil {
		return nil, nil, fmt.Errorf("failed to write CRIU config file: %w", err)
	}
	criuOpts.ConfigFile = proto.String(configPath)

	// Build and save the checkpoint manifest
	m := manifest.NewCheckpointManifest(
		req.CheckpointHash,
		manifest.NewCRIUDumpManifest(criuOpts, spec.CRIU),
		manifest.NewSourcePodManifest(req.ContainerID, state.PID, req.NodeName, req.PodName, req.PodNamespace),
		manifest.NewFilesystemManifest(spec.RootfsExclusions, state.UpperDir, state.OCISpec),
		manifest.NewNamespaceManifest(state.Namespaces),
	)

	cudaReq := cuda.CheckpointRequest{
		PodName:       req.PodName,
		PodNamespace:  req.PodNamespace,
		ContainerName: req.ContainerName,
	}
	if err := cuda.PrepareCheckpoint(ctx, cudaReq, state.PID, m, c.log); err != nil {
		return nil, nil, fmt.Errorf("failed to prepare external CUDA checkpoint metadata: %w", err)
	}

	if err := manifest.Write(checkpointDir, m); err != nil {
		cuda.UnlockFromManifest(m, c.log)
		return nil, nil, fmt.Errorf("failed to write checkpoint manifest: %w", err)
	}

	return criuOpts, m, nil
}

func (c *Checkpointer) capture(
	criuOpts *criurpc.CriuOpts,
	data *manifest.CheckpointManifest,
	state *containerSnapshot,
	checkpointDir string,
) (time.Duration, error) {
	criuDumpDuration, err := criu.ExecuteDump(criuOpts, checkpointDir, c.log)
	if err != nil {
		return 0, err
	}

	if err := filesystem.CaptureDevShm(state.PID, checkpointDir, c.log); err != nil {
		c.log.Error(err, "Failed to capture /dev/shm contents")
	}

	filesystem.CaptureRootfsState(state.UpperDir, checkpointDir, data, c.log)

	return criuDumpDuration, nil
}
