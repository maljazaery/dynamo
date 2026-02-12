// Package manifest defines the checkpoint manifest data types and constructors.
// The manifest is written at checkpoint time and read at restore time.
package manifest

import (
	"time"

	criurpc "github.com/checkpoint-restore/go-criu/v8/rpc"
	specs "github.com/opencontainers/runtime-spec/specs-go"
	"google.golang.org/protobuf/proto"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/config"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/namespace"
)

// CheckpointManifest is saved as manifest.yaml at checkpoint time and loaded at restore.
type CheckpointManifest struct {
	CheckpointHash string    `yaml:"checkpointHash"`
	CreatedAt      time.Time `yaml:"createdAt"`

	CRIUDump    CRIUDumpManifest     `yaml:"criuDump"`
	K8s         SourcePodManifest    `yaml:"k8s"`
	Filesystem  FilesystemManifest   `yaml:"filesystem"`
	Namespaces  []NamespaceEntry     `yaml:"namespaces"`
	CUDARestore *CUDARestoreManifest `yaml:"cudaRestore,omitempty"`
}

// NewCheckpointManifest assembles a CheckpointManifest from per-module builders.
func NewCheckpointManifest(
	checkpointHash string,
	criuDump CRIUDumpManifest,
	k8s SourcePodManifest,
	filesystem FilesystemManifest,
	namespaces []NamespaceEntry,
) *CheckpointManifest {
	return &CheckpointManifest{
		CheckpointHash: checkpointHash,
		CreatedAt:      time.Now().UTC(),
		CRIUDump:       criuDump,
		K8s:            k8s,
		Filesystem:     filesystem,
		Namespaces:     namespaces,
	}
}

// CRIUDumpManifest stores the resolved dump-time CRIU mount plan used for restore.
type CRIUDumpManifest struct {
	CRIU     config.CRIUSettings `yaml:"criu"`
	ExtMnt   []ExtMountEntry     `yaml:"extMnt,omitempty"`
	External []string            `yaml:"external,omitempty"`
	SkipMnt  []string            `yaml:"skipMnt,omitempty"`
}

// ExtMountEntry is a serializable CRIU ext-mount entry in checkpoint manifests.
type ExtMountEntry struct {
	Key string `yaml:"key"`
	Val string `yaml:"val"`
}

// NewCRIUDumpManifest serializes resolved dump options for restore.
func NewCRIUDumpManifest(criuOpts *criurpc.CriuOpts, settings config.CRIUSettings) CRIUDumpManifest {
	m := CRIUDumpManifest{CRIU: settings}
	if criuOpts == nil {
		return m
	}

	for _, mount := range criuOpts.ExtMnt {
		if mount == nil || mount.GetKey() == "" {
			continue
		}
		m.ExtMnt = append(m.ExtMnt, ExtMountEntry{
			Key: mount.GetKey(),
			Val: mount.GetVal(),
		})
	}
	m.External = append([]string(nil), criuOpts.External...)
	m.SkipMnt = append([]string(nil), criuOpts.SkipMnt...)
	return m
}

// SourcePodManifest records the source pod identity at checkpoint time.
type SourcePodManifest struct {
	ContainerID  string `yaml:"containerId"`
	PID          int    `yaml:"pid"`
	SourceNode   string `yaml:"sourceNode"`
	PodName      string `yaml:"podName"`
	PodNamespace string `yaml:"podNamespace"`
}

// NewSourcePodManifest constructs a SourcePodManifest from individual fields.
func NewSourcePodManifest(containerID string, pid int, sourceNode, podName, podNamespace string) SourcePodManifest {
	return SourcePodManifest{
		ContainerID:  containerID,
		PID:          pid,
		SourceNode:   sourceNode,
		PodName:      podName,
		PodNamespace: podNamespace,
	}
}

// FilesystemManifest holds runtime filesystem state captured at checkpoint time.
type FilesystemManifest struct {
	Exclusions      config.FilesystemConfig `yaml:"exclusions"`
	UpperDir        string                  `yaml:"upperDir,omitempty"`
	ExternalPaths   []string                `yaml:"externalPaths,omitempty"`
	BindMountDests  []string                `yaml:"bindMountDests,omitempty"`
	HasRootfsDiff   bool                    `yaml:"hasRootfsDiff"`
	HasDeletedFiles bool                    `yaml:"hasDeletedFiles"`
}

// NewFilesystemManifest constructs FilesystemManifest from config, overlay state, and OCI spec.
func NewFilesystemManifest(exclusions config.FilesystemConfig, upperDir string, ociSpec *specs.Spec) FilesystemManifest {
	meta := FilesystemManifest{
		Exclusions: exclusions,
		UpperDir:   upperDir,
	}
	if ociSpec == nil {
		return meta
	}

	if ociSpec.Linux != nil {
		meta.ExternalPaths = make([]string, 0, len(ociSpec.Linux.MaskedPaths)+len(ociSpec.Linux.ReadonlyPaths))
		meta.ExternalPaths = append(meta.ExternalPaths, ociSpec.Linux.MaskedPaths...)
		meta.ExternalPaths = append(meta.ExternalPaths, ociSpec.Linux.ReadonlyPaths...)
	}
	for _, m := range ociSpec.Mounts {
		if m.Type == "bind" {
			meta.BindMountDests = append(meta.BindMountDests, m.Destination)
		}
	}
	return meta
}

// NamespaceEntry stores namespace information saved in checkpoint manifests.
type NamespaceEntry struct {
	Type       string `yaml:"type"`
	Inode      uint64 `yaml:"inode"`
	IsExternal bool   `yaml:"isExternal"`
}

// NewNamespaceEntries constructs namespace manifest entries from introspected namespaces.
func NewNamespaceEntries(namespaces map[namespace.Type]*namespace.Info) []NamespaceEntry {
	if len(namespaces) == 0 {
		return nil
	}

	result := make([]NamespaceEntry, 0, len(namespaces))
	for nsType, nsInfo := range namespaces {
		result = append(result, NamespaceEntry{
			Type:       string(nsType),
			Inode:      nsInfo.Inode,
			IsExternal: nsInfo.IsExternal,
		})
	}
	return result
}

// CUDARestoreManifest captures CUDA state from checkpoint time for restore.
type CUDARestoreManifest struct {
	PIDs           []int    `yaml:"pids"`
	SourceGPUUUIDs []string `yaml:"sourceGpuUuids"`
	Locked         bool     `yaml:"locked"`
}

// buildExternalMountMaps is used by criu/dump to serialize externalized mount paths.
// Exported so pkg/criu can use it.
func BuildExternalMountMaps(paths []string) []*criurpc.ExtMountMap {
	extMnt := make([]*criurpc.ExtMountMap, 0, len(paths))
	existing := make(map[string]struct{}, len(paths))
	for _, path := range paths {
		if path == "" {
			continue
		}
		if _, ok := existing[path]; ok {
			continue
		}
		extMnt = append(extMnt, &criurpc.ExtMountMap{
			Key: proto.String(path),
			Val: proto.String(path),
		})
		existing[path] = struct{}{}
	}
	return extMnt
}
