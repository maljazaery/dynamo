// Package manifest defines the checkpoint manifest data types and constructors.
// The manifest is written at checkpoint time and read at restore time.
package manifest

import (
	"sort"
	"time"

	criurpc "github.com/checkpoint-restore/go-criu/v8/rpc"
	specs "github.com/opencontainers/runtime-spec/specs-go"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/config"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/inspect"
)

// CheckpointManifest is saved as manifest.yaml at checkpoint time and loaded at restore.
type CheckpointManifest struct {
	CheckpointHash string    `yaml:"checkpointHash"`
	CreatedAt      time.Time `yaml:"createdAt"`

	CRIUDump   CRIUDumpManifest   `yaml:"criuDump"`
	K8s        SourcePodManifest  `yaml:"k8s"`
	Filesystem FilesystemManifest `yaml:"filesystem"`
	Namespaces NamespaceManifest  `yaml:"namespaces"`
	CUDA       CUDAManifest       `yaml:"cudaRestore,omitempty"`
}

// NewCheckpointManifest assembles a CheckpointManifest from per-module builders.
func NewCheckpointManifest(
	checkpointHash string,
	criuDump CRIUDumpManifest,
	k8s SourcePodManifest,
	filesystem FilesystemManifest,
	namespaces NamespaceManifest,
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
	ExtMnt   map[string]string   `yaml:"extMnt,omitempty"`
	External []string            `yaml:"external,omitempty"`
	SkipMnt  []string            `yaml:"skipMnt,omitempty"`
}

// NewCRIUDumpManifest serializes resolved dump options for restore.
func NewCRIUDumpManifest(criuOpts *criurpc.CriuOpts, settings config.CRIUSettings) CRIUDumpManifest {
	m := CRIUDumpManifest{CRIU: settings}
	if criuOpts == nil {
		return m
	}

	m.ExtMnt = make(map[string]string, len(criuOpts.ExtMnt))
	for _, mount := range criuOpts.ExtMnt {
		if mount == nil || mount.GetKey() == "" {
			continue
		}
		m.ExtMnt[mount.GetKey()] = mount.GetVal()
	}
	if len(m.ExtMnt) == 0 {
		m.ExtMnt = nil
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

type NamespaceManifest struct {
	Entries []inspect.NamespaceInfo `yaml:"entries,omitempty"`
}

func NewNamespaceManifest(namespaces map[inspect.NamespaceType]*inspect.NamespaceInfo) NamespaceManifest {
	manifest := NamespaceManifest{
		Entries: make([]inspect.NamespaceInfo, 0, len(namespaces)),
	}
	if len(namespaces) == 0 {
		return manifest
	}

	keys := make([]string, 0, len(namespaces))
	for nsType := range namespaces {
		keys = append(keys, string(nsType))
	}
	sort.Strings(keys)
	for _, key := range keys {
		nsInfo := namespaces[inspect.NamespaceType(key)]
		manifest.Entries = append(manifest.Entries, inspect.NamespaceInfo{
			Type:       inspect.NamespaceType(key),
			Inode:      nsInfo.Inode,
			IsExternal: nsInfo.IsExternal,
		})
	}
	return manifest
}

// CUDAManifest captures CUDA state from checkpoint time for restore.
type CUDAManifest struct {
	PIDs           []int    `yaml:"pids"`
	SourceGPUUUIDs []string `yaml:"sourceGpuUuids"`
}

func NewCUDAManifest(pids []int, sourceGPUUUIDs []string) CUDAManifest {
	return CUDAManifest{
		PIDs:           append([]int(nil), pids...),
		SourceGPUUUIDs: append([]string(nil), sourceGPUUUIDs...),
	}
}

func (m CUDAManifest) IsEmpty() bool {
	return len(m.PIDs) == 0
}
