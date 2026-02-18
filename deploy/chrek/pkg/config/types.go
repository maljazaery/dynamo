package config

import (
	"fmt"
	"os"
	"strings"
)

// AgentConfig holds the full agent configuration: static checkpoint settings
// from the ConfigMap YAML, plus runtime fields from environment variables.
type AgentConfig struct {
	// NodeName is the Kubernetes node name (from NODE_NAME env, downward API)
	NodeName string `yaml:"-"`

	// RestrictedNamespace restricts pod watching to this namespace (optional)
	RestrictedNamespace string `yaml:"-"`

	Checkpoint CheckpointSpec `yaml:"checkpoint"`
}

// LoadEnvOverrides applies environment variable overrides to the AgentConfig.
func (c *AgentConfig) LoadEnvOverrides() {
	if v := os.Getenv("NODE_NAME"); v != "" {
		c.NodeName = v
	}
	if v := os.Getenv("RESTRICTED_NAMESPACE"); v != "" {
		c.RestrictedNamespace = v
	}
}

// Validate checks that the configuration has valid values.
func (c *AgentConfig) Validate() error {
	return c.Checkpoint.Validate()
}

// CheckpointSpec is the static checkpoint spec loaded from ConfigMap YAML.
type CheckpointSpec struct {
	// BasePath is the base directory for checkpoint storage (PVC mount point).
	BasePath string `yaml:"basePath"`

	// NSRestorePath is the path to the nsrestore binary in the placeholder image.
	NSRestorePath string `yaml:"nsRestorePath"`

	// CRIU options for dump and restore operations
	CRIU CRIUSettings `yaml:"criu"`

	// RootfsExclusions defines paths to exclude from rootfs diff capture
	RootfsExclusions FilesystemConfig `yaml:"rootfsExclusions"`
}

// Validate checks that the CheckpointSpec has valid values.
func (c *CheckpointSpec) Validate() error {
	if strings.TrimSpace(c.BasePath) == "" {
		return &ConfigError{Field: "basePath", Message: "basePath is required"}
	}
	if c.NSRestorePath == "" {
		return &ConfigError{Field: "nsRestorePath", Message: "nsRestorePath is required"}
	}
	cgroupMode := strings.ToLower(strings.TrimSpace(c.CRIU.ManageCgroupsMode))
	switch cgroupMode {
	case "ignore", "soft", "full", "strict":
		c.CRIU.ManageCgroupsMode = cgroupMode
	default:
		return &ConfigError{
			Field:   "criu.manageCgroupsMode",
			Message: "must be one of: ignore, soft, full, strict",
		}
	}
	return c.RootfsExclusions.Validate()
}

// CRIUSettings holds CRIU-specific configuration options.
// Options are categorized by how they are passed to CRIU:
//   - RPC options: Passed via go-criu CriuOpts protobuf
//   - CRIU conf file options: Written to criu.conf (NOT available via RPC)
type CRIUSettings struct {
	// === Dump RPC Options (passed via go-criu CriuOpts) ===

	// GhostLimit is the maximum ghost file size in bytes.
	// Ghost files are deleted-but-open files that CRIU needs to checkpoint.
	// 512MB is recommended for GPU workloads with large memory allocations.
	GhostLimit uint32 `yaml:"ghostLimit"`

	// LogLevel is the CRIU logging verbosity (0-4).
	LogLevel int32 `yaml:"logLevel"`

	// WorkDir is the CRIU work directory for temporary files.
	WorkDir string `yaml:"workDir"`

	// AutoDedup enables auto-deduplication of memory pages.
	AutoDedup bool `yaml:"autoDedup"`

	// LazyPages enables lazy page migration (experimental).
	LazyPages bool `yaml:"lazyPages"`

	// LeaveRunning keeps the process running after checkpoint (dump only).
	LeaveRunning bool `yaml:"leaveRunning"`

	// ShellJob allows checkpointing session leaders (containers are often session leaders).
	ShellJob bool `yaml:"shellJob"`

	// TcpClose closes TCP connections instead of preserving them (pod IPs change on restore).
	TcpClose bool `yaml:"tcpClose"`

	// FileLocks allows checkpointing processes with file locks.
	FileLocks bool `yaml:"fileLocks"`

	// OrphanPtsMaster allows checkpointing containers with TTYs.
	OrphanPtsMaster bool `yaml:"orphanPtsMaster"`

	// ExtUnixSk allows external Unix sockets.
	ExtUnixSk bool `yaml:"extUnixSk"`

	// LinkRemap handles deleted-but-open files.
	LinkRemap bool `yaml:"linkRemap"`

	// ExtMasters allows external bind mount masters.
	ExtMasters bool `yaml:"extMasters"`

	// ManageCgroupsMode controls cgroup handling: ignore/soft/full/strict.
	ManageCgroupsMode string `yaml:"manageCgroupsMode"`

	// === Restore-specific RPC Options ===
	// These only apply during CRIU restore (not dump). They are persisted into
	// the checkpoint manifest at dump time and read by nsrestore at restore time.

	// RstSibling restores the process as a sibling (required for swrk/go-criu restore).
	RstSibling bool `yaml:"rstSibling"`

	// MntnsCompatMode enables mount namespace compatibility mode in CRIU restore.
	MntnsCompatMode bool `yaml:"mntnsCompatMode"`

	// EvasiveDevices allows CRIU to use any path to a device file if the original is inaccessible.
	EvasiveDevices bool `yaml:"evasiveDevices"`

	// ForceIrmap forces resolving names for inotify/fsnotify watches during restore.
	ForceIrmap bool `yaml:"forceIrmap"`

	// === CRIU Binary and Conf File Options ===

	// BinaryPath is the path to the criu binary.
	BinaryPath string `yaml:"binaryPath"`

	// LibDir is the path to CRIU plugin directory (e.g., /usr/local/lib/criu).
	// Required for CUDA checkpoint/restore.
	LibDir string `yaml:"libDir"`

	// AllowUprobes allows user-space probes (required for CUDA checkpoints).
	AllowUprobes bool `yaml:"allowUprobes"`

	// SkipInFlight skips in-flight TCP connections during checkpoint/restore.
	SkipInFlight bool `yaml:"skipInFlight"`
}

// FilesystemConfig is the static config for rootfs exclusions (from values.yaml).
type FilesystemConfig struct {
	// SystemDirs are system directories that should be excluded from rootfs diff.
	SystemDirs []string `yaml:"systemDirs"`

	// CacheDirs are cache directories that can safely be excluded to reduce checkpoint size.
	CacheDirs []string `yaml:"cacheDirs"`

	// AdditionalExclusions are custom paths to exclude from the rootfs diff.
	AdditionalExclusions []string `yaml:"additionalExclusions"`
}

// GetAllExclusions returns all exclusion paths combined.
func (c *FilesystemConfig) GetAllExclusions() []string {
	if c == nil {
		return nil
	}
	total := len(c.SystemDirs) + len(c.CacheDirs) + len(c.AdditionalExclusions)
	exclusions := make([]string, 0, total)
	exclusions = append(exclusions, c.SystemDirs...)
	exclusions = append(exclusions, c.CacheDirs...)
	exclusions = append(exclusions, c.AdditionalExclusions...)
	return exclusions
}

// Validate checks that the FilesystemConfig has valid values.
func (c *FilesystemConfig) Validate() error {
	if c == nil {
		return nil
	}
	for _, path := range c.GetAllExclusions() {
		if !strings.HasPrefix(path, "./") {
			return &ConfigError{
				Field:   "rootfsExclusions",
				Message: "all exclusion paths must start with './' (got: " + path + ")",
			}
		}
	}
	return nil
}

// ConfigError represents a configuration validation error.
type ConfigError struct {
	Field   string
	Message string
}

func (e *ConfigError) Error() string {
	return fmt.Sprintf("config error: %s: %s", e.Field, e.Message)
}
