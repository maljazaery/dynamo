// Package config defines shared constants, configuration types, and settings
// used across all chrek packages.
package config

const (
	// HostProcPath is the mount point for the host's /proc in DaemonSet pods.
	HostProcPath = "/host/proc"

	// DevShmDirName is the directory name for captured /dev/shm contents.
	DevShmDirName = "dev-shm"

	// KubeLabelIsCheckpointSource is the pod label that triggers automatic checkpointing.
	KubeLabelIsCheckpointSource = "nvidia.com/chrek-is-checkpoint-source"

	// KubeLabelCheckpointHash is the pod label specifying the checkpoint identity hash.
	KubeLabelCheckpointHash = "nvidia.com/chrek-checkpoint-hash"

	// KubeLabelIsRestoreTarget is the pod label that triggers automatic restore.
	KubeLabelIsRestoreTarget = "nvidia.com/chrek-is-restore-target"

	// KubeAnnotationCheckpointStatus tracks checkpoint progress on source pods.
	// Values: "in_progress", "completed", "failed".
	KubeAnnotationCheckpointStatus = "nvidia.com/chrek-checkpoint-status"

	// KubeAnnotationRestoreStatus tracks restore progress on target pods.
	// Values: "in_progress", "completed", "failed".
	KubeAnnotationRestoreStatus = "nvidia.com/chrek-restore-status"

	// DumpLogFilename is the CRIU dump (checkpoint) log filename.
	DumpLogFilename = "dump.log"

	// CheckpointCRIUConfFilename is the CRIU config file written at checkpoint time.
	CheckpointCRIUConfFilename = "criu.conf"

	// TmpCheckpointDir is the subdirectory under the checkpoint base path where
	// in-progress checkpoints are staged. On completion, the directory is renamed
	// from tmp/<id> to <id> at the base path root, so directory existence at the
	// root means the checkpoint is complete and ready.
	TmpCheckpointDir = "tmp"

	// CheckpointManifestFilename is the name of the manifest file in checkpoint directories.
	CheckpointManifestFilename = "manifest.yaml"

	// RootfsDiffFilename is the name of the rootfs diff tar in checkpoint directories.
	RootfsDiffFilename = "rootfs-diff.tar"

	// DeletedFilesFilename is the name of the deleted files JSON in checkpoint directories.
	DeletedFilesFilename = "deleted-files.json"

	// RestoreLogFilename is the CRIU restore log filename.
	RestoreLogFilename = "restore.log"
)
