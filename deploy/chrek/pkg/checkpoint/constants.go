// constants.go defines shared constants used across checkpoint and restore packages.
package checkpoint

const (
	// HostProcPath is the mount point for the host's /proc in DaemonSet pods.
	HostProcPath = "/host/proc"

	// DevShmDirName is the directory name for captured /dev/shm contents.
	DevShmDirName = "dev-shm"

	// KubeLabelIsCheckpointSource is the pod label that triggers automatic checkpointing.
	// Set by the operator on checkpoint-eligible pods.
	KubeLabelIsCheckpointSource = "nvidia.com/chrek-is-checkpoint-source"

	// KubeLabelCheckpointHash is the pod label specifying the checkpoint identity hash.
	// Set by the operator on checkpoint-eligible pods. Also used as the DynamoCheckpoint
	// CR name, so it doubles as the CR lookup key.
	KubeLabelCheckpointHash = "nvidia.com/chrek-checkpoint-hash"

	// KubeLabelIsRestoreTarget is the pod label that triggers automatic restore.
	// Set by the operator on restore-eligible (placeholder) pods.
	KubeLabelIsRestoreTarget = "nvidia.com/chrek-is-restore-target"

	// KubeAnnotationCheckpointStatus is set on checkpoint-source pods by the watcher
	// to track checkpoint progress. Values: "in_progress", "completed", "failed".
	// Persists across agent restarts for idempotent checkpoint operations.
	KubeAnnotationCheckpointStatus = "nvidia.com/chrek-checkpoint-status"

	// KubeAnnotationRestoreStatus is set on restore-target (placeholder) pods by the
	// watcher to track restore progress. Values: "in_progress", "completed", "failed".
	// Persists across agent restarts for idempotent restore operations.
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

	// DescriptorsFilename is the name of the file descriptors file.
	DescriptorsFilename = "descriptors.yaml"

	// RootfsDiffFilename is the name of the rootfs diff tar in checkpoint directories.
	RootfsDiffFilename = "rootfs-diff.tar"

	// DeletedFilesFilename is the name of the deleted files JSON in checkpoint directories.
	DeletedFilesFilename = "deleted-files.json"
)
