// Package externalrestore provides external restore orchestration for the DaemonSet.
// The DaemonSet performs all restore operations externally: rootfs replay, CRIU via
// nsenter + criu-helper, and CUDA restore. The placeholder pod just donates namespaces.
package externalrestore

// CheckpointAPIRequest is the JSON body for POST /checkpoint.
type CheckpointAPIRequest struct {
	ContainerID    string `json:"container_id"`
	ContainerName  string `json:"container_name,omitempty"`
	CheckpointHash string `json:"checkpoint_hash,omitempty"`
	PodName        string `json:"pod_name,omitempty"`
	PodNamespace   string `json:"pod_namespace,omitempty"`
}

// CheckpointAPIResponse is the JSON response for POST /checkpoint.
type CheckpointAPIResponse struct {
	Success        bool   `json:"success"`
	CheckpointHash string `json:"checkpoint_hash,omitempty"`
	Message        string `json:"message,omitempty"`
	Error          string `json:"error,omitempty"`
}

// RestoreAPIRequest is the JSON body for POST /restore.
type RestoreAPIRequest struct {
	CheckpointHash string `json:"checkpoint_hash"`
	PodName        string `json:"pod_name"`
	PodNamespace   string `json:"pod_namespace"`
	ContainerName  string `json:"container_name"`
}

// RestoreAPIResponse is the JSON response for POST /restore.
type RestoreAPIResponse struct {
	Success         bool     `json:"success"`
	RestoredPID     int      `json:"restored_pid,omitempty"`
	RestoredHostPID int      `json:"restored_host_pid,omitempty"`
	CompletedSteps  []string `json:"completed_steps,omitempty"`
	Error           string   `json:"error,omitempty"`
}
