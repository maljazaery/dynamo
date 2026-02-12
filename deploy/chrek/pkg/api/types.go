// Package api provides the UDS HTTP server and client for checkpoint/restore operations.
package api

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
