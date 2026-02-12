// server.go provides the HTTP-over-UDS server for checkpoint and restore operations.
package externalrestore

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/go-logr/logr"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/checkpoint"
)

const (
	// DefaultSocketPath is the default UDS socket path.
	DefaultSocketPath = "/var/run/chrek/chrek.sock"
)

// ServerConfig holds configuration for the UDS server.
type ServerConfig struct {
	SocketPath     string
	NodeName       string
	CheckpointSpec *checkpoint.CheckpointSpec
}

// Server is the HTTP-over-UDS server for checkpoint and restore operations.
type Server struct {
	cfg          ServerConfig
	httpServer   *http.Server
	listener     net.Listener
	restorer     *Restorer
	checkpointer *checkpoint.Checkpointer
	log          logr.Logger
}

// NewServer creates a new UDS server.
func NewServer(cfg ServerConfig, checkpointer *checkpoint.Checkpointer, restorer *Restorer, log logr.Logger) *Server {
	s := &Server{
		cfg:          cfg,
		restorer:     restorer,
		checkpointer: checkpointer,
		log:          log,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/checkpoint", s.handleCheckpoint)
	mux.HandleFunc("/restore", s.handleRestore)

	s.httpServer = &http.Server{
		Handler: mux,
	}

	return s
}

// Start begins listening on the UDS socket. Blocks until shutdown.
func (s *Server) Start() error {
	socketPath := s.cfg.SocketPath
	if socketPath == "" {
		socketPath = DefaultSocketPath
	}

	// Ensure parent directory exists
	if err := os.MkdirAll(filepath.Dir(socketPath), 0755); err != nil {
		return fmt.Errorf("failed to create socket directory: %w", err)
	}

	// Remove stale socket file
	os.Remove(socketPath)

	ln, err := net.Listen("unix", socketPath)
	if err != nil {
		return fmt.Errorf("failed to listen on %s: %w", socketPath, err)
	}
	s.listener = ln

	// Make socket world-accessible so sidecars can connect
	if err := os.Chmod(socketPath, 0666); err != nil {
		s.log.Error(err, "Failed to chmod socket")
	}

	s.log.Info("UDS server listening", "socket", socketPath)
	return s.httpServer.Serve(ln)
}

// Shutdown gracefully shuts down the server.
func (s *Server) Shutdown(ctx context.Context) error {
	return s.httpServer.Shutdown(ctx)
}

func (s *Server) handleCheckpoint(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req CheckpointAPIRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, CheckpointAPIResponse{
			Success: false,
			Error:   fmt.Sprintf("Invalid request body: %v", err),
		})
		return
	}

	if req.ContainerID == "" {
		writeJSON(w, http.StatusBadRequest, CheckpointAPIResponse{
			Success: false,
			Error:   "container_id is required",
		})
		return
	}

	if req.CheckpointHash == "" {
		req.CheckpointHash = fmt.Sprintf("ckpt-%d", time.Now().UnixNano())
	}

	params := checkpoint.CheckpointRequest{
		ContainerID:    req.ContainerID,
		ContainerName:  req.ContainerName,
		CheckpointHash: req.CheckpointHash,
		CheckpointDir:  s.cfg.CheckpointSpec.BasePath,
		NodeName:       s.cfg.NodeName,
		PodName:        req.PodName,
		PodNamespace:   req.PodNamespace,
	}

	result, err := s.checkpointer.Checkpoint(r.Context(), params, s.cfg.CheckpointSpec)
	if err != nil {
		s.log.Error(err, "Checkpoint failed")
		writeJSON(w, http.StatusInternalServerError, CheckpointAPIResponse{
			Success: false,
			Error:   err.Error(),
		})
		return
	}

	s.log.Info("Checkpoint completed", "checkpoint_hash", result.CheckpointHash)
	writeJSON(w, http.StatusOK, CheckpointAPIResponse{
		Success:        true,
		CheckpointHash: result.CheckpointHash,
		Message:        fmt.Sprintf("Checkpoint created at %s", result.CheckpointDir),
	})
}

func (s *Server) handleRestore(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req RestoreAPIRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, RestoreAPIResponse{
			Success: false,
			Error:   fmt.Sprintf("Invalid request body: %v", err),
		})
		return
	}

	if req.CheckpointHash == "" || req.PodName == "" || req.PodNamespace == "" {
		writeJSON(w, http.StatusBadRequest, RestoreAPIResponse{
			Success: false,
			Error:   "checkpoint_hash, pod_name, and pod_namespace are required",
		})
		return
	}

	result, err := s.restorer.Restore(r.Context(), req)
	if err != nil {
		s.log.Error(err, "Restore failed")
		writeJSON(w, http.StatusInternalServerError, RestoreAPIResponse{
			Success: false,
			Error:   err.Error(),
		})
		return
	}

	writeJSON(w, http.StatusOK, result)
}

func writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}
