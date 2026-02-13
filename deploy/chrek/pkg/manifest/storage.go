package manifest

import (
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/config"
)

// Write writes a checkpoint manifest file in the checkpoint directory.
func Write(checkpointDir string, data *CheckpointManifest) error {
	content, err := yaml.Marshal(data)
	if err != nil {
		return fmt.Errorf("failed to marshal checkpoint manifest: %w", err)
	}

	manifestPath := filepath.Join(checkpointDir, config.CheckpointManifestFilename)
	if err := os.WriteFile(manifestPath, content, 0600); err != nil {
		return fmt.Errorf("failed to write checkpoint manifest: %w", err)
	}

	return nil
}

// Read reads checkpoint manifest from a checkpoint directory.
func Read(checkpointDir string) (*CheckpointManifest, error) {
	manifestPath := filepath.Join(checkpointDir, config.CheckpointManifestFilename)

	content, err := os.ReadFile(manifestPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read checkpoint manifest: %w", err)
	}

	var data CheckpointManifest
	if err := yaml.Unmarshal(content, &data); err != nil {
		return nil, fmt.Errorf("failed to unmarshal checkpoint manifest: %w", err)
	}

	return &data, nil
}

