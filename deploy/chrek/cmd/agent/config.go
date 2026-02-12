// config.go provides configuration loading for the checkpoint agent.
package main

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/config"
)

// ConfigMapPath is the default path where the ConfigMap is mounted.
const ConfigMapPath = "/etc/chrek/config.yaml"

// FullConfig is the root configuration structure loaded from the ConfigMap.
type FullConfig struct {
	Agent      AgentConfig          `yaml:"agent"`
	Checkpoint config.CheckpointSpec `yaml:"checkpoint"`
}

// AgentConfig holds the runtime configuration for the checkpoint agent daemon.
type AgentConfig struct {
	// NodeName is the Kubernetes node name (from NODE_NAME env, downward API)
	NodeName string `yaml:"-"`

	// RestrictedNamespace restricts pod watching to this namespace (optional)
	RestrictedNamespace string `yaml:"-"`
}

// LoadConfig loads the full configuration from a YAML file.
func LoadConfig(path string) (*FullConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file %s: %w", path, err)
	}

	cfg := &FullConfig{}
	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config file %s: %w", path, err)
	}

	cfg.Agent.loadEnvOverrides()
	return cfg, nil
}

// LoadConfigOrDefault loads configuration from a file, falling back to defaults if the file doesn't exist.
func LoadConfigOrDefault(path string) (*FullConfig, error) {
	cfg, err := LoadConfig(path)
	if err != nil {
		if os.IsNotExist(err) {
			cfg = &FullConfig{}
			cfg.Agent.loadEnvOverrides()
			return cfg, nil
		}
		return nil, err
	}
	return cfg, nil
}

// loadEnvOverrides applies environment variable overrides to the AgentConfig.
func (c *AgentConfig) loadEnvOverrides() {
	if v := os.Getenv("NODE_NAME"); v != "" {
		c.NodeName = v
	}
	if v := os.Getenv("RESTRICTED_NAMESPACE"); v != "" {
		c.RestrictedNamespace = v
	}
}

// Validate checks that the configuration has valid values.
func (c *FullConfig) Validate() error {
	return c.Checkpoint.Validate()
}
