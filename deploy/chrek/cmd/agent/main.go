// Package main provides the chrek DaemonSet agent.
// The agent watches for pods with checkpoint/restore labels on its node
// and triggers operations via the orchestrators.
package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/go-logr/logr"
	"github.com/go-logr/zapr"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/config"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/containerd"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/orchestrate"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/watcher"
)

func main() {
	rootLog := configureLogging()
	agentLog := rootLog.WithName("agent")

	cfg, err := LoadConfigOrDefault(ConfigMapPath)
	if err != nil {
		fatal(agentLog, err, "Failed to load configuration")
	}
	if err := cfg.Validate(); err != nil {
		fatal(agentLog, err, "Invalid configuration")
	}

	discoveryClient, err := containerd.NewDiscoveryClient()
	if err != nil {
		fatal(agentLog, err, "Failed to create discovery client")
	}
	defer discoveryClient.Close()

	checkpointer := orchestrate.NewCheckpointer(discoveryClient, rootLog.WithName("checkpointer"))

	restorer := orchestrate.NewRestorer(
		orchestrate.RestorerConfig{
			CheckpointBasePath: cfg.Checkpoint.BasePath,
			CRIUSettings:       &cfg.Checkpoint.CRIU,
		},
		discoveryClient,
		rootLog.WithName("restorer"),
	)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	agentLog.Info("Starting chrek agent",
		"node", cfg.Agent.NodeName,
		"checkpoint_dir", cfg.Checkpoint.BasePath,
		"watch_namespace", cfg.Agent.RestrictedNamespace,
	)

	watcherCfg := watcher.Config{
		NodeName:            cfg.Agent.NodeName,
		RestrictedNamespace: cfg.Agent.RestrictedNamespace,
		CheckpointSpec:      &cfg.Checkpoint,
	}
	podWatcher, err := watcher.NewWatcher(watcherCfg, checkpointer, restorer, discoveryClient, rootLog.WithName("watcher"))
	if err != nil {
		fatal(agentLog, err, "Failed to create pod watcher")
	}

	// Run watcher in the background
	watcherDone := make(chan error, 1)
	go func() {
		agentLog.Info("Pod watcher started", "label", config.KubeLabelIsCheckpointSource)
		watcherDone <- podWatcher.Start(ctx)
	}()

	// Wait for signal or watcher exit
	select {
	case <-sigChan:
		agentLog.Info("Shutting down")
		cancel()
	case err := <-watcherDone:
		if err != nil {
			fatal(agentLog, err, "Pod watcher exited with error")
		}
	}

	agentLog.Info("Agent stopped")
}

func configureLogging() logr.Logger {
	level := strings.TrimSpace(strings.ToLower(os.Getenv("CHREK_LOG_LEVEL")))
	if level == "" {
		level = "info"
	}

	zapLevel := zapcore.InfoLevel
	parseErr := error(nil)
	switch level {
	case "trace", "debug":
		zapLevel = zapcore.DebugLevel
	case "info":
		zapLevel = zapcore.InfoLevel
	case "warn", "warning":
		zapLevel = zapcore.WarnLevel
	case "error":
		zapLevel = zapcore.ErrorLevel
	default:
		parseErr = fmt.Errorf("invalid level %q", level)
	}

	zapCfg := zap.Config{
		Level:            zap.NewAtomicLevelAt(zapLevel),
		Development:      true,
		Encoding:         "console",
		EncoderConfig:    zap.NewDevelopmentEncoderConfig(),
		OutputPaths:      []string{"stdout"},
		ErrorOutputPaths: []string{"stderr"},
	}
	zapCfg.EncoderConfig.EncodeTime = zapcore.RFC3339NanoTimeEncoder
	zapLog, err := zapCfg.Build()
	if err != nil {
		zapLog, _ = zap.NewDevelopment()
	}

	log := zapr.NewLogger(zapLog)
	if parseErr != nil {
		log.WithName("setup").Info("Invalid CHREK_LOG_LEVEL, falling back to info", "value", level, "error", parseErr)
	}
	return log
}

func fatal(log logr.Logger, err error, msg string, keysAndValues ...interface{}) {
	if err != nil {
		log.Error(err, msg, keysAndValues...)
	} else {
		log.Info(msg, keysAndValues...)
	}
	os.Exit(1)
}
