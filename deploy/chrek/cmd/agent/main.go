// Package main provides the chrek DaemonSet agent.
// The agent runs a UDS HTTP server for checkpoint/restore operations and
// optionally watches pods for automatic checkpointing.
package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/go-logr/logr"
	gozap "go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	ctrl "sigs.k8s.io/controller-runtime"
	ctrlzap "sigs.k8s.io/controller-runtime/pkg/log/zap"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/externalrestore"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/watcher"
)

func main() {
	configureLogging()
	agentLog := ctrl.Log.WithName("agent")

	cfg, err := LoadConfigOrDefault(ConfigMapPath)
	if err != nil {
		fatal(agentLog, err, "Failed to load configuration")
	}
	if err := cfg.Validate(); err != nil {
		fatal(agentLog, err, "Invalid configuration")
	}

	discoveryClient, err := checkpoint.NewDiscoveryClient()
	if err != nil {
		fatal(agentLog, err, "Failed to create discovery client")
	}
	defer discoveryClient.Close()

	checkpointer := checkpoint.NewCheckpointer(discoveryClient)

	// Create the external restorer
	restorer := externalrestore.NewRestorer(
		externalrestore.RestorerConfig{
			CheckpointBasePath: cfg.Checkpoint.BasePath,
		},
		discoveryClient,
	)

	// Create UDS server
	serverCfg := externalrestore.ServerConfig{
		SocketPath:     cfg.Agent.SocketPath,
		NodeName:       cfg.Agent.NodeName,
		CheckpointSpec: &cfg.Checkpoint,
		CRIUTimeout:    cfg.Checkpoint.CRIU.Timeout,
	}
	srv := externalrestore.NewServer(serverCfg, checkpointer, restorer)

	// Context for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	agentLog.Info("Starting chrek agent",
		"node", cfg.Agent.NodeName,
		"checkpoint_dir", cfg.Checkpoint.BasePath,
		"socket_path", cfg.Agent.SocketPath,
		"watcher_enabled", cfg.Agent.EnableWatcher,
		"watch_namespace", cfg.Agent.RestrictedNamespace,
	)

	// Start optional watcher alongside UDS server
	if cfg.Agent.EnableWatcher {
		watcherCfg := watcher.WatcherConfig{
			NodeName:            cfg.Agent.NodeName,
			RestrictedNamespace: cfg.Agent.RestrictedNamespace,
			AgentSocketPath:     cfg.Agent.SocketPath,
			CheckpointSpec:      &cfg.Checkpoint,
		}
		podWatcher, err := watcher.NewWatcher(watcherCfg, discoveryClient)
		if err != nil {
			fatal(agentLog, err, "Failed to create pod watcher")
		}
		go func() {
			agentLog.Info("Pod watcher started", "label", checkpoint.KubeLabelCheckpointSource)
			if err := podWatcher.Start(ctx); err != nil {
				agentLog.Error(err, "Pod watcher exited")
			}
		}()
	}

	// Handle graceful shutdown
	go func() {
		<-sigChan
		agentLog.Info("Shutting down")
		cancel()
		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer shutdownCancel()
		if err := srv.Shutdown(shutdownCtx); err != nil {
			agentLog.Error(err, "Server shutdown error")
		}
	}()

	// Start UDS server (blocks until shutdown)
	if err := srv.Start(); err != nil && err != http.ErrServerClosed {
		fatal(agentLog, err, "Server error")
	}

	agentLog.Info("Agent stopped")
}

func configureLogging() {
	level := strings.TrimSpace(strings.ToLower(os.Getenv("CHREK_LOG_LEVEL")))
	if level == "" {
		level = "info"
	}

	atomicLevel := gozap.NewAtomicLevelAt(zapcore.InfoLevel)
	parseErr := error(nil)
	switch level {
	case "trace", "debug":
		atomicLevel = gozap.NewAtomicLevelAt(zapcore.DebugLevel)
	case "info":
		atomicLevel = gozap.NewAtomicLevelAt(zapcore.InfoLevel)
	case "warn", "warning":
		atomicLevel = gozap.NewAtomicLevelAt(zapcore.WarnLevel)
	case "error":
		atomicLevel = gozap.NewAtomicLevelAt(zapcore.ErrorLevel)
	default:
		parseErr = fmt.Errorf("invalid level %q", level)
	}
	if parseErr != nil {
		atomicLevel = gozap.NewAtomicLevelAt(zapcore.InfoLevel)
	}

	opts := ctrlzap.Options{
		Development: true,
		Level:       &atomicLevel,
		TimeEncoder: zapcore.RFC3339NanoTimeEncoder,
	}
	ctrl.SetLogger(ctrlzap.New(ctrlzap.UseFlagOptions(&opts), ctrlzap.WriteTo(os.Stdout)))

	if parseErr != nil {
		ctrl.Log.WithName("setup").Info("Invalid CHREK_LOG_LEVEL, falling back to info", "value", level, "error", parseErr)
	}
}

func fatal(log logr.Logger, err error, msg string, keysAndValues ...interface{}) {
	if err != nil {
		log.Error(err, msg, keysAndValues...)
	} else {
		log.Info(msg, keysAndValues...)
	}
	os.Exit(1)
}
