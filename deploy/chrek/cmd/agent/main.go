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
	"github.com/go-logr/zapr"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/config"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/containerd"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/orchestrate"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/server"
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

	serverCfg := server.Config{
		SocketPath:     cfg.Agent.SocketPath,
		NodeName:       cfg.Agent.NodeName,
		CheckpointSpec: &cfg.Checkpoint,
	}
	srv := server.NewServer(serverCfg, checkpointer, restorer, rootLog.WithName("uds-server"))

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

	if cfg.Agent.EnableWatcher {
		watcherCfg := watcher.Config{
			NodeName:            cfg.Agent.NodeName,
			RestrictedNamespace: cfg.Agent.RestrictedNamespace,
			AgentSocketPath:     cfg.Agent.SocketPath,
			CheckpointSpec:      &cfg.Checkpoint,
		}
		podWatcher, err := watcher.NewWatcher(watcherCfg, discoveryClient, rootLog.WithName("watcher"))
		if err != nil {
			fatal(agentLog, err, "Failed to create pod watcher")
		}
		go func() {
			agentLog.Info("Pod watcher started", "label", config.KubeLabelIsCheckpointSource)
			if err := podWatcher.Start(ctx); err != nil {
				agentLog.Error(err, "Pod watcher exited")
			}
		}()
	}

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

	if err := srv.Start(); err != nil && err != http.ErrServerClosed {
		fatal(agentLog, err, "Server error")
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
