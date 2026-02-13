// Package main provides the chrek DaemonSet agent.
// The agent watches for pods with checkpoint/restore labels on its node
// and triggers operations via the orchestrators.
package main

import (
	"context"
	"os"
	"os/signal"
	"syscall"

	"github.com/go-logr/logr"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/config"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/inspect"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/logging"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/orchestrate"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/watcher"
)

func main() {
	rootLog := logging.ConfigureLogger("stdout")
	agentLog := rootLog.WithName("agent")

	cfg, err := LoadConfigOrDefault(ConfigMapPath)
	if err != nil {
		fatal(agentLog, err, "Failed to load configuration")
	}
	if err := cfg.Validate(); err != nil {
		fatal(agentLog, err, "Invalid configuration")
	}

	discoveryClient, err := inspect.NewClient()
	if err != nil {
		fatal(agentLog, err, "Failed to create discovery client")
	}
	defer discoveryClient.Close()

	checkpointer := orchestrate.NewCheckpointer(discoveryClient, rootLog.WithName("checkpointer"))

	restorer := orchestrate.NewRestorer(
		orchestrate.RestorerConfig{
			CheckpointBasePath: cfg.Checkpoint.BasePath,
			NSRestorePath:      cfg.Checkpoint.NSRestorePath,
		},
		discoveryClient,
		rootLog.WithName("restorer"),
	)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	agentLog.Info("Starting chrek agent",
		"node", cfg.NodeName,
		"checkpoint_dir", cfg.Checkpoint.BasePath,
		"watch_namespace", cfg.RestrictedNamespace,
	)

	podWatcher, err := watcher.NewWatcher(cfg, checkpointer, restorer, discoveryClient, rootLog.WithName("watcher"))
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
		select {
		case err := <-watcherDone:
			if err != nil {
				agentLog.Error(err, "Pod watcher exited with error during shutdown")
			}
		default:
		}
	case err := <-watcherDone:
		if err != nil {
			fatal(agentLog, err, "Pod watcher exited with error")
		}
	}

	agentLog.Info("Agent stopped")
}

func fatal(log logr.Logger, err error, msg string, keysAndValues ...interface{}) {
	if err != nil {
		log.Error(err, msg, keysAndValues...)
	} else {
		log.Info(msg, keysAndValues...)
	}
	os.Exit(1)
}
