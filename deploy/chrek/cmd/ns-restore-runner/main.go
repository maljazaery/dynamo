package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/go-logr/logr"
	"github.com/go-logr/zapr"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/nsrunner"
)

func main() {
	rootLog := configureLogging()

	checkpointPath := flag.String("checkpoint-path", "", "Path to checkpoint directory")
	workDir := flag.String("work-dir", "", "CRIU work directory")
	cudaDeviceMap := flag.String("cuda-device-map", "", "CUDA device map for cuda-checkpoint restore")
	rstSibling := flag.Bool("rst-sibling", false, "Restore process as sibling (required for go-criu swrk mode)")
	mntnsCompatMode := flag.Bool("mntns-compat-mode", false, "Enable mount namespace compatibility mode")
	evasiveDevices := flag.Bool("evasive-devices", false, "Use any device path if original is inaccessible")
	forceIrmap := flag.Bool("force-irmap", false, "Force resolving inotify/fsnotify watch names")
	flag.Parse()

	log := rootLog.WithName("ns-restore-runner")
	if *checkpointPath == "" {
		fatal(log, nil, "--checkpoint-path is required")
	}

	opts := nsrunner.RestoreOptions{
		CheckpointPath:  *checkpointPath,
		WorkDir:         *workDir,
		CUDADeviceMap:   *cudaDeviceMap,
		RstSibling:      *rstSibling,
		MntnsCompatMode: *mntnsCompatMode,
		EvasiveDevices:  *evasiveDevices,
		ForceIrmap:      *forceIrmap,
	}

	restoredPID, restoredHostPID, err := nsrunner.RestoreInNamespace(context.Background(), opts, log)
	if err != nil {
		fatal(log, err, "CRIU restore failed")
	}

	fmt.Printf("RESTORED_PID=%d\n", restoredPID)
	if restoredHostPID > 0 {
		fmt.Printf("RESTORED_HOST_PID=%d\n", restoredHostPID)
	}
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
