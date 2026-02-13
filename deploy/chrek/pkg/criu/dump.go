package criu

import (
	"fmt"
	"strings"
	"time"

	criulib "github.com/checkpoint-restore/go-criu/v8"
	criurpc "github.com/checkpoint-restore/go-criu/v8/rpc"
	"github.com/go-logr/logr"
	specs "github.com/opencontainers/runtime-spec/specs-go"
	"google.golang.org/protobuf/proto"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/config"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/inspect"
)

// GenerateCRIUConfContent generates the criu.conf file content for options
// that cannot be passed via RPC.
func GenerateCRIUConfContent(c *config.CRIUSettings) string {
	var content string

	if c.LibDir != "" {
		content += "libdir " + c.LibDir + "\n"
	}
	if c.AllowUprobes {
		content += "allow-uprobes\n"
	}
	if c.SkipInFlight {
		content += "skip-in-flight\n"
	}

	return content
}

// BuildDumpOptions creates CRIU options directly from spec settings and runtime state.
func BuildDumpOptions(
	settings *config.CRIUSettings,
	pid int,
	imageDirFD int32,
	rootFS string,
	mountInfo []inspect.MountInfo,
	ociSpec *specs.Spec,
	namespaces map[inspect.NamespaceType]*inspect.NamespaceInfo,
	log logr.Logger,
) (*criurpc.CriuOpts, error) {
	mountPolicy := inspect.BuildMountPolicy(mountInfo, ociSpec, rootFS)

	extMnt := buildExternalMountMaps(mountPolicy.Externalized)
	skipMnt := mountPolicy.Skipped
	external := buildExternalNamespaces(namespaces, log)
	log.V(1).Info("Resolved mount policy for CRIU dump",
		"externalized_count", len(mountPolicy.Externalized),
		"skipped_count", len(mountPolicy.Skipped),
	)

	criuOpts := &criurpc.CriuOpts{
		Pid:         proto.Int32(int32(pid)),
		ImagesDirFd: proto.Int32(imageDirFD),
		Root:        proto.String(rootFS),
		LogFile:     proto.String(config.DumpLogFilename),
	}
	criuOpts.ExtMnt = extMnt
	criuOpts.External = external
	criuOpts.SkipMnt = skipMnt

	if settings == nil {
		return criuOpts, nil
	}

	// RPC options from spec
	criuOpts.LogLevel = proto.Int32(settings.LogLevel)
	criuOpts.LeaveRunning = proto.Bool(settings.LeaveRunning)
	criuOpts.ShellJob = proto.Bool(settings.ShellJob)
	criuOpts.TcpClose = proto.Bool(settings.TcpClose)
	criuOpts.FileLocks = proto.Bool(settings.FileLocks)
	criuOpts.OrphanPtsMaster = proto.Bool(settings.OrphanPtsMaster)
	criuOpts.ExtUnixSk = proto.Bool(settings.ExtUnixSk)
	criuOpts.LinkRemap = proto.Bool(settings.LinkRemap)
	criuOpts.ExtMasters = proto.Bool(settings.ExtMasters)
	criuOpts.AutoDedup = proto.Bool(settings.AutoDedup)
	criuOpts.LazyPages = proto.Bool(settings.LazyPages)

	// Cgroup management mode
	criuOpts.ManageCgroups = proto.Bool(true)
	cgMode := criurpc.CriuCgMode_IGNORE
	switch strings.ToLower(strings.TrimSpace(settings.ManageCgroupsMode)) {
	case "soft":
		cgMode = criurpc.CriuCgMode_SOFT
	case "full":
		cgMode = criurpc.CriuCgMode_FULL
	case "strict":
		cgMode = criurpc.CriuCgMode_STRICT
	}
	criuOpts.ManageCgroupsMode = &cgMode

	if settings.GhostLimit > 0 {
		criuOpts.GhostLimit = proto.Uint32(settings.GhostLimit)
	}

	return criuOpts, nil
}

// ExecuteDump runs the CRIU dump and logs timing plus dump-log location on failure.
func ExecuteDump(criuOpts *criurpc.CriuOpts, checkpointDir string, log logr.Logger) (time.Duration, error) {
	criuDumpStart := time.Now()
	criuClient := criulib.MakeCriu()
	if err := criuClient.Dump(criuOpts, nil); err != nil {
		dumpDuration := time.Since(criuDumpStart)
		log.Error(err, "CRIU dump failed",
			"duration", dumpDuration,
			"checkpoint_dir", checkpointDir,
			"dump_log_path", fmt.Sprintf("%s/%s", checkpointDir, config.DumpLogFilename),
		)
		return 0, fmt.Errorf("CRIU dump failed: %w", err)
	}

	criuDumpDuration := time.Since(criuDumpStart)
	log.Info("CRIU dump completed", "duration", criuDumpDuration)
	return criuDumpDuration, nil
}

func buildExternalNamespaces(namespaces map[inspect.NamespaceType]*inspect.NamespaceInfo, log logr.Logger) []string {
	external := make([]string, 0, 1)

	if netNs, ok := namespaces[inspect.NSNet]; ok {
		external = append(external, fmt.Sprintf("%s[%d]:%s", inspect.NSNet, netNs.Inode, "extNetNs"))
		log.V(1).Info("Marked network namespace as external", "inode", netNs.Inode)
	}

	return external
}

func buildExternalMountMaps(paths []string) []*criurpc.ExtMountMap {
	extMnt := make([]*criurpc.ExtMountMap, 0, len(paths))
	existing := make(map[string]struct{}, len(paths))
	for _, path := range paths {
		if path == "" {
			continue
		}
		if _, ok := existing[path]; ok {
			continue
		}
		extMnt = append(extMnt, &criurpc.ExtMountMap{
			Key: proto.String(path),
			Val: proto.String(path),
		})
		existing[path] = struct{}{}
	}
	return extMnt
}
