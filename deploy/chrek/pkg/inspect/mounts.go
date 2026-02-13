package inspect

import (
	"bufio"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"

	specs "github.com/opencontainers/runtime-spec/specs-go"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/config"
)

// MountInfo holds parsed mount information from /proc/pid/mountinfo.
// MountPoint, Source, and Root are normalized (cleaned, absolute) at parse time.
type MountInfo struct {
	MountID      string
	ParentID     string
	MountPoint   string
	Root         string
	FSType       string
	Source       string
	Options      string
	SuperOptions string

	// IsOCIManaged is true when the mount destination matches an OCI spec entry
	// (including /run/ ↔ /var/run/ aliasing). Set by ClassifyMounts.
	IsOCIManaged bool

	// IsRunRuntimeMount is true for non-OCI /run/ submounts backed by runtime
	// filesystems (tmpfs, overlay) or runtime source paths. Set at parse time.
	IsRunRuntimeMount bool
}

// ReadMountInfo reads and parses mountinfo for a container process via /host/proc.
func ReadMountInfo(pid int) ([]MountInfo, error) {
	mountinfoPath := fmt.Sprintf("%s/%d/mountinfo", config.HostProcPath, pid)
	return parseMountInfoFile(mountinfoPath)
}

// ClassifyMounts sets IsOCIManaged on each mount using the container's OCI spec,
// and appends synthetic entries for OCI-managed paths not already in mountinfo.
func ClassifyMounts(mounts []MountInfo, ociSpec *specs.Spec, rootFS string) []MountInfo {
	ociSet := collectOCIManagedDestinations(ociSpec, rootFS)
	seen := make(map[string]struct{}, len(mounts))

	for i := range mounts {
		mp := mounts[i].MountPoint
		seen[mp] = struct{}{}

		if _, ok := ociSet[mp]; ok {
			mounts[i].IsOCIManaged = true
			continue
		}
		// /run/ ↔ /var/run/ aliasing
		if strings.HasPrefix(mp, "/run/") {
			if _, ok := ociSet["/var"+mp]; ok {
				mounts[i].IsOCIManaged = true
				continue
			}
		}
		if strings.HasPrefix(mp, "/var/run/") {
			if _, ok := ociSet[strings.TrimPrefix(mp, "/var")]; ok {
				mounts[i].IsOCIManaged = true
			}
		}
	}

	// Add synthetic entries for OCI-managed paths not present in mountinfo
	for p := range ociSet {
		if _, ok := seen[p]; !ok {
			mounts = append(mounts, MountInfo{MountPoint: p, IsOCIManaged: true})
		}
	}

	return mounts
}

func parseMountInfoFile(filePath string) ([]MountInfo, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open mountinfo: %w", err)
	}
	defer file.Close()

	var mounts []MountInfo
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		mount, err := parseMountInfoLine(scanner.Text())
		if err != nil {
			continue
		}
		mounts = append(mounts, mount)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading mountinfo: %w", err)
	}
	return mounts, nil
}

// parseMountInfoLine parses a single line from mountinfo.
// Format: mountID parentID major:minor root mountPoint options [optional] - fstype source superOptions
func parseMountInfoLine(line string) (MountInfo, error) {
	fields := strings.Fields(line)
	if len(fields) < 10 {
		return MountInfo{}, fmt.Errorf("malformed mountinfo line")
	}

	sepIdx := -1
	for i, f := range fields {
		if f == "-" {
			sepIdx = i
			break
		}
	}
	if sepIdx == -1 || sepIdx+2 >= len(fields) {
		return MountInfo{}, fmt.Errorf("malformed mountinfo line (no separator)")
	}

	superOpts := ""
	if sepIdx+3 < len(fields) {
		superOpts = fields[sepIdx+3]
	}

	mp := normalizeMountPath(fields[4])
	source := path.Clean(strings.TrimSpace(fields[sepIdx+2]))
	root := path.Clean(strings.TrimSpace(fields[3]))
	fsType := fields[sepIdx+1]

	isRunRuntime := strings.HasPrefix(mp, "/run/") &&
		(fsType == "tmpfs" ||
			fsType == "overlay" ||
			strings.HasPrefix(source, "/run/") ||
			strings.HasPrefix(source, "/var/run/") ||
			strings.HasPrefix(root, "/run/") ||
			strings.HasPrefix(root, "/var/run/"))

	return MountInfo{
		MountID:           fields[0],
		ParentID:          fields[1],
		Root:              root,
		MountPoint:        mp,
		Options:           fields[5],
		FSType:            fsType,
		Source:            source,
		SuperOptions:      superOpts,
		IsRunRuntimeMount: isRunRuntime,
	}, nil
}

func normalizeMountPath(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ""
	}
	p := path.Clean(raw)
	if !strings.HasPrefix(p, "/") {
		p = "/" + p
	}
	return path.Clean(p)
}

func collectOCIManagedDestinations(ociSpec *specs.Spec, rootFS string) map[string]struct{} {
	set := map[string]struct{}{}
	if ociSpec == nil {
		return set
	}

	paths := make([]string, 0, len(ociSpec.Mounts))
	for _, mount := range ociSpec.Mounts {
		paths = append(paths, mount.Destination)
	}
	if ociSpec.Linux != nil {
		paths = append(paths, ociSpec.Linux.MaskedPaths...)
		paths = append(paths, ociSpec.Linux.ReadonlyPaths...)
	}
	for _, raw := range paths {
		if p := normalizeOCIDestinationPath(raw, rootFS); p != "" {
			set[p] = struct{}{}
		}
	}
	return set
}

func normalizeOCIDestinationPath(raw, rootFS string) string {
	p := normalizeMountPath(raw)
	if p == "" || rootFS == "" {
		return p
	}

	hostPath := filepath.Join(rootFS, strings.TrimPrefix(p, "/"))
	resolved, err := filepath.EvalSymlinks(hostPath)
	if err != nil {
		return p
	}

	rel, err := filepath.Rel(rootFS, resolved)
	if err != nil {
		return p
	}
	rel = filepath.ToSlash(rel)
	if rel == "." {
		return "/"
	}
	if strings.HasPrefix(rel, "../") || rel == ".." {
		return p
	}

	return normalizeMountPath("/" + rel)
}
