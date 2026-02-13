package inspect

import (
	"path"
	"path/filepath"
	"strings"

	specs "github.com/opencontainers/runtime-spec/specs-go"
)

// MountPolicy is the classified mount plan for CRIU dump options.
type MountPolicy struct {
	Externalized []string
	Skipped      []string
}

// BuildMountPolicy classifies mounts into CRIU extMnt and skipMnt lists.
//
// Rule order and precedence (top to bottom):
//  1. Skip non-OCI proc/sys submounts and non-OCI runtime /run submounts.
//  2. Externalize everything else.
//
// Precedence: skip > externalize.
func BuildMountPolicy(mountInfo []MountInfo, ociSpec *specs.Spec, rootFS string) *MountPolicy {
	ociManagedSet := collectOCIManagedDestinations(ociSpec, rootFS)

	externalizedSet := make(map[string]struct{}, len(mountInfo)+len(ociManagedSet))
	skippedSet := make(map[string]struct{}, len(mountInfo))

	for _, mount := range mountInfo {
		mp := normalizeMountPath(mount.MountPoint)
		if mp == "" {
			continue
		}

		source := path.Clean(strings.TrimSpace(mount.Source))
		root := path.Clean(strings.TrimSpace(mount.Root))
		isOCIManaged := false
		if _, ok := ociManagedSet[mp]; ok {
			isOCIManaged = true
		}
		if !isOCIManaged && strings.HasPrefix(mp, "/run/") {
			if _, ok := ociManagedSet["/var"+mp]; ok {
				isOCIManaged = true
			}
		}
		if !isOCIManaged && strings.HasPrefix(mp, "/var/run/") {
			if _, ok := ociManagedSet[strings.TrimPrefix(mp, "/var")]; ok {
				isOCIManaged = true
			}
		}

		isRunRuntimeMount := strings.HasPrefix(mp, "/run/") &&
			(mount.FSType == "tmpfs" ||
				mount.FSType == "overlay" ||
				strings.HasPrefix(source, "/run/") ||
				strings.HasPrefix(source, "/var/run/") ||
				strings.HasPrefix(root, "/run/") ||
				strings.HasPrefix(root, "/var/run/"))

		if !isOCIManaged && (strings.HasPrefix(mp, "/proc/") || strings.HasPrefix(mp, "/sys/") || isRunRuntimeMount) {
			skippedSet[mp] = struct{}{}
			continue
		}

		externalizedSet[mp] = struct{}{}
	}

	for mp := range ociManagedSet {
		if _, skipped := skippedSet[mp]; skipped {
			continue
		}
		externalizedSet[mp] = struct{}{}
	}

	externalized := make([]string, 0, len(externalizedSet))
	for mp := range externalizedSet {
		externalized = append(externalized, mp)
	}
	skipped := make([]string, 0, len(skippedSet))
	for mp := range skippedSet {
		skipped = append(skipped, mp)
	}

	return &MountPolicy{
		Externalized: externalized,
		Skipped:      skipped,
	}
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
