package inspect

import (
	"fmt"
	"strings"

	"golang.org/x/sys/unix"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/config"
)

// NamespaceType represents a Linux namespace type.
type NamespaceType string

const (
	NSNet    NamespaceType = "net"
	NSPID    NamespaceType = "pid"
	NSMnt    NamespaceType = "mnt"
	NSUTS    NamespaceType = "uts"
	NSIPC    NamespaceType = "ipc"
	NSUser   NamespaceType = "user"
	NSCgroup NamespaceType = "cgroup"
)

// NamespaceInfo holds namespace identification information.
type NamespaceInfo struct {
	Type       NamespaceType `yaml:"type"`
	Inode      uint64        `yaml:"inode"`
	IsExternal bool          `yaml:"isExternal"` // Whether NS is external (shared with pause container)
}

// getNamespaceInfo returns detailed namespace information for a process.
func getNamespaceInfo(pid int, nsType NamespaceType) (*NamespaceInfo, error) {
	nsPath := fmt.Sprintf("%s/%d/ns/%s", config.HostProcPath, pid, nsType)

	var stat unix.Stat_t
	if err := unix.Stat(nsPath, &stat); err != nil {
		return nil, fmt.Errorf("failed to stat namespace %s: %w", nsPath, err)
	}

	// Check if this is different from init's namespace (PID 1)
	initNsPath := fmt.Sprintf("%s/1/ns/%s", config.HostProcPath, nsType)
	var initStat unix.Stat_t
	isExternal := false
	if err := unix.Stat(initNsPath, &initStat); err == nil {
		isExternal = stat.Ino != initStat.Ino
	}

	return &NamespaceInfo{
		Type:       nsType,
		Inode:      stat.Ino,
		IsExternal: isExternal,
	}, nil
}

// GetAllNamespaces returns information about all namespaces for a process.
func GetAllNamespaces(pid int) (map[NamespaceType]*NamespaceInfo, error) {
	nsTypes := []NamespaceType{NSNet, NSPID, NSMnt, NSUTS, NSIPC, NSUser, NSCgroup}

	namespaces := make(map[NamespaceType]*NamespaceInfo)
	var errs []string
	for _, nsType := range nsTypes {
		info, err := getNamespaceInfo(pid, nsType)
		if err != nil {
			errs = append(errs, fmt.Sprintf("%s: %v", nsType, err))
			continue
		}
		namespaces[nsType] = info
	}

	if len(errs) > 0 {
		return namespaces, fmt.Errorf("failed to inspect some namespaces: %s", strings.Join(errs, "; "))
	}
	return namespaces, nil
}
