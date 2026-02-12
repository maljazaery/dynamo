// Package namespace provides Linux namespace introspection for checkpoint/restore.
package namespace

import (
	"fmt"

	"golang.org/x/sys/unix"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/config"
)

// Type represents a Linux namespace type.
type Type string

const (
	Net    Type = "net"
	PID    Type = "pid"
	Mnt    Type = "mnt"
	UTS    Type = "uts"
	IPC    Type = "ipc"
	User   Type = "user"
	Cgroup Type = "cgroup"
)

// NamespaceInfo holds namespace identification information.
type NamespaceInfo struct {
	Type       Type   `yaml:"type"`
	Inode      uint64 `yaml:"inode"`
	IsExternal bool   `yaml:"isExternal"` // Whether NS is external (shared with pause container)
}

// getNamespaceInfo returns detailed namespace information for a process.
func getNamespaceInfo(pid int, nsType Type) (*NamespaceInfo, error) {
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

// GetAll returns information about all namespaces for a process.
func GetAll(pid int) (map[Type]*NamespaceInfo, error) {
	nsTypes := []Type{Net, PID, Mnt, UTS, IPC, User, Cgroup}

	namespaces := make(map[Type]*NamespaceInfo)
	for _, nsType := range nsTypes {
		if info, err := getNamespaceInfo(pid, nsType); err == nil {
			namespaces[nsType] = info
		}
	}

	return namespaces, nil
}
