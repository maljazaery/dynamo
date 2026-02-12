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

// Info holds namespace identification information.
type Info struct {
	Type       Type
	Inode      uint64
	IsExternal bool // Whether NS is external (shared with pause container)
}

// GetInfo returns detailed namespace information for a process.
func GetInfo(pid int, nsType Type) (*Info, error) {
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

	return &Info{
		Type:       nsType,
		Inode:      stat.Ino,
		IsExternal: isExternal,
	}, nil
}

// GetAll returns information about all namespaces for a process.
func GetAll(pid int) (map[Type]*Info, error) {
	nsTypes := []Type{Net, PID, Mnt, UTS, IPC, User, Cgroup}

	namespaces := make(map[Type]*Info)
	for _, nsType := range nsTypes {
		if info, err := GetInfo(pid, nsType); err == nil {
			namespaces[nsType] = info
		}
	}

	return namespaces, nil
}
