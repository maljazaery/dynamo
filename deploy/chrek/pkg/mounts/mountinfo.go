// Package mounts provides mountinfo parsing and mount policy classification
// for CRIU checkpoint and restore.
package mounts

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/config"
)

// Info holds parsed mount information from /proc/pid/mountinfo.
type Info struct {
	MountID      string
	ParentID     string
	MountPoint   string
	Root         string
	FSType       string
	Source        string
	Options      string
	SuperOptions string
}

// ReadFromHostProc reads and parses mountinfo for a container process via /host/proc.
func ReadFromHostProc(pid int) ([]Info, error) {
	mountinfoPath := fmt.Sprintf("%s/%d/mountinfo", config.HostProcPath, pid)
	return ParseFile(mountinfoPath)
}

// ParseFile parses a mountinfo file and returns all mount points.
func ParseFile(path string) ([]Info, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open mountinfo: %w", err)
	}
	defer file.Close()

	var mounts []Info
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		mount, err := parseLine(scanner.Text())
		if err != nil {
			continue // Skip malformed lines
		}
		mounts = append(mounts, mount)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading mountinfo: %w", err)
	}

	return mounts, nil
}

// GetPaths returns just the mount point paths from a mountinfo file.
func GetPaths(path string) ([]string, error) {
	mounts, err := ParseFile(path)
	if err != nil {
		return nil, err
	}

	paths := make([]string, 0, len(mounts))
	for _, m := range mounts {
		paths = append(paths, m.MountPoint)
	}
	return paths, nil
}

// parseLine parses a single line from mountinfo.
// Format: mountID parentID major:minor root mountPoint options [optional] - fstype source superOptions
func parseLine(line string) (Info, error) {
	fields := strings.Fields(line)
	if len(fields) < 10 {
		return Info{}, fmt.Errorf("malformed mountinfo line")
	}

	// Find separator (-) to get fstype and source
	sepIdx := -1
	for i, f := range fields {
		if f == "-" {
			sepIdx = i
			break
		}
	}

	if sepIdx == -1 || sepIdx+2 >= len(fields) {
		return Info{}, fmt.Errorf("malformed mountinfo line (no separator)")
	}

	superOpts := ""
	if sepIdx+3 < len(fields) {
		superOpts = fields[sepIdx+3]
	}

	return Info{
		MountID:      fields[0],
		ParentID:     fields[1],
		Root:         fields[3],
		MountPoint:   fields[4],
		Options:      fields[5],
		FSType:       fields[sepIdx+1],
		Source:        fields[sepIdx+2],
		SuperOptions: superOpts,
	}, nil
}
