package inspect

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/config"
)

// MountInfo holds parsed mount information from /proc/pid/mountinfo.
type MountInfo struct {
	MountID      string
	ParentID     string
	MountPoint   string
	Root         string
	FSType       string
	Source       string
	Options      string
	SuperOptions string
}

// ReadMountInfo reads and parses mountinfo for a container process via /host/proc.
func ReadMountInfo(pid int) ([]MountInfo, error) {
	mountinfoPath := fmt.Sprintf("%s/%d/mountinfo", config.HostProcPath, pid)
	return parseMountInfoFile(mountinfoPath)
}

// parseMountInfoFile parses a mountinfo file and returns all mount points.
func parseMountInfoFile(path string) ([]MountInfo, error) {
	file, err := os.Open(path)
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

// GetMountPaths returns just the mount point paths from a mountinfo file.
func GetMountPaths(path string) ([]string, error) {
	mounts, err := parseMountInfoFile(path)
	if err != nil {
		return nil, err
	}

	paths := make([]string, 0, len(mounts))
	for _, m := range mounts {
		paths = append(paths, m.MountPoint)
	}
	return paths, nil
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

	return MountInfo{
		MountID:      fields[0],
		ParentID:     fields[1],
		Root:         fields[3],
		MountPoint:   fields[4],
		Options:      fields[5],
		FSType:       fields[sepIdx+1],
		Source:       fields[sepIdx+2],
		SuperOptions: superOpts,
	}, nil
}
