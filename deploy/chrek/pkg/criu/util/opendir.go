// Package criu provides CRIU dump and restore operations, configuration
// generation, and FD management for go-criu integration.
package criu

import (
	"fmt"
	"os"

	"golang.org/x/sys/unix"
)

// OpenPathForCRIU opens a path (directory or file) and clears the CLOEXEC flag
// so the FD can be inherited by CRIU child processes.
// Returns the opened file and its FD. Caller must close the file when done.
func OpenPathForCRIU(path string) (*os.File, int32, error) {
	dir, err := os.Open(path)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to open %s: %w", path, err)
	}

	// Clear CLOEXEC so the FD is inherited by CRIU child process.
	// Go's os.Open() sets O_CLOEXEC by default, but go-criu's swrk mode
	// requires the FD to be inherited.
	if _, err := unix.FcntlInt(dir.Fd(), unix.F_SETFD, 0); err != nil {
		dir.Close()
		return nil, 0, fmt.Errorf("failed to clear CLOEXEC on %s: %w", path, err)
	}

	return dir, int32(dir.Fd()), nil
}
