package filesystem

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"syscall"

	"github.com/go-logr/logr"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/config"
)

// CaptureDevShm captures files from /dev/shm to the checkpoint directory.
func CaptureDevShm(pid int, checkpointDir string, log logr.Logger) error {
	shmPath := filepath.Join(config.HostProcPath, fmt.Sprintf("%d/root/dev/shm", pid))

	entries, err := os.ReadDir(shmPath)
	if err != nil {
		if os.IsNotExist(err) {
			log.V(1).Info("Container /dev/shm does not exist, skipping capture")
			return nil
		}
		return fmt.Errorf("failed to read container /dev/shm: %w", err)
	}

	var filesToCapture []os.DirEntry
	for _, entry := range entries {
		if entry.IsDir() {
			log.V(1).Info("Skipping directory in /dev/shm", "dir", entry.Name())
			continue
		}
		filesToCapture = append(filesToCapture, entry)
	}

	if len(filesToCapture) == 0 {
		log.V(1).Info("No files to capture from /dev/shm")
		return nil
	}

	destDir := filepath.Join(checkpointDir, config.DevShmDirName)
	if err := os.MkdirAll(destDir, 0755); err != nil {
		return fmt.Errorf("failed to create dev-shm directory: %w", err)
	}

	var captured []string
	var totalSize int64

	for _, entry := range filesToCapture {
		name := entry.Name()
		srcPath := filepath.Join(shmPath, name)
		destPath := filepath.Join(destDir, name)

		info, err := entry.Info()
		if err != nil {
			log.Error(err, "Failed to get file info, skipping", "file", name)
			continue
		}

		size := info.Size()
		uid, gid := -1, -1
		if stat, ok := info.Sys().(*syscall.Stat_t); ok {
			uid = int(stat.Uid)
			gid = int(stat.Gid)
		}

		if err := copyFile(srcPath, destPath, info.Mode(), uid, gid); err != nil {
			log.Error(err, "Failed to copy file, skipping", "file", name)
			continue
		}

		captured = append(captured, name)
		totalSize += size
		log.V(1).Info("Captured /dev/shm file", "file", name, "size", size)
	}

	if len(captured) > 0 {
		log.Info("Captured /dev/shm files", "count", len(captured), "total_size", totalSize, "files", captured)
	}

	return nil
}

// RestoreDevShm restores /dev/shm files into the target root's /dev/shm.
func RestoreDevShm(checkpointPath, targetRoot string, log logr.Logger) error {
	srcDir := filepath.Join(checkpointPath, config.DevShmDirName)
	entries, err := os.ReadDir(srcDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("failed to read dev-shm dir: %w", err)
	}

	destDir := filepath.Join(targetRoot, "dev", "shm")
	if err := os.MkdirAll(destDir, 0777); err != nil {
		return fmt.Errorf("failed to create target /dev/shm: %w", err)
	}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		srcPath := filepath.Join(srcDir, entry.Name())
		destPath := filepath.Join(destDir, entry.Name())

		info, err := entry.Info()
		if err != nil {
			continue
		}
		uid, gid := -1, -1
		if stat, ok := info.Sys().(*syscall.Stat_t); ok {
			uid = int(stat.Uid)
			gid = int(stat.Gid)
		}

		src, err := os.Open(srcPath)
		if err != nil {
			continue
		}

		mode := info.Mode()
		if mode == 0 {
			mode = 0666
		}
		dst, err := os.OpenFile(destPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode)
		if err != nil {
			src.Close()
			continue
		}

		if _, err := io.Copy(dst, src); err != nil {
			src.Close()
			dst.Close()
			continue
		}
		if uid >= 0 && gid >= 0 {
			if err := dst.Chown(uid, gid); err != nil {
				src.Close()
				dst.Close()
				continue
			}
		}
		src.Close()
		dst.Close()
	}

	log.V(1).Info("Restored /dev/shm files", "count", len(entries))
	return nil
}

func copyFile(src, dest string, mode os.FileMode, uid, gid int) error {
	srcFile, err := os.Open(src)
	if err != nil {
		return fmt.Errorf("failed to open source: %w", err)
	}
	defer srcFile.Close()

	destFile, err := os.OpenFile(dest, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode)
	if err != nil {
		return fmt.Errorf("failed to create destination: %w", err)
	}
	defer destFile.Close()

	if _, err := io.Copy(destFile, srcFile); err != nil {
		return fmt.Errorf("failed to copy contents: %w", err)
	}

	if uid >= 0 && gid >= 0 {
		if err := destFile.Chown(uid, gid); err != nil {
			return fmt.Errorf("failed to set ownership to %d:%d: %w", uid, gid, err)
		}
	}

	if err := destFile.Sync(); err != nil {
		return fmt.Errorf("failed to sync destination: %w", err)
	}

	return nil
}
