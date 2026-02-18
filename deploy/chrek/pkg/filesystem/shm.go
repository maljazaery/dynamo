package filesystem

import (
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"syscall"

	"github.com/go-logr/logr"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/config"
)

// CaptureDevShm recursively captures files and directories from /dev/shm to the checkpoint directory.
func CaptureDevShm(pid int, checkpointDir string, log logr.Logger) error {
	shmPath := filepath.Join(config.HostProcPath, fmt.Sprintf("%d/root/dev/shm", pid))

	if _, err := os.Stat(shmPath); os.IsNotExist(err) {
		log.V(1).Info("Container /dev/shm does not exist, skipping capture")
		return nil
	}

	destDir := filepath.Join(checkpointDir, config.DevShmDirName)
	if err := os.MkdirAll(destDir, 0755); err != nil {
		return fmt.Errorf("failed to create dev-shm directory: %w", err)
	}

	var captured int
	var totalSize int64

	err := filepath.WalkDir(shmPath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			log.Error(err, "Failed to access path, skipping", "path", path)
			return nil
		}

		rel, err := filepath.Rel(shmPath, path)
		if err != nil {
			return err
		}
		if rel == "." {
			return nil
		}
		destPath := filepath.Join(destDir, rel)

		info, err := d.Info()
		if err != nil {
			log.Error(err, "Failed to get file info, skipping", "path", rel)
			return nil
		}

		uid, gid := -1, -1
		if stat, ok := info.Sys().(*syscall.Stat_t); ok {
			uid = int(stat.Uid)
			gid = int(stat.Gid)
		}

		if d.IsDir() {
			if err := os.MkdirAll(destPath, info.Mode()); err != nil {
				log.Error(err, "Failed to create directory, skipping", "path", rel)
				return filepath.SkipDir
			}
			if uid >= 0 && gid >= 0 {
				os.Chown(destPath, uid, gid)
			}
			return nil
		}

		if err := copyFile(path, destPath, info.Mode(), uid, gid); err != nil {
			log.Error(err, "Failed to copy file, skipping", "file", rel)
			return nil
		}

		captured++
		totalSize += info.Size()
		log.V(1).Info("Captured /dev/shm file", "file", rel, "size", info.Size())
		return nil
	})
	if err != nil {
		return fmt.Errorf("failed to walk /dev/shm: %w", err)
	}

	if captured > 0 {
		log.Info("Captured /dev/shm contents", "file_count", captured, "total_size", totalSize)
	}

	return nil
}

// RestoreDevShm recursively restores /dev/shm contents into the target root's /dev/shm.
func RestoreDevShm(checkpointPath, targetRoot string, log logr.Logger) error {
	srcDir := filepath.Join(checkpointPath, config.DevShmDirName)
	if _, err := os.Stat(srcDir); os.IsNotExist(err) {
		return nil
	}

	destDir := filepath.Join(targetRoot, "dev", "shm")
	if err := os.MkdirAll(destDir, 0777); err != nil {
		return fmt.Errorf("failed to create target /dev/shm: %w", err)
	}

	restored := 0
	err := filepath.WalkDir(srcDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			log.Error(err, "Failed to access path, skipping", "path", path)
			return nil
		}

		rel, err := filepath.Rel(srcDir, path)
		if err != nil {
			return err
		}
		if rel == "." {
			return nil
		}
		destPath := filepath.Join(destDir, rel)

		info, err := d.Info()
		if err != nil {
			log.Error(err, "Failed to get file info, skipping", "path", rel)
			return nil
		}

		uid, gid := -1, -1
		if stat, ok := info.Sys().(*syscall.Stat_t); ok {
			uid = int(stat.Uid)
			gid = int(stat.Gid)
		}

		if d.IsDir() {
			if err := os.MkdirAll(destPath, info.Mode()); err != nil {
				log.Error(err, "Failed to create directory, skipping", "path", rel)
				return filepath.SkipDir
			}
			if uid >= 0 && gid >= 0 {
				os.Chown(destPath, uid, gid)
			}
			return nil
		}

		mode := info.Mode()
		if mode == 0 {
			mode = 0666
		}
		if err := copyFile(path, destPath, mode, uid, gid); err != nil {
			log.Error(err, "Failed to restore /dev/shm file, skipping", "file", rel)
			return nil
		}
		restored++
		return nil
	})
	if err != nil {
		return fmt.Errorf("failed to walk dev-shm dir: %w", err)
	}

	log.V(1).Info("Restored /dev/shm contents", "file_count", restored)
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
