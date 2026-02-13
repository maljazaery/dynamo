// Package filesystem provides rootfs introspection, diff capture/apply,
// /dev/shm capture/restore, and CRIU link_remap stub creation.
package filesystem

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/go-logr/logr"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/config"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/manifest"
)

// CaptureRootfsDiff captures the overlay upperdir to a tar file.
func CaptureRootfsDiff(upperDir, checkpointDir string, exclusions *config.FilesystemConfig, bindMountDests []string) (string, error) {
	if upperDir == "" {
		return "", fmt.Errorf("upperdir is empty")
	}

	rootfsDiffPath := filepath.Join(checkpointDir, config.RootfsDiffFilename)

	tarArgs := []string{"--xattrs"}
	if exclusions != nil {
		for _, excl := range exclusions.GetAllExclusions() {
			tarArgs = append(tarArgs, "--exclude="+excl)
		}
	}
	for _, dest := range bindMountDests {
		tarArgs = append(tarArgs, "--exclude=."+dest)
	}
	tarArgs = append(tarArgs, "-C", upperDir, "-cf", rootfsDiffPath, ".")

	cmd := exec.Command("tar", tarArgs...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("tar failed: %w (output: %s)", err, string(output))
	}

	return rootfsDiffPath, nil
}

// CaptureDeletedFiles finds whiteout files and saves them to a JSON file.
func CaptureDeletedFiles(upperDir, checkpointDir string) (bool, error) {
	if upperDir == "" {
		return false, nil
	}

	whiteouts, err := FindWhiteoutFiles(upperDir)
	if err != nil {
		return false, fmt.Errorf("failed to find whiteout files: %w", err)
	}

	if len(whiteouts) == 0 {
		return false, nil
	}

	deletedFilesPath := filepath.Join(checkpointDir, config.DeletedFilesFilename)
	data, err := json.Marshal(whiteouts)
	if err != nil {
		return false, fmt.Errorf("failed to marshal whiteouts: %w", err)
	}

	if err := os.WriteFile(deletedFilesPath, data, 0644); err != nil {
		return false, fmt.Errorf("failed to write deleted files: %w", err)
	}

	return true, nil
}

// FindWhiteoutFiles finds overlay whiteout files in the upperdir.
func FindWhiteoutFiles(upperDir string) ([]string, error) {
	var whiteouts []string

	err := filepath.Walk(upperDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		name := info.Name()
		if strings.HasPrefix(name, ".wh.") {
			relPath, _ := filepath.Rel(upperDir, path)
			dir := filepath.Dir(relPath)
			deletedFile := strings.TrimPrefix(name, ".wh.")
			deletedPath := deletedFile
			if dir != "." {
				deletedPath = filepath.Join(dir, deletedFile)
			}
			whiteouts = append(whiteouts, deletedPath)
		}
		return nil
	})

	return whiteouts, err
}

// CaptureRootfsState captures the overlay upperdir and deleted files after CRIU dump.
// Updates the checkpoint manifest with rootfs diff information.
func CaptureRootfsState(upperDir, checkpointDir string, data *manifest.CheckpointManifest, log logr.Logger) {
	if upperDir == "" || data == nil {
		return
	}

	configuredExclusions := data.Filesystem.Exclusions.GetAllExclusions()
	log.V(1).Info("Rootfs diff exclusions",
		"configured_exclusions", configuredExclusions,
		"bind_mount_exclusions", data.Filesystem.BindMountDests,
	)
	rootfsDiffPath, err := CaptureRootfsDiff(upperDir, checkpointDir, &data.Filesystem.Exclusions, data.Filesystem.BindMountDests)
	if err != nil {
		log.Error(err, "Failed to capture rootfs diff")
	} else {
		data.Filesystem.HasRootfsDiff = true
		log.Info("Captured rootfs diff", "upperdir", upperDir, "tar_path", rootfsDiffPath)
	}

	hasDeletedFiles, err := CaptureDeletedFiles(upperDir, checkpointDir)
	if err != nil {
		log.Error(err, "Failed to capture deleted files")
	} else if hasDeletedFiles {
		data.Filesystem.HasDeletedFiles = true
		log.Info("Recorded deleted files (whiteouts)")
	}

	if err := manifest.Write(checkpointDir, data); err != nil {
		log.Error(err, "Failed to update checkpoint manifest with rootfs diff info")
	}
}

// ApplyRootfsDiff extracts rootfs-diff.tar into the target root.
func ApplyRootfsDiff(checkpointPath, targetRoot string, log logr.Logger) error {
	rootfsDiffPath := filepath.Join(checkpointPath, config.RootfsDiffFilename)
	if _, err := os.Stat(rootfsDiffPath); os.IsNotExist(err) {
		log.V(1).Info("No rootfs-diff.tar, skipping")
		return nil
	}

	log.Info("Applying rootfs diff", "target", targetRoot)
	cmd := exec.Command("tar", "--keep-old-files", "-C", targetRoot, "-xf", rootfsDiffPath)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok && exitErr.ExitCode() <= 2 {
			log.V(1).Info("Rootfs diff applied (some files skipped)", "exit_code", exitErr.ExitCode())
			return nil
		}
		return fmt.Errorf("tar extract failed: %w", err)
	}
	return nil
}

// ApplyDeletedFiles removes files marked as deleted in the checkpoint.
func ApplyDeletedFiles(checkpointPath, targetRoot string, log logr.Logger) error {
	deletedFilesPath := filepath.Join(checkpointPath, config.DeletedFilesFilename)
	data, err := os.ReadFile(deletedFilesPath)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("failed to read deleted files: %w", err)
	}

	var deletedFiles []string
	if err := json.Unmarshal(data, &deletedFiles); err != nil {
		return fmt.Errorf("failed to parse deleted files: %w", err)
	}

	count := 0
	for _, f := range deletedFiles {
		if f == "" {
			continue
		}
		target := filepath.Join(targetRoot, f)
		if _, err := os.Stat(target); os.IsNotExist(err) {
			continue
		}
		if err := os.RemoveAll(target); err != nil {
			log.V(1).Info("Could not delete file", "path", target, "error", err)
			continue
		}
		count++
	}
	log.Info("Deleted files applied", "count", count)
	return nil
}
