package filesystem

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/go-logr/logr"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/criu"
)

// CreateLinkRemapStubs creates placeholder files for CRIU link_remap entries.
// These stubs must exist in the target rootfs before CRIU restore.
func CreateLinkRemapStubs(checkpointPath, targetRoot string, log logr.Logger) error {
	remapPath := filepath.Join(checkpointPath, "remap-fpath.img")
	remaps, err := criu.ParseRemapFilePath(remapPath)
	if err != nil {
		if os.IsNotExist(err) {
			log.V(1).Info("No remap-fpath.img found, no link_remap stubs needed")
			return nil
		}
		return fmt.Errorf("failed to parse remap-fpath.img: %w", err)
	}
	if len(remaps) == 0 {
		return nil
	}

	regFilesPath := filepath.Join(checkpointPath, "reg-files.img")
	filesPath := filepath.Join(checkpointPath, "files.img")

	fileMap, parseErr := criu.ParseRegFilesWithMode(regFilesPath)
	if parseErr != nil {
		fileMap, parseErr = criu.ParseFilesWithMode(filesPath)
		if parseErr != nil {
			return fmt.Errorf("failed to parse files metadata for remap stubs: %w", parseErr)
		}
	}

	created := 0
	for _, remap := range remaps {
		origInfo, ok := fileMap[remap.OrigID]
		if !ok {
			continue
		}

		remapPathInContainer := ""
		remapMode := origInfo.Mode
		if remapInfo, ok := fileMap[remap.RemapID]; ok {
			remapPathInContainer = remapInfo.Name
			remapMode = remapInfo.Mode
		} else {
			dir := filepath.Dir(origInfo.Name)
			if !strings.HasPrefix(dir, "/") {
				dir = "/" + dir
			}
			remapPathInContainer = filepath.Join(dir, fmt.Sprintf("link_remap.%d", remap.RemapID))
		}

		if !strings.HasPrefix(remapPathInContainer, "/") {
			remapPathInContainer = "/" + remapPathInContainer
		}

		hostPath := filepath.Join(targetRoot, strings.TrimPrefix(filepath.Clean(remapPathInContainer), "/"))
		if _, err := os.Stat(hostPath); err == nil {
			continue
		}

		if err := createStub(hostPath, remapMode); err != nil {
			log.Error(err, "Failed to create link_remap stub", "stub", hostPath, "target", origInfo.Name)
			continue
		}
		created++
	}

	if created > 0 {
		log.Info("Created link_remap stubs", "count", created)
	}
	return nil
}

func createStub(path string, mode os.FileMode) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write(make([]byte, 32))
	return err
}
