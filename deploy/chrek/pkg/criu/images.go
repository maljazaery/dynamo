package criu

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"

	"github.com/checkpoint-restore/go-criu/v8/crit"
	"github.com/checkpoint-restore/go-criu/v8/crit/images/fdinfo"
	"github.com/checkpoint-restore/go-criu/v8/crit/images/regfile"
	remapfile "github.com/checkpoint-restore/go-criu/v8/crit/images/remap-file-path"
	"google.golang.org/protobuf/proto"
)

type RemapFilePathEntry struct {
	OrigID  uint32
	RemapID uint32
}

type FileMetadata struct {
	Name string
	Mode os.FileMode
}

func ParseRemapFilePath(path string) ([]RemapFilePathEntry, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	magic, err := crit.ReadMagic(f)
	if err != nil {
		return nil, fmt.Errorf("failed to read remap image magic: %w", err)
	}
	if magic != "REMAP_FPATH" {
		return nil, fmt.Errorf("unexpected remap image magic: %s", magic)
	}

	sizeBuf := make([]byte, 4)
	var entries []RemapFilePathEntry
	for {
		_, err := io.ReadFull(f, sizeBuf)
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("failed to read remap entry size: %w", err)
		}

		entrySize := binary.LittleEndian.Uint32(sizeBuf)
		entryBuf := make([]byte, entrySize)
		if _, err := io.ReadFull(f, entryBuf); err != nil {
			return nil, fmt.Errorf("failed to read remap entry: %w", err)
		}

		entry := &remapfile.RemapFilePathEntry{}
		if err := proto.Unmarshal(entryBuf, entry); err != nil {
			return nil, fmt.Errorf("failed to decode remap entry: %w", err)
		}
		entries = append(entries, RemapFilePathEntry{
			OrigID:  entry.GetOrigId(),
			RemapID: entry.GetRemapId(),
		})
	}

	return entries, nil
}

func ParseRegFilesWithMode(path string) (map[uint32]FileMetadata, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	magic, err := crit.ReadMagic(f)
	if err != nil {
		return nil, fmt.Errorf("failed to read reg-files image magic: %w", err)
	}
	if magic != "REG_FILES" {
		return nil, fmt.Errorf("unexpected reg-files image magic: %s", magic)
	}

	sizeBuf := make([]byte, 4)
	fileMap := make(map[uint32]FileMetadata)
	for {
		_, err := io.ReadFull(f, sizeBuf)
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("failed to read reg-files entry size: %w", err)
		}

		entrySize := binary.LittleEndian.Uint32(sizeBuf)
		entryBuf := make([]byte, entrySize)
		if _, err := io.ReadFull(f, entryBuf); err != nil {
			return nil, fmt.Errorf("failed to read reg-files entry: %w", err)
		}

		entry := &regfile.RegFileEntry{}
		if err := proto.Unmarshal(entryBuf, entry); err != nil {
			return nil, fmt.Errorf("failed to decode reg-files entry: %w", err)
		}

		mode := os.FileMode(entry.GetMode() & 0777)
		if mode == 0 {
			mode = 0600
		}
		fileMap[entry.GetId()] = FileMetadata{
			Name: entry.GetName(),
			Mode: mode,
		}
	}

	return fileMap, nil
}

func ParseFilesWithMode(path string) (map[uint32]FileMetadata, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	magic, err := crit.ReadMagic(f)
	if err != nil {
		return nil, fmt.Errorf("failed to read files image magic: %w", err)
	}
	if magic != "FILES" {
		return nil, fmt.Errorf("unexpected files image magic: %s", magic)
	}

	sizeBuf := make([]byte, 4)
	fileMap := make(map[uint32]FileMetadata)
	for {
		_, err := io.ReadFull(f, sizeBuf)
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("failed to read files entry size: %w", err)
		}

		entrySize := binary.LittleEndian.Uint32(sizeBuf)
		entryBuf := make([]byte, entrySize)
		if _, err := io.ReadFull(f, entryBuf); err != nil {
			return nil, fmt.Errorf("failed to read files entry: %w", err)
		}

		entry := &fdinfo.FileEntry{}
		if err := proto.Unmarshal(entryBuf, entry); err != nil {
			return nil, fmt.Errorf("failed to decode files entry: %w", err)
		}
		if entry.GetReg() == nil {
			continue
		}

		mode := os.FileMode(entry.GetReg().GetMode() & 0777)
		if mode == 0 {
			mode = 0600
		}
		fileMap[entry.GetId()] = FileMetadata{
			Name: entry.GetReg().GetName(),
			Mode: mode,
		}
	}

	return fileMap, nil
}

func DiscoverStdioInheritResources(checkpointPath string) ([]string, []string, error) {
	resourcesByFileID, err := loadInheritResourcesByFileID(filepath.Join(checkpointPath, "files.img"))
	if err != nil {
		return nil, nil, err
	}

	fdinfoPaths, err := filepath.Glob(filepath.Join(checkpointPath, "fdinfo-*.img"))
	if err != nil {
		return nil, nil, fmt.Errorf("failed to list fdinfo images: %w", err)
	}
	if len(fdinfoPaths) == 0 {
		return nil, nil, fmt.Errorf("no fdinfo images found in %s", checkpointPath)
	}
	sort.Strings(fdinfoPaths)

	stdoutSet := map[string]struct{}{}
	stderrSet := map[string]struct{}{}
	for _, fdinfoPath := range fdinfoPaths {
		fdinfoFile, err := os.Open(fdinfoPath)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to open %s: %w", fdinfoPath, err)
		}

		img, decodeErr := crit.New(fdinfoFile, nil, "", false, false).Decode(&fdinfo.FdinfoEntry{})
		closeErr := fdinfoFile.Close()
		if decodeErr != nil {
			return nil, nil, fmt.Errorf("failed to decode %s: %w", fdinfoPath, decodeErr)
		}
		if closeErr != nil {
			return nil, nil, fmt.Errorf("failed to close %s: %w", fdinfoPath, closeErr)
		}

		for _, entry := range img.Entries {
			fdEntry, ok := entry.Message.(*fdinfo.FdinfoEntry)
			if !ok {
				continue
			}

			resource := resourcesByFileID[fdEntry.GetId()]
			if resource == "" {
				continue
			}

			switch fdEntry.GetFd() {
			case 1:
				stdoutSet[resource] = struct{}{}
			case 2:
				stderrSet[resource] = struct{}{}
			}
		}
	}

	return sortedSetValues(stdoutSet), sortedSetValues(stderrSet), nil
}

func loadInheritResourcesByFileID(filesImagePath string) (map[uint32]string, error) {
	filesImage, err := os.Open(filesImagePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open %s: %w", filesImagePath, err)
	}

	img, decodeErr := crit.New(filesImage, nil, "", false, false).Decode(&fdinfo.FileEntry{})
	closeErr := filesImage.Close()
	if decodeErr != nil {
		return nil, fmt.Errorf("failed to decode %s: %w", filesImagePath, decodeErr)
	}
	if closeErr != nil {
		return nil, fmt.Errorf("failed to close %s: %w", filesImagePath, closeErr)
	}

	resources := make(map[uint32]string, len(img.Entries))
	for _, entry := range img.Entries {
		fileEntry, ok := entry.Message.(*fdinfo.FileEntry)
		if !ok {
			continue
		}
		resource := fileEntryInheritResource(fileEntry)
		if resource != "" {
			resources[fileEntry.GetId()] = resource
		}
	}

	return resources, nil
}

func fileEntryInheritResource(fileEntry *fdinfo.FileEntry) string {
	if fileEntry == nil {
		return ""
	}
	if pipeEntry := fileEntry.GetPipe(); pipeEntry != nil {
		return fmt.Sprintf("pipe:[%d]", pipeEntry.GetPipeId())
	}
	if socketEntry := fileEntry.GetUsk(); socketEntry != nil {
		return fmt.Sprintf("socket[%d]", socketEntry.GetIno())
	}
	if regEntry := fileEntry.GetReg(); regEntry != nil && regEntry.GetName() != "" {
		return regEntry.GetName()
	}
	return ""
}

func sortedSetValues(values map[string]struct{}) []string {
	if len(values) == 0 {
		return nil
	}
	out := make([]string, 0, len(values))
	for value := range values {
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}
