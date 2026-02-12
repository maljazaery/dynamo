package criu

import (
	"os"
	"path/filepath"
	"testing"
)

func TestParseRemapFilePathRejectsInvalidMagic(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "remap-fpath.img")
	if err := os.WriteFile(path, []byte("INVALID"), 0644); err != nil {
		t.Fatalf("write test file: %v", err)
	}

	if _, err := ParseRemapFilePath(path); err == nil {
		t.Fatalf("expected parse failure for invalid remap image")
	}
}

func TestParseFilesWithModeRejectsInvalidMagic(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "files.img")
	if err := os.WriteFile(path, []byte("INVALID"), 0644); err != nil {
		t.Fatalf("write test file: %v", err)
	}

	if _, err := ParseFilesWithMode(path); err == nil {
		t.Fatalf("expected parse failure for invalid files image")
	}
}

func TestDiscoverStdioInheritResourcesRequiresFDInfo(t *testing.T) {
	dir := t.TempDir()
	if _, _, err := DiscoverStdioInheritResources(dir); err == nil {
		t.Fatalf("expected failure when fdinfo image files are missing")
	}
}
