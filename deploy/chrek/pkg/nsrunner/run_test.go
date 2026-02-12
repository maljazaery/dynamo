package nsrunner

import (
	"strings"
	"testing"
)

func TestIsUnsupportedFlagError(t *testing.T) {
	output := "flag provided but not defined: -mntns-compat-mode"
	if !IsUnsupportedFlagError(output, []string{"--mntns-compat-mode"}) {
		t.Fatalf("expected unsupported flag detection")
	}
	if IsUnsupportedFlagError(output, []string{"--rst-sibling"}) {
		t.Fatalf("expected non-matching flag to return false")
	}
}

func TestParseRunnerLogLineLogfmt(t *testing.T) {
	line := `time=2026-02-12T00:00:00Z level=info msg="restore step" phase=criu`
	level, msg, fields, ok := parseRunnerLogLine(line)
	if !ok {
		t.Fatalf("expected logfmt line to parse")
	}
	if level != "info" {
		t.Fatalf("unexpected level: %s", level)
	}
	if msg != "restore step" {
		t.Fatalf("unexpected message: %s", msg)
	}
	if fields["phase"] != "criu" {
		t.Fatalf("expected phase field")
	}
	if fields["helper_time"] != "2026-02-12T00:00:00Z" {
		t.Fatalf("expected helper_time field")
	}
}

func TestParseRunnerLogLineZap(t *testing.T) {
	line := "2026-02-12T00:00:00.000Z\tINFO\tns-restore-runner\trestore step\t{\"foo\":\"bar\"}"
	level, msg, fields, ok := parseRunnerLogLine(line)
	if !ok {
		t.Fatalf("expected zap-dev line to parse")
	}
	if level != "info" {
		t.Fatalf("unexpected level: %s", level)
	}
	if msg != "restore step" {
		t.Fatalf("unexpected message: %s", msg)
	}
	if fields["foo"] != "bar" {
		t.Fatalf("expected json field")
	}
	if fields["helper_time"] != "2026-02-12T00:00:00.000Z" {
		t.Fatalf("expected helper_time field")
	}
}

func TestSplitLogfmtTokensQuotedValue(t *testing.T) {
	line := `level=info msg="hello world" source=ns-restore-runner`
	tokens := splitLogfmtTokens(line)
	if len(tokens) != 3 {
		t.Fatalf("unexpected token count: %d", len(tokens))
	}
	if !strings.Contains(tokens[1], `"hello world"`) {
		t.Fatalf("expected quoted token to stay intact")
	}
}
