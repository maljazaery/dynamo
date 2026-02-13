package orchestrate

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strconv"
)

// nsrestoreResult matches the JSON written to stdout by the nsrestore binary.
type nsrestoreResult struct {
	RestoredPID     int `json:"restoredPID"`
	RestoredHostPID int `json:"restoredHostPID,omitempty"`
}

// execNSRestore launches the nsrestore binary inside the placeholder container's
// namespaces via nsenter and parses the JSON result from stdout.
// Logs from nsrestore flow directly to the agent's stderr.
func (r *Restorer) execNSRestore(ctx context.Context, placeholderPID int, checkpointPath, cudaDeviceMap, cgroupRoot string) (int, int, error) {
	args := []string{
		"-t", strconv.Itoa(placeholderPID),
		// Intentionally exclude cgroup namespace (-C): CRIU must manage cgroups
		// from the host-visible hierarchy so --cgroup-root remap works.
		"-m", // mount
		"-u", // uts
		"-i", // ipc
		"-n", // net
		"-p", // pid
		"--", r.cfg.NSRestorePath,
		"--checkpoint-path", checkpointPath,
	}
	if cudaDeviceMap != "" {
		args = append(args, "--cuda-device-map", cudaDeviceMap)
	}
	if cgroupRoot != "" {
		args = append(args, "--cgroup-root", cgroupRoot)
	}

	cmd := exec.CommandContext(ctx, "nsenter", args...)
	r.log.V(1).Info("Executing nsenter + nsrestore", "cmd", cmd.String())

	var stdout bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return 0, 0, fmt.Errorf("nsrestore failed: %w\nstdout: %s", err, stdout.String())
	}

	var result nsrestoreResult
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		return 0, 0, fmt.Errorf("failed to parse nsrestore result: %w\nstdout: %s", err, stdout.String())
	}
	if result.RestoredPID <= 0 {
		return 0, 0, fmt.Errorf("nsrestore returned invalid PID %d", result.RestoredPID)
	}

	return result.RestoredPID, result.RestoredHostPID, nil
}
