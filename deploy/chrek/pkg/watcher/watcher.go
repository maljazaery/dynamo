// Package watcher provides Kubernetes pod watching for automatic checkpointing.
package watcher

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/go-logr/logr"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	ktypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	ctrl "sigs.k8s.io/controller-runtime"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/externalrestore"
)

// WatcherConfig holds watcher configuration.
type WatcherConfig struct {
	NodeName            string
	RestrictedNamespace string // Optional: restrict watching to this namespace (empty = cluster-wide)
	AgentSocketPath     string // Pod-local UDS socket path exposed by the chrek API server.

	// Checkpoint configuration (from ConfigMap)
	CheckpointSpec *checkpoint.CheckpointSpec
}

// Watcher watches for pods with checkpoint/restore labels and triggers operations.
// All state tracking is done via pod annotations for idempotency across agent restarts.
type Watcher struct {
	config          WatcherConfig
	clientset       kubernetes.Interface
	agentClient     *externalrestore.Client
	discoveryClient *checkpoint.DiscoveryClient
	log             logr.Logger

	// In-flight guards prevent concurrent operations on the same pod within
	// a single agent lifetime. NOT used for cross-restart idempotency —
	// that comes from pod annotations.
	inFlight   map[string]struct{}
	inFlightMu sync.Mutex

	stopCh chan struct{}
}

const restoreReadyTimeout = 2 * time.Minute

// NewWatcher creates a new pod watcher.
func NewWatcher(cfg WatcherConfig, discoveryClient *checkpoint.DiscoveryClient) (*Watcher, error) {
	// Create in-cluster Kubernetes client
	restConfig, err := rest.InClusterConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to get in-cluster config: %w", err)
	}

	clientset, err := kubernetes.NewForConfig(restConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create kubernetes client: %w", err)
	}

	if cfg.AgentSocketPath == "" {
		return nil, fmt.Errorf("agent socket path is required")
	}

	return &Watcher{
		config:          cfg,
		clientset:       clientset,
		agentClient:     externalrestore.NewClient(cfg.AgentSocketPath),
		discoveryClient: discoveryClient,
		log:             ctrl.Log.WithName("watcher"),
		inFlight:        make(map[string]struct{}),
		stopCh:          make(chan struct{}),
	}, nil
}

// Start begins watching for pods and processing checkpoint/restore events.
func (w *Watcher) Start(ctx context.Context) error {
	if w.config.CheckpointSpec == nil {
		return fmt.Errorf("checkpoint spec is required")
	}

	w.log.Info("Starting pod watcher",
		"node", w.config.NodeName,
		"checkpoint", checkpoint.KubeLabelCheckpointSource,
		"restore", checkpoint.KubeLabelCheckpointRestore,
		"restore_enabled", w.agentClient != nil,
		"socket_path", w.config.AgentSocketPath,
	)

	// Namespace restriction options shared across informer factories
	var nsOptions []informers.SharedInformerOption
	if w.config.RestrictedNamespace != "" {
		w.log.Info("Restricting pod watching to namespace", "namespace", w.config.RestrictedNamespace)
		nsOptions = append(nsOptions, informers.WithNamespace(w.config.RestrictedNamespace))
	} else {
		w.log.Info("Watching pods cluster-wide (all namespaces)")
	}

	var syncFuncs []cache.InformerSynced

	// --- Checkpoint informer: watches pods with checkpoint-source=true ---
	checkpointSelector := labels.SelectorFromSet(labels.Set{
		checkpoint.KubeLabelCheckpointSource: "true",
	}).String()

	ckptFactoryOpts := append([]informers.SharedInformerOption{
		informers.WithTweakListOptions(func(opts *metav1.ListOptions) {
			opts.LabelSelector = checkpointSelector
		}),
	}, nsOptions...)

	ckptFactory := informers.NewSharedInformerFactoryWithOptions(
		w.clientset, 30*time.Second, ckptFactoryOpts...,
	)

	ckptInformer := ckptFactory.Core().V1().Pods().Informer()
	ckptInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			w.handleCheckpointPodEvent(ctx, obj.(*corev1.Pod))
		},
		UpdateFunc: func(_, newObj interface{}) {
			w.handleCheckpointPodEvent(ctx, newObj.(*corev1.Pod))
		},
	})
	go ckptFactory.Start(w.stopCh)
	syncFuncs = append(syncFuncs, ckptInformer.HasSynced)

	// --- Restore informer: watches pods with checkpoint-restore=true ---
	if w.agentClient != nil {
		restoreSelector := labels.SelectorFromSet(labels.Set{
			checkpoint.KubeLabelCheckpointRestore: "true",
		}).String()

		restoreFactoryOpts := append([]informers.SharedInformerOption{
			informers.WithTweakListOptions(func(opts *metav1.ListOptions) {
				opts.LabelSelector = restoreSelector
			}),
		}, nsOptions...)

		restoreFactory := informers.NewSharedInformerFactoryWithOptions(
			w.clientset, 30*time.Second, restoreFactoryOpts...,
		)

		restoreInformer := restoreFactory.Core().V1().Pods().Informer()
		restoreInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				w.handleRestorePodEvent(ctx, obj.(*corev1.Pod))
			},
			UpdateFunc: func(_, newObj interface{}) {
				w.handleRestorePodEvent(ctx, newObj.(*corev1.Pod))
			},
		})
		go restoreFactory.Start(w.stopCh)
		syncFuncs = append(syncFuncs, restoreInformer.HasSynced)
	}

	// Wait for all caches to sync
	if !cache.WaitForCacheSync(w.stopCh, syncFuncs...) {
		return fmt.Errorf("failed to sync informer caches")
	}

	w.log.Info("Pod watcher started and caches synced")

	// Wait for context cancellation
	<-ctx.Done()
	close(w.stopCh)

	return nil
}

// Stop stops the watcher
func (w *Watcher) Stop() {
	close(w.stopCh)
}

// handleCheckpointPodEvent processes a checkpoint pod event.
// Idempotency: reads nvidia.com/checkpoint-status annotation to avoid re-checkpointing.
func (w *Watcher) handleCheckpointPodEvent(ctx context.Context, pod *corev1.Pod) {
	if pod.Spec.NodeName != w.config.NodeName {
		return
	}
	if !w.isPodReady(pod) {
		return
	}

	podKey := fmt.Sprintf("%s/%s", pod.Namespace, pod.Name)

	checkpointHash, ok := pod.Labels[checkpoint.KubeLabelCheckpointHash]
	if !ok || checkpointHash == "" {
		w.log.Info("Pod has checkpoint label but no checkpoint-hash label", "pod", podKey)
		return
	}

	// Pod annotation is the source of truth — survives agent restarts.
	annotationStatus := pod.Annotations[checkpoint.KubeAnnotationCheckpointStatus]
	if annotationStatus == "completed" || annotationStatus == "in_progress" {
		return
	}

	// In-flight guard prevents concurrent operations within this agent lifetime.
	if !w.tryAcquire(podKey) {
		return
	}

	w.log.Info("Pod ready, triggering checkpoint", "pod", podKey, "checkpoint_hash", checkpointHash)
	w.emitPodEvent(ctx, pod, corev1.EventTypeNormal, "CheckpointRequested", fmt.Sprintf("Checkpoint requested: %s", checkpointHash))

	go w.doCheckpoint(ctx, pod, checkpointHash, podKey)
}

// isPodRunning checks if the pod phase is Running (containers started).
func (w *Watcher) isPodRunning(pod *corev1.Pod) bool {
	return pod.Status.Phase == corev1.PodRunning
}

// isPodReady checks if all containers in the pod are ready
func (w *Watcher) isPodReady(pod *corev1.Pod) bool {
	if pod.Status.Phase != corev1.PodRunning {
		return false
	}

	for _, cond := range pod.Status.Conditions {
		if cond.Type == corev1.PodReady && cond.Status == corev1.ConditionTrue {
			return true
		}
	}

	return false
}

// handleRestorePodEvent processes a restore pod event.
// Idempotency: reads nvidia.com/restore-status annotation to avoid re-restoring.
func (w *Watcher) handleRestorePodEvent(ctx context.Context, pod *corev1.Pod) {
	if pod.Spec.NodeName != w.config.NodeName {
		return
	}

	podKey := fmt.Sprintf("%s/%s", pod.Namespace, pod.Name)

	if !w.isPodRunning(pod) {
		return
	}

	// Pod annotation is the source of truth — survives agent restarts.
	annotationStatus := pod.Annotations[checkpoint.KubeAnnotationRestoreStatus]

	// Once restored and serving, readiness flips to true and re-triggers are skipped.
	if w.isPodReady(pod) {
		return
	}

	// Terminal states: don't retry completed or failed restores.
	if annotationStatus == "completed" || annotationStatus == "in_progress" || annotationStatus == "failed" {
		return
	}

	checkpointHash, ok := pod.Labels[checkpoint.KubeLabelCheckpointHash]
	if !ok || checkpointHash == "" {
		w.log.Info("Restore pod has no checkpoint-hash label", "pod", podKey)
		return
	}

	// Verify checkpoint is ready on disk: directory exists without tmp_ prefix.
	checkpointDir := filepath.Join(w.config.CheckpointSpec.BasePath, checkpointHash)
	if _, err := os.Stat(checkpointDir); os.IsNotExist(err) {
		w.log.V(1).Info("Checkpoint not ready on disk, skipping restore", "pod", podKey, "checkpoint_hash", checkpointHash)
		return
	}

	if !w.tryAcquire(podKey) {
		return
	}

	w.log.Info("Restore pod running, triggering external restore", "pod", podKey, "checkpoint_hash", checkpointHash)
	w.emitPodEvent(ctx, pod, corev1.EventTypeNormal, "RestoreRequested", fmt.Sprintf("Restore requested from checkpoint %s", checkpointHash))

	go w.doRestore(ctx, pod, checkpointHash, podKey)
}

// doRestore performs external restore by calling the local chrek UDS API.
// Annotates the pod with restore status for cross-restart idempotency.
func (w *Watcher) doRestore(ctx context.Context, pod *corev1.Pod, checkpointHash, podKey string) {
	defer w.release(podKey)

	log := w.log.WithValues("pod", podKey, "checkpoint_hash", checkpointHash)

	// Mark in_progress on the pod before starting work.
	if err := w.annotatePod(ctx, pod, map[string]string{
		checkpoint.KubeAnnotationRestoreStatus: "in_progress",
	}); err != nil {
		log.Error(err, "Failed to annotate pod with restore in_progress")
		return
	}

	if w.agentClient == nil {
		err := fmt.Errorf("agent UDS client is not configured")
		log.Error(err, "External restore failed")
		w.emitPodEvent(ctx, pod, corev1.EventTypeWarning, "RestoreFailed", err.Error())
		w.annotatePod(ctx, pod, map[string]string{checkpoint.KubeAnnotationRestoreStatus: "failed"})
		return
	}

	containerName := resolveMainContainerName(pod)
	if containerName == "" {
		err := fmt.Errorf("no containers found in pod spec")
		log.Error(err, "Restore failed")
		w.emitPodEvent(ctx, pod, corev1.EventTypeWarning, "RestoreFailed", err.Error())
		w.annotatePod(ctx, pod, map[string]string{checkpoint.KubeAnnotationRestoreStatus: "failed"})
		return
	}

	req := externalrestore.RestoreAPIRequest{
		CheckpointHash: checkpointHash,
		PodName:        pod.Name,
		PodNamespace:   pod.Namespace,
		ContainerName:  containerName,
	}

	result, err := w.agentClient.Restore(ctx, req)
	if err != nil {
		log.Error(err, "External restore failed")
		w.emitPodEvent(ctx, pod, corev1.EventTypeWarning, "RestoreFailed", err.Error())
		w.annotatePod(ctx, pod, map[string]string{checkpoint.KubeAnnotationRestoreStatus: "failed"})
		return
	}

	log.Info("External restore completed successfully",
		"restored_pid", result.RestoredPID,
		"restored_host_pid", result.RestoredHostPID,
		"completed_steps", result.CompletedSteps,
	)

	placeholderHostPID, _, err := w.discoveryClient.ResolveContainerByPod(ctx, pod.Name, pod.Namespace, containerName)
	if err != nil {
		log.Error(err, "Failed to resolve placeholder host PID for watcher signaling")
		w.emitPodEvent(ctx, pod, corev1.EventTypeWarning, "RestoreFailed", err.Error())
		w.annotatePod(ctx, pod, map[string]string{checkpoint.KubeAnnotationRestoreStatus: "failed"})
		return
	}

	if err := w.sendSignalViaPIDNamespace(placeholderHostPID, result.RestoredPID, syscall.SIGUSR2, "restore complete"); err != nil {
		log.Error(err, "Failed to signal restored runtime process")
		w.emitPodEvent(ctx, pod, corev1.EventTypeWarning, "RestoreFailed", err.Error())
		w.annotatePod(ctx, pod, map[string]string{checkpoint.KubeAnnotationRestoreStatus: "failed"})
		return
	}

	if err := w.waitForPodReady(ctx, pod.Namespace, pod.Name, containerName, restoreReadyTimeout); err != nil {
		log.Error(err, "Restore post-signal readiness check failed")
		w.emitPodEvent(ctx, pod, corev1.EventTypeWarning, "RestoreFailed", err.Error())
		w.annotatePod(ctx, pod, map[string]string{checkpoint.KubeAnnotationRestoreStatus: "failed"})
		return
	}

	w.emitPodEvent(ctx, pod, corev1.EventTypeNormal, "RestoreSucceeded", fmt.Sprintf("Restore completed from checkpoint %s", checkpointHash))
	w.annotatePod(ctx, pod, map[string]string{checkpoint.KubeAnnotationRestoreStatus: "completed"})
}

// doCheckpoint performs the checkpoint and signals the runtime directly.
// Annotates the pod with checkpoint status for cross-restart idempotency.
func (w *Watcher) doCheckpoint(ctx context.Context, pod *corev1.Pod, checkpointHash, podKey string) {
	defer w.release(podKey)

	log := w.log.WithValues("pod", podKey, "checkpoint_hash", checkpointHash)

	// Mark in_progress on the pod before starting work.
	if err := w.annotatePod(ctx, pod, map[string]string{
		checkpoint.KubeAnnotationCheckpointStatus: "in_progress",
	}); err != nil {
		log.Error(err, "Failed to annotate pod with checkpoint in_progress")
		return
	}

	if w.agentClient == nil {
		err := fmt.Errorf("agent UDS client is not configured")
		log.Error(err, "Checkpoint failed")
		w.emitPodEvent(ctx, pod, corev1.EventTypeWarning, "CheckpointFailed", err.Error())
		w.annotatePod(ctx, pod, map[string]string{checkpoint.KubeAnnotationCheckpointStatus: "failed"})
		return
	}

	containerName := resolveMainContainerName(pod)

	// Get container ID from status
	var containerID string
	for _, cs := range pod.Status.ContainerStatuses {
		if cs.Name == containerName {
			containerID = cs.ContainerID
			if len(containerID) > 13 && containerID[:13] == "containerd://" {
				containerID = containerID[13:]
			}
			break
		}
	}

	if containerID == "" {
		log.Info("Could not find container ID")
		w.emitPodEvent(ctx, pod, corev1.EventTypeWarning, "CheckpointFailed", "Could not resolve target container ID")
		w.annotatePod(ctx, pod, map[string]string{checkpoint.KubeAnnotationCheckpointStatus: "failed"})
		return
	}

	log.Info("Found container, starting checkpoint", "container_id", containerID)

	containerPID, _, err := w.discoveryClient.ResolveContainer(ctx, containerID)
	if err != nil {
		log.Error(err, "Failed to resolve container")
		w.emitPodEvent(ctx, pod, corev1.EventTypeWarning, "CheckpointFailed", fmt.Sprintf("Container resolve failed: %v", err))
		w.annotatePod(ctx, pod, map[string]string{checkpoint.KubeAnnotationCheckpointStatus: "failed"})
		return
	}

	if w.config.CheckpointSpec == nil {
		log.Info("CheckpointSpec is nil - cannot perform checkpoint")
		w.emitPodEvent(ctx, pod, corev1.EventTypeWarning, "CheckpointFailed", "CheckpointSpec is nil")
		w.annotatePod(ctx, pod, map[string]string{checkpoint.KubeAnnotationCheckpointStatus: "failed"})
		return
	}

	req := externalrestore.CheckpointAPIRequest{
		ContainerID:    containerID,
		ContainerName:  containerName,
		CheckpointHash: checkpointHash,
		PodName:        pod.Name,
		PodNamespace:   pod.Namespace,
	}

	result, err := w.agentClient.Checkpoint(ctx, req)
	if err != nil {
		log.Error(err, "Checkpoint failed")
		w.emitPodEvent(ctx, pod, corev1.EventTypeWarning, "CheckpointFailed", err.Error())
		if signalErr := w.sendSignalToPID(containerPID, syscall.SIGTERM, "checkpoint failed"); signalErr != nil {
			log.Error(signalErr, "Failed to signal checkpoint failure to runtime process")
		}
		w.annotatePod(ctx, pod, map[string]string{checkpoint.KubeAnnotationCheckpointStatus: "failed"})
		return
	}

	checkpointPath := filepath.Join(w.config.CheckpointSpec.BasePath, checkpointHash)
	if result != nil && result.CheckpointHash != "" {
		checkpointHash = result.CheckpointHash
		checkpointPath = filepath.Join(w.config.CheckpointSpec.BasePath, result.CheckpointHash)
	}

	log.Info("Checkpoint completed successfully", "checkpoint_dir", checkpointPath)
	w.emitPodEvent(ctx, pod, corev1.EventTypeNormal, "CheckpointSucceeded", fmt.Sprintf("Checkpoint completed: %s", checkpointHash))

	if err := w.sendSignalToPID(containerPID, syscall.SIGUSR1, "checkpoint complete"); err != nil {
		log.Error(err, "Failed to signal checkpoint completion to runtime process")
		w.emitPodEvent(ctx, pod, corev1.EventTypeWarning, "CheckpointFailed", err.Error())
		w.annotatePod(ctx, pod, map[string]string{checkpoint.KubeAnnotationCheckpointStatus: "failed"})
		return
	}

	w.annotatePod(ctx, pod, map[string]string{checkpoint.KubeAnnotationCheckpointStatus: "completed"})
}

func (w *Watcher) sendSignalToPID(pid int, sig syscall.Signal, reason string) error {
	signalID := int(sig)
	if pid <= 0 {
		return fmt.Errorf("invalid PID %d for signal %d", pid, signalID)
	}
	if err := syscall.Kill(pid, sig); err != nil {
		return fmt.Errorf("failed to signal PID %d with signal %d (%s): %w", pid, signalID, reason, err)
	}
	w.log.Info("Signaled runtime process", "pid", pid, "signal", signalID, "reason", reason)
	return nil
}

func (w *Watcher) sendSignalViaPIDNamespace(referenceHostPID, targetNamespacePID int, sig syscall.Signal, reason string) error {
	if referenceHostPID <= 0 {
		return fmt.Errorf("invalid reference host PID %d for signal %d", referenceHostPID, int(sig))
	}
	if targetNamespacePID <= 0 {
		return fmt.Errorf("invalid namespace PID %d for signal %d", targetNamespacePID, int(sig))
	}

	cmd := exec.Command(
		"nsenter",
		"-t", strconv.Itoa(referenceHostPID),
		"-p",
		"--",
		"kill",
		fmt.Sprintf("-%d", int(sig)),
		strconv.Itoa(targetNamespacePID),
	)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf(
			"failed to signal namespace PID %d via reference host PID %d with signal %d (%s): %w (output: %s)",
			targetNamespacePID,
			referenceHostPID,
			int(sig),
			reason,
			err,
			strings.TrimSpace(string(output)),
		)
	}

	w.log.Info("Signaled runtime process in PID namespace",
		"reference_host_pid", referenceHostPID,
		"namespace_pid", targetNamespacePID,
		"signal", int(sig),
		"reason", reason,
	)
	return nil
}

func (w *Watcher) resolveHostPIDForRestoredProcess(ctx context.Context, pod *corev1.Pod, containerName string, restoredNamespacePID int) (int, error) {
	placeholderHostPID, _, err := w.discoveryClient.ResolveContainerByPod(ctx, pod.Name, pod.Namespace, containerName)
	if err != nil {
		return 0, fmt.Errorf("failed to resolve placeholder host PID: %w", err)
	}
	hostPID, err := resolveHostPIDInSameNamespace(placeholderHostPID, restoredNamespacePID)
	if err != nil {
		return 0, fmt.Errorf("failed to map restored namespace PID %d to host PID: %w", restoredNamespacePID, err)
	}
	return hostPID, nil
}

func resolveHostPIDInSameNamespace(referenceHostPID, targetNamespacePID int) (int, error) {
	if referenceHostPID <= 0 {
		return 0, fmt.Errorf("invalid reference host PID %d", referenceHostPID)
	}
	if targetNamespacePID <= 0 {
		return 0, fmt.Errorf("invalid target namespace PID %d", targetNamespacePID)
	}

	referenceMountNS, err := os.Readlink(fmt.Sprintf("%s/%d/ns/mnt", checkpoint.HostProcPath, referenceHostPID))
	if err != nil {
		return 0, fmt.Errorf("failed to read reference mount namespace for host PID %d: %w", referenceHostPID, err)
	}
	referencePIDNS, err := os.Readlink(fmt.Sprintf("%s/%d/ns/pid", checkpoint.HostProcPath, referenceHostPID))
	if err != nil {
		return 0, fmt.Errorf("failed to read reference PID namespace for host PID %d: %w", referenceHostPID, err)
	}

	entries, err := os.ReadDir(checkpoint.HostProcPath)
	if err != nil {
		return 0, fmt.Errorf("failed to list host proc entries: %w", err)
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		hostPID, err := strconv.Atoi(entry.Name())
		if err != nil {
			continue
		}

		mountNSPath := fmt.Sprintf("%s/%d/ns/mnt", checkpoint.HostProcPath, hostPID)
		mountNS, err := os.Readlink(mountNSPath)
		if err != nil || mountNS != referenceMountNS {
			continue
		}

		namespacePID, err := readInnermostNamespacePID(hostPID)
		if err != nil {
			continue
		}
		if namespacePID == targetNamespacePID {
			return hostPID, nil
		}
	}

	return 0, fmt.Errorf(
		"no host PID found for namespace PID %d in mount namespace %s (reference PID namespace %s)",
		targetNamespacePID,
		referenceMountNS,
		referencePIDNS,
	)
}

func readInnermostNamespacePID(hostPID int) (int, error) {
	statusPath := fmt.Sprintf("%s/%d/status", checkpoint.HostProcPath, hostPID)
	data, err := os.ReadFile(statusPath)
	if err != nil {
		return 0, err
	}

	for _, line := range strings.Split(string(data), "\n") {
		if !strings.HasPrefix(line, "NSpid:") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			return 0, fmt.Errorf("malformed NSpid line in %s", statusPath)
		}
		return strconv.Atoi(fields[len(fields)-1])
	}

	return 0, fmt.Errorf("NSpid not found in %s", statusPath)
}

func (w *Watcher) waitForPodReady(ctx context.Context, namespace, podName, containerName string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	lastPhase := ""

	for time.Now().Before(deadline) {
		pod, err := w.clientset.CoreV1().Pods(namespace).Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("failed to get restore pod %s/%s: %w", namespace, podName, err)
		}

		lastPhase = string(pod.Status.Phase)
		ready := false
		for _, condition := range pod.Status.Conditions {
			if condition.Type == corev1.PodReady && condition.Status == corev1.ConditionTrue {
				ready = true
				break
			}
		}
		if ready {
			return nil
		}

		for _, cs := range pod.Status.ContainerStatuses {
			if cs.Name != containerName {
				continue
			}
			if cs.State.Terminated != nil {
				return fmt.Errorf(
					"restore pod %s/%s container %s terminated: reason=%s exitCode=%d",
					namespace,
					podName,
					containerName,
					cs.State.Terminated.Reason,
					cs.State.Terminated.ExitCode,
				)
			}
		}

		time.Sleep(1 * time.Second)
	}

	return fmt.Errorf("restore pod %s/%s did not become Ready within %s (last phase: %s)", namespace, podName, timeout, lastPhase)
}

// tryAcquire claims the in-flight slot for podKey. Returns false if already held.
func (w *Watcher) tryAcquire(podKey string) bool {
	w.inFlightMu.Lock()
	defer w.inFlightMu.Unlock()
	if _, held := w.inFlight[podKey]; held {
		return false
	}
	w.inFlight[podKey] = struct{}{}
	return true
}

// release frees the in-flight slot for podKey.
func (w *Watcher) release(podKey string) {
	w.inFlightMu.Lock()
	defer w.inFlightMu.Unlock()
	delete(w.inFlight, podKey)
}

// annotatePod merges the given annotations onto the pod via the Kubernetes API.
func (w *Watcher) annotatePod(ctx context.Context, pod *corev1.Pod, annotations map[string]string) error {
	patch := fmt.Sprintf(`{"metadata":{"annotations":{`)
	first := true
	for k, v := range annotations {
		if !first {
			patch += ","
		}
		patch += fmt.Sprintf("%q:%q", k, v)
		first = false
	}
	patch += `}}}`

	_, err := w.clientset.CoreV1().Pods(pod.Namespace).Patch(
		ctx, pod.Name, ktypes.MergePatchType, []byte(patch), metav1.PatchOptions{},
	)
	if err != nil {
		w.log.Error(err, "Failed to annotate pod",
			"pod", fmt.Sprintf("%s/%s", pod.Namespace, pod.Name),
			"annotations", annotations,
		)
	}
	return err
}

// resolveMainContainerName picks the "main" container or falls back to the first one.
func resolveMainContainerName(pod *corev1.Pod) string {
	containerName := ""
	for _, c := range pod.Spec.Containers {
		if c.Name == "main" {
			return c.Name
		}
		if containerName == "" {
			containerName = c.Name
		}
	}
	return containerName
}

func (w *Watcher) emitPodEvent(ctx context.Context, pod *corev1.Pod, eventType, reason, message string) {
	event := &corev1.Event{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: fmt.Sprintf("%s-", pod.Name),
			Namespace:    pod.Namespace,
		},
		InvolvedObject: corev1.ObjectReference{
			Kind:       "Pod",
			Namespace:  pod.Namespace,
			Name:       pod.Name,
			UID:        pod.UID,
			APIVersion: "v1",
		},
		Type:    eventType,
		Reason:  reason,
		Message: message,
		Source: corev1.EventSource{
			Component: "chrek-watcher",
		},
		Count:          1,
		FirstTimestamp: metav1.Now(),
		LastTimestamp:  metav1.Now(),
	}

	if _, err := w.clientset.CoreV1().Events(pod.Namespace).Create(ctx, event, metav1.CreateOptions{}); err != nil {
		w.log.Error(err, "Failed to create watcher event",
			"pod", fmt.Sprintf("%s/%s", pod.Namespace, pod.Name),
			"reason", reason,
			"message", message,
		)
	}
}
