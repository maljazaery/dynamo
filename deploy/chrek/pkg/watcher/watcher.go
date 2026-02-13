// Package watcher provides Kubernetes pod watching for automatic checkpoint/restore.
// The watcher is the sole entry point for chrek operations — it detects pods with
// checkpoint/restore labels and calls the orchestrators directly.
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

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/config"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/inspect"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/orchestrate"
)

// Watcher watches for pods with checkpoint/restore labels and triggers operations.
type Watcher struct {
	config          *config.AgentConfig
	clientset       kubernetes.Interface
	checkpointer    *orchestrate.Checkpointer
	restorer        *orchestrate.Restorer
	discoveryClient *inspect.Client
	log             logr.Logger

	inFlight   map[string]struct{}
	inFlightMu sync.Mutex

	stopCh chan struct{}
}

// NewWatcher creates a new pod watcher.
func NewWatcher(
	cfg *config.AgentConfig,
	checkpointer *orchestrate.Checkpointer,
	restorer *orchestrate.Restorer,
	discoveryClient *inspect.Client,
	log logr.Logger,
) (*Watcher, error) {
	restConfig, err := rest.InClusterConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to get in-cluster config: %w", err)
	}

	clientset, err := kubernetes.NewForConfig(restConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create kubernetes client: %w", err)
	}

	return &Watcher{
		config:          cfg,
		clientset:       clientset,
		checkpointer:    checkpointer,
		restorer:        restorer,
		discoveryClient: discoveryClient,
		log:             log,
		inFlight:        make(map[string]struct{}),
		stopCh:          make(chan struct{}),
	}, nil
}

// Start begins watching for pods and processing checkpoint/restore events.
func (w *Watcher) Start(ctx context.Context) error {
	w.log.Info("Starting pod watcher",
		"node", w.config.NodeName,
		"checkpoint", config.KubeLabelIsCheckpointSource,
		"restore", config.KubeLabelIsRestoreTarget,
	)

	var nsOptions []informers.SharedInformerOption
	if w.config.RestrictedNamespace != "" {
		w.log.Info("Restricting pod watching to namespace", "namespace", w.config.RestrictedNamespace)
		nsOptions = append(nsOptions, informers.WithNamespace(w.config.RestrictedNamespace))
	} else {
		w.log.Info("Watching pods cluster-wide (all namespaces)")
	}

	var syncFuncs []cache.InformerSynced

	// Checkpoint informer
	checkpointSelector := labels.SelectorFromSet(labels.Set{
		config.KubeLabelIsCheckpointSource: "true",
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

	// Restore informer
	restoreSelector := labels.SelectorFromSet(labels.Set{
		config.KubeLabelIsRestoreTarget: "true",
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

	if !cache.WaitForCacheSync(w.stopCh, syncFuncs...) {
		return fmt.Errorf("failed to sync informer caches")
	}

	w.log.Info("Pod watcher started and caches synced")
	<-ctx.Done()
	close(w.stopCh)
	return nil
}

func (w *Watcher) handleCheckpointPodEvent(ctx context.Context, pod *corev1.Pod) {
	if pod.Spec.NodeName != w.config.NodeName {
		return
	}
	if !w.isPodReady(pod) {
		return
	}

	podKey := fmt.Sprintf("%s/%s", pod.Namespace, pod.Name)

	checkpointHash, ok := pod.Labels[config.KubeLabelCheckpointHash]
	if !ok || checkpointHash == "" {
		w.log.Info("Pod has checkpoint label but no checkpoint-hash label", "pod", podKey)
		return
	}

	annotationStatus := pod.Annotations[config.KubeAnnotationCheckpointStatus]
	if annotationStatus == "completed" || annotationStatus == "in_progress" {
		return
	}

	if !w.tryAcquire(podKey) {
		return
	}

	w.log.Info("Pod ready, triggering checkpoint", "pod", podKey, "checkpoint_hash", checkpointHash)
	w.emitPodEvent(ctx, pod, corev1.EventTypeNormal, "CheckpointRequested", fmt.Sprintf("Checkpoint requested: %s", checkpointHash))

	go w.doCheckpoint(ctx, pod, checkpointHash, podKey)
}

func (w *Watcher) isPodRunning(pod *corev1.Pod) bool {
	return pod.Status.Phase == corev1.PodRunning
}

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

func (w *Watcher) handleRestorePodEvent(ctx context.Context, pod *corev1.Pod) {
	if pod.Spec.NodeName != w.config.NodeName {
		return
	}

	podKey := fmt.Sprintf("%s/%s", pod.Namespace, pod.Name)

	if !w.isPodRunning(pod) {
		return
	}

	annotationStatus := pod.Annotations[config.KubeAnnotationRestoreStatus]

	if w.isPodReady(pod) {
		return
	}

	if annotationStatus == "completed" || annotationStatus == "in_progress" || annotationStatus == "failed" {
		return
	}

	checkpointHash, ok := pod.Labels[config.KubeLabelCheckpointHash]
	if !ok || checkpointHash == "" {
		w.log.Info("Restore pod has no checkpoint-hash label", "pod", podKey)
		return
	}

	checkpointDir := filepath.Join(w.config.Checkpoint.BasePath, checkpointHash)
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

func (w *Watcher) doRestore(ctx context.Context, pod *corev1.Pod, checkpointHash, podKey string) {
	defer w.release(podKey)

	log := w.log.WithValues("pod", podKey, "checkpoint_hash", checkpointHash)

	if err := w.annotatePod(ctx, pod, map[string]string{
		config.KubeAnnotationRestoreStatus: "in_progress",
	}); err != nil {
		log.Error(err, "Failed to annotate pod with restore in_progress")
		return
	}

	containerName := resolveMainContainerName(pod)
	if containerName == "" {
		err := fmt.Errorf("no containers found in pod spec")
		log.Error(err, "Restore failed")
		w.emitPodEvent(ctx, pod, corev1.EventTypeWarning, "RestoreFailed", err.Error())
		w.annotatePod(ctx, pod, map[string]string{config.KubeAnnotationRestoreStatus: "failed"})
		return
	}

	req := orchestrate.RestoreRequest{
		CheckpointHash: checkpointHash,
		PodName:        pod.Name,
		PodNamespace:   pod.Namespace,
		ContainerName:  containerName,
	}

	result, err := w.restorer.Restore(ctx, req)
	if err != nil {
		log.Error(err, "External restore failed")
		w.emitPodEvent(ctx, pod, corev1.EventTypeWarning, "RestoreFailed", err.Error())
		w.annotatePod(ctx, pod, map[string]string{config.KubeAnnotationRestoreStatus: "failed"})
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
		w.annotatePod(ctx, pod, map[string]string{config.KubeAnnotationRestoreStatus: "failed"})
		return
	}

	if err := w.sendSignalViaPIDNamespace(placeholderHostPID, result.RestoredPID, syscall.SIGUSR2, "restore complete"); err != nil {
		log.Error(err, "Failed to signal restored runtime process")
		w.emitPodEvent(ctx, pod, corev1.EventTypeWarning, "RestoreFailed", err.Error())
		w.annotatePod(ctx, pod, map[string]string{config.KubeAnnotationRestoreStatus: "failed"})
		return
	}

	if err := w.waitForPodReady(ctx, pod.Namespace, pod.Name, containerName); err != nil {
		log.Error(err, "Restore post-signal readiness check failed")
		w.emitPodEvent(ctx, pod, corev1.EventTypeWarning, "RestoreFailed", err.Error())
		w.annotatePod(ctx, pod, map[string]string{config.KubeAnnotationRestoreStatus: "failed"})
		return
	}

	w.emitPodEvent(ctx, pod, corev1.EventTypeNormal, "RestoreSucceeded", fmt.Sprintf("Restore completed from checkpoint %s", checkpointHash))
	w.annotatePod(ctx, pod, map[string]string{config.KubeAnnotationRestoreStatus: "completed"})
}

func (w *Watcher) doCheckpoint(ctx context.Context, pod *corev1.Pod, checkpointHash, podKey string) {
	defer w.release(podKey)

	log := w.log.WithValues("pod", podKey, "checkpoint_hash", checkpointHash)

	if err := w.annotatePod(ctx, pod, map[string]string{
		config.KubeAnnotationCheckpointStatus: "in_progress",
	}); err != nil {
		log.Error(err, "Failed to annotate pod with checkpoint in_progress")
		return
	}

	containerName := resolveMainContainerName(pod)

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
		w.annotatePod(ctx, pod, map[string]string{config.KubeAnnotationCheckpointStatus: "failed"})
		return
	}

	log.Info("Found container, starting checkpoint", "container_id", containerID)

	containerPID, _, err := w.discoveryClient.ResolveContainer(ctx, containerID)
	if err != nil {
		log.Error(err, "Failed to resolve container")
		w.emitPodEvent(ctx, pod, corev1.EventTypeWarning, "CheckpointFailed", fmt.Sprintf("Container resolve failed: %v", err))
		w.annotatePod(ctx, pod, map[string]string{config.KubeAnnotationCheckpointStatus: "failed"})
		return
	}

	req := orchestrate.CheckpointRequest{
		ContainerID:    containerID,
		ContainerName:  containerName,
		CheckpointHash: checkpointHash,
		CheckpointDir:  w.config.Checkpoint.BasePath,
		NodeName:       w.config.NodeName,
		PodName:        pod.Name,
		PodNamespace:   pod.Namespace,
	}

	result, err := w.checkpointer.Checkpoint(ctx, req, &w.config.Checkpoint)
	if err != nil {
		log.Error(err, "Checkpoint failed")
		w.emitPodEvent(ctx, pod, corev1.EventTypeWarning, "CheckpointFailed", err.Error())
		if signalErr := w.sendSignalToPID(containerPID, syscall.SIGTERM, "checkpoint failed"); signalErr != nil {
			log.Error(signalErr, "Failed to signal checkpoint failure to runtime process")
		}
		w.annotatePod(ctx, pod, map[string]string{config.KubeAnnotationCheckpointStatus: "failed"})
		return
	}

	checkpointPath := filepath.Join(w.config.Checkpoint.BasePath, checkpointHash)
	if result != nil && result.CheckpointHash != "" {
		checkpointHash = result.CheckpointHash
		checkpointPath = filepath.Join(w.config.Checkpoint.BasePath, result.CheckpointHash)
	}

	log.Info("Checkpoint completed successfully", "checkpoint_dir", checkpointPath)
	w.emitPodEvent(ctx, pod, corev1.EventTypeNormal, "CheckpointSucceeded", fmt.Sprintf("Checkpoint completed: %s", checkpointHash))

	if err := w.sendSignalToPID(containerPID, syscall.SIGUSR1, "checkpoint complete"); err != nil {
		log.Error(err, "Failed to signal checkpoint completion to runtime process")
		w.emitPodEvent(ctx, pod, corev1.EventTypeWarning, "CheckpointFailed", err.Error())
		w.annotatePod(ctx, pod, map[string]string{config.KubeAnnotationCheckpointStatus: "failed"})
		return
	}

	w.annotatePod(ctx, pod, map[string]string{config.KubeAnnotationCheckpointStatus: "completed"})
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
			targetNamespacePID, referenceHostPID, int(sig), reason, err, strings.TrimSpace(string(output)),
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

func (w *Watcher) waitForPodReady(ctx context.Context, namespace, podName, containerName string) error {
	lastPhase := ""

	for {
		pod, err := w.clientset.CoreV1().Pods(namespace).Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("failed to get restore pod %s/%s: %w", namespace, podName, err)
		}

		lastPhase = string(pod.Status.Phase)
		for _, condition := range pod.Status.Conditions {
			if condition.Type == corev1.PodReady && condition.Status == corev1.ConditionTrue {
				return nil
			}
		}

		for _, cs := range pod.Status.ContainerStatuses {
			if cs.Name != containerName {
				continue
			}
			if cs.State.Terminated != nil {
				return fmt.Errorf(
					"restore pod %s/%s container %s terminated: reason=%s exitCode=%d",
					namespace, podName, containerName,
					cs.State.Terminated.Reason, cs.State.Terminated.ExitCode,
				)
			}
		}

		select {
		case <-ctx.Done():
			return fmt.Errorf("restore pod %s/%s did not become Ready (last phase: %s): %w", namespace, podName, lastPhase, ctx.Err())
		case <-time.After(1 * time.Second):
		}
	}
}

func (w *Watcher) tryAcquire(podKey string) bool {
	w.inFlightMu.Lock()
	defer w.inFlightMu.Unlock()
	if _, held := w.inFlight[podKey]; held {
		return false
	}
	w.inFlight[podKey] = struct{}{}
	return true
}

func (w *Watcher) release(podKey string) {
	w.inFlightMu.Lock()
	defer w.inFlightMu.Unlock()
	delete(w.inFlight, podKey)
}

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
