package cuda

import (
	"context"
	"time"

	"github.com/go-logr/logr"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	podresourcesv1 "k8s.io/kubelet/pkg/apis/podresources/v1"
)

// GetPodGPUUUIDsWithRetry resolves GPU UUIDs for a pod/container from kubelet pod-resources API.
func GetPodGPUUUIDsWithRetry(ctx context.Context, podName, podNamespace, containerName string, log logr.Logger) ([]string, error) {
	ticker := time.NewTicker(podGPUDiscoverTick)
	defer ticker.Stop()

	attempt := 0
	var lastErr error
	for {
		attempt++
		uuids, err := getPodGPUUUIDs(ctx, podName, podNamespace, containerName)
		if err == nil && len(uuids) > 0 {
			return uuids, nil
		}
		if err != nil {
			lastErr = err
		}

		uuidCount := 0
		if uuids != nil {
			uuidCount = len(uuids)
		}
		log.V(1).Info("Waiting for pod GPU UUIDs in pod-resources",
			"attempt", attempt,
			"pod", podName,
			"namespace", podNamespace,
			"container", containerName,
			"uuid_count", uuidCount,
		)

		select {
		case <-ctx.Done():
			if lastErr != nil {
				return nil, lastErr
			}
			return nil, nil
		case <-ticker.C:
		}
	}
}

func getPodGPUUUIDs(ctx context.Context, podName, podNamespace, containerName string) ([]string, error) {
	if podName == "" || podNamespace == "" {
		return nil, nil
	}

	conn, err := grpc.DialContext(
		ctx,
		"unix://"+podResourcesSocket,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
	)
	if err != nil {
		return nil, err
	}
	defer conn.Close()

	client := podresourcesv1.NewPodResourcesListerClient(conn)
	resp, err := client.List(ctx, &podresourcesv1.ListPodResourcesRequest{})
	if err != nil {
		return nil, err
	}

	for _, pod := range resp.GetPodResources() {
		if pod.GetName() != podName || pod.GetNamespace() != podNamespace {
			continue
		}
		for _, container := range pod.GetContainers() {
			if containerName != "" && container.GetName() != containerName {
				continue
			}
			for _, device := range container.GetDevices() {
				if device.GetResourceName() == nvidiaGPUResource {
					return device.GetDeviceIds(), nil
				}
			}
		}
	}

	return nil, nil
}
