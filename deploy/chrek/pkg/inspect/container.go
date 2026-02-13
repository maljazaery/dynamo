// Package inspect provides host-side container and device introspection.
// It consolidates discovery of container state (PID, OCI spec, rootfs,
// mounts, namespaces) and device resources (GPU UUIDs) used during
// checkpoint and restore orchestration.
package inspect

import (
	"context"
	"fmt"

	"github.com/containerd/containerd"
	"github.com/containerd/containerd/namespaces"
	specs "github.com/opencontainers/runtime-spec/specs-go"
)

const (
	// K8sNamespace is the containerd namespace used by Kubernetes.
	K8sNamespace = "k8s.io"

	// ContainerdSocket is the default containerd socket path.
	ContainerdSocket = "/run/containerd/containerd.sock"
)

// Client wraps a containerd client for container introspection.
type Client struct {
	client *containerd.Client
}

// NewClient creates a new containerd-backed introspection client.
func NewClient() (*Client, error) {
	client, err := containerd.New(ContainerdSocket)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to containerd at %s: %w", ContainerdSocket, err)
	}
	return &Client{client: client}, nil
}

// Close closes the containerd client connection.
func (c *Client) Close() error {
	if c.client != nil {
		return c.client.Close()
	}
	return nil
}

// ResolveContainer resolves a container by ID and returns its PID and OCI spec.
func (c *Client) ResolveContainer(ctx context.Context, containerID string) (int, *specs.Spec, error) {
	ctx = namespaces.WithNamespace(ctx, K8sNamespace)

	container, err := c.client.LoadContainer(ctx, containerID)
	if err != nil {
		return 0, nil, fmt.Errorf("failed to load container %s: %w", containerID, err)
	}

	task, err := container.Task(ctx, nil)
	if err != nil {
		return 0, nil, fmt.Errorf("failed to get task for container %s: %w", containerID, err)
	}

	spec, err := container.Spec(ctx)
	if err != nil {
		return 0, nil, fmt.Errorf("failed to get spec for container %s: %w", containerID, err)
	}

	return int(task.Pid()), spec, nil
}

// ResolveContainerByPod finds a container by pod name, namespace, and container name
// by listing containerd containers and matching CRI labels.
func (c *Client) ResolveContainerByPod(ctx context.Context, podName, podNamespace, containerName string) (int, *specs.Spec, error) {
	ctx = namespaces.WithNamespace(ctx, K8sNamespace)

	filter := fmt.Sprintf("labels.\"io.kubernetes.pod.name\"==%s,labels.\"io.kubernetes.pod.namespace\"==%s,labels.\"io.kubernetes.container.name\"==%s",
		podName, podNamespace, containerName)

	containers, err := c.client.Containers(ctx, filter)
	if err != nil {
		return 0, nil, fmt.Errorf("failed to list containers for pod %s/%s: %w", podNamespace, podName, err)
	}

	if len(containers) == 0 {
		return 0, nil, fmt.Errorf("no container found for pod %s/%s container %s", podNamespace, podName, containerName)
	}

	container := containers[0]
	task, err := container.Task(ctx, nil)
	if err != nil {
		return 0, nil, fmt.Errorf("failed to get task for container in pod %s/%s: %w", podNamespace, podName, err)
	}

	spec, err := container.Spec(ctx)
	if err != nil {
		return 0, nil, fmt.Errorf("failed to get spec for container in pod %s/%s: %w", podNamespace, podName, err)
	}

	return int(task.Pid()), spec, nil
}
