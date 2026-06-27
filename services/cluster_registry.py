from __future__ import annotations

import os

from kubernetes import client, config


KUBECONFIG_PATH = os.getenv("KUBECONFIG", "/app/.kube/config")


def get_api_client(cluster_name: str | None) -> client.ApiClient:
    """Return a Kubernetes ApiClient for the target cluster.

    None means the cluster this pod is running in — uses in-cluster config
    and falls back to local kubeconfig for local development.

    A non-None value loads the named context from the kubeconfig file mounted
    at KUBECONFIG (default /app/.kube/config).
    """
    if cluster_name is None:
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        return client.ApiClient()

    conf = client.Configuration()
    config.load_kube_config(
        config_file=KUBECONFIG_PATH,
        context=cluster_name,
        client_configuration=conf,
    )
    return client.ApiClient(configuration=conf)
