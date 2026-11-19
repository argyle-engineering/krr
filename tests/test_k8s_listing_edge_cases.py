from unittest.mock import MagicMock, patch

import pytest
from kubernetes.client import ApiException

from robusta_krr.core.integrations.kubernetes import ClusterLoader, KubernetesLoader
from robusta_krr.core.models.config import Config
from robusta_krr.core.models.objects import HPAData

# tests/conftest.py patches KubernetesLoader.list_scannable_objects for the whole session;
# keep a handle on the real implementation (captured at collection, before that patch applies).
_real_list_scannable_objects = KubernetesLoader.list_scannable_objects


@pytest.fixture
def mock_settings():
    config = MagicMock(spec=Config)
    config.max_workers = 2
    config.get_kube_client = MagicMock()
    config.selector = None
    config.namespaces = "*"
    with patch("robusta_krr.core.integrations.kubernetes.settings", config):
        yield config


def make_v1_hpa(namespace: str, kind: str, name: str) -> MagicMock:
    hpa = MagicMock()
    hpa.metadata.namespace = namespace
    hpa.spec.scale_target_ref.kind = kind
    hpa.spec.scale_target_ref.name = name
    hpa.spec.min_replicas = 1
    hpa.spec.max_replicas = 5
    hpa.spec.target_cpu_utilization_percentage = 70
    hpa.status.current_replicas = 2
    hpa.status.desired_replicas = 3
    return hpa


@pytest.mark.asyncio
async def test_hpa_falls_back_to_v1_when_v2_is_unavailable(mock_settings):
    loader = ClusterLoader()
    loader.autoscaling_v2 = MagicMock()
    loader.autoscaling_v2.list_horizontal_pod_autoscaler_for_all_namespaces.side_effect = ApiException(status=404)
    loader.autoscaling_v1 = MagicMock()
    loader.autoscaling_v1.list_horizontal_pod_autoscaler_for_all_namespaces.return_value = MagicMock(
        items=[make_v1_hpa("default", "Deployment", "web")]
    )

    result = await loader._ClusterLoader__list_hpa()  # type: ignore[attr-defined]

    assert result == {
        ("default", "Deployment", "web"): HPAData(
            min_replicas=1,
            max_replicas=5,
            current_replicas=2,
            desired_replicas=3,
            target_cpu_utilization_percentage=70,
            target_memory_utilization_percentage=None,
        )
    }
    loader.autoscaling_v1.list_horizontal_pod_autoscaler_for_all_namespaces.assert_called_once_with(
        watch=False, label_selector=None
    )


@pytest.mark.asyncio
async def test_hpa_v1_fallback_lists_each_configured_namespace(mock_settings):
    mock_settings.namespaces = ["ns-a", "ns-b"]
    loader = ClusterLoader()
    loader.autoscaling_v2 = MagicMock()
    loader.autoscaling_v2.list_namespaced_horizontal_pod_autoscaler.side_effect = ApiException(status=404)
    loader.autoscaling_v1 = MagicMock()
    loader.autoscaling_v1.list_namespaced_horizontal_pod_autoscaler.side_effect = lambda namespace, **_: MagicMock(
        items=[make_v1_hpa(namespace, "StatefulSet", f"db-{namespace}")]
    )

    result = await loader._try_list_hpa()

    assert set(result) == {
        ("ns-a", "StatefulSet", "db-ns-a"),
        ("ns-b", "StatefulSet", "db-ns-b"),
    }


@pytest.mark.asyncio
async def test_list_scannable_objects_returns_empty_list_when_no_cluster_loads(mock_settings):
    loader = KubernetesLoader()
    with patch.object(KubernetesLoader, "_try_create_cluster_loader", return_value=None):
        result = await _real_list_scannable_objects(loader, ["unreachable-a", "unreachable-b"])

    assert result == []
