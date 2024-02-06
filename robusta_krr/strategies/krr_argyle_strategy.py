#!/usr/bin/env python3

import pydantic as pd
import numpy as np

import robusta_krr
from robusta_krr.api.models import (
    K8sObjectData, MetricsPodData, ResourceRecommendation, ResourceType, RunResult
)
from robusta_krr.core.abstract.strategies import (
    BaseStrategy,
    PodsTimeData,
)
from robusta_krr.core.integrations.prometheus.metrics import (
    CPUAmountLoader,
    MaxMemoryLoader,
    MemoryAmountLoader,
    PercentileCPULoader,
    PrometheusMetric,
)
from robusta_krr.strategies.simple import SimpleStrategySettings

from robusta_krr.core.integrations.prometheus.metrics.base import QueryType
from robusta_krr.utils.resource_units import parse


def SecondaryPercentileCPULoader(percentile: float) -> type[PrometheusMetric]:
    """
    A factory for creating percentile CPU usage metric loaders.
    """

    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")

    class SecondaryPercentileCPULoader(PrometheusMetric):
        def get_query(self, object: K8sObjectData, duration: str, step: str) -> str:
            pods_selector = "|".join(pod.name for pod in object.pods)
            cluster_label = self.get_prometheus_cluster_label()
            return f"""
                quantile_over_time(
                    {round(percentile / 100, 2)},
                    max(
                        rate(
                            container_cpu_usage_seconds_total{{
                                namespace="{object.namespace}",
                                pod=~"{pods_selector}",
                                container="{object.container}"
                                {cluster_label}
                            }}[{step}]
                        )
                    ) by (container, pod, job)
                    [{duration}:{step}]
                )
            """

    return SecondaryPercentileCPULoader


class OOMKilledLoader(PrometheusMetric):
    """
    A metric loader for loading OOMKilled metrics.
    """

    query_type: QueryType = QueryType.Query

    def get_query(self, object: K8sObjectData, duration: str, step: str) -> str:
        pods_selector = "|".join(pod.name for pod in object.pods)
        cluster_label = self.get_prometheus_cluster_label()
        return f"""
            sum without (uid, instance) (
                        kube_pod_container_status_last_terminated_reason{{reason="OOMKilled",
                        namespace="{object.namespace}",
                        pod=~"{pods_selector}",
                        container="{object.container}"
                        {cluster_label}
                        }}
                )
        """

# Providing description to the settings will make it available in the CLI help
class ArgyleStrategySettings(SimpleStrategySettings):
    memory_buffer_percentage: float = pd.Field(
        50, gt=0, description="The percentage of added buffer to the peak memory usage for memory limit recommendation."
    )

    def calculate_memory_usage(self, data: PodsTimeData) -> float:
        data_ = [np.max(values[:, 1]) for values in data.values()]
        if len(data_) == 0:
            return float("NaN")

        return np.max(data_)

    secondary_cpu_percentile: float = pd.Field(
        50, gt=0, description="The percentile to use for the secondary CPU usage metric."
    )


class ArgyleStrategy(BaseStrategy[ArgyleStrategySettings]):
    """
    A custom strategy that uses the provided parameters for CPU and memory.
    """

    display_name = "argyle"  # The name of the strategy
    rich_console = True  # Whether to use rich console for the CLI

    @staticmethod
    def info_from_list(info_list: list[str]):
        '''Helper function to join a list of strings into a single string'''

        if len(info_list) == 0:
            return None
        return ", ".join(info_list)

    @property
    def metrics(self) -> list[type[PrometheusMetric]]:
        return [PercentileCPULoader(self.settings.cpu_percentile), MaxMemoryLoader,
                CPUAmountLoader, MemoryAmountLoader, OOMKilledLoader,
                SecondaryPercentileCPULoader(self.settings.secondary_cpu_percentile),
        ]

    def _calculate_cpu_limit(self, cpu_recommended: float):
        if cpu_recommended < 0.3:
            return 2
        if cpu_recommended < 0.400:
            return cpu_recommended * 7
        elif cpu_recommended < 0.800:
            return cpu_recommended * 5
        elif cpu_recommended < 2:
            return cpu_recommended * 4
        else:
            cpu_recommended = cpu_recommended * 3
            if cpu_recommended > 10:
                # print(cpu_recommended)
                return None
            return cpu_recommended

    def __calculate_cpu_proposal(
        self, history_data: MetricsPodData, object_data: K8sObjectData
    ) -> ResourceRecommendation:
        if object_data.kind == "DaemonSet":
            data = history_data["SecondaryPercentileCPULoader"]
        else:
            data = history_data["PercentileCPULoader"]

        if len(data) == 0:
            return ResourceRecommendation.undefined(info="No data")

        data_count = {pod: values[0, 1] for pod, values in history_data["CPUAmountLoader"].items()}
        # Here we filter out pods from calculation that have less than `points_required` data points
        filtered_data = {
            pod: values for pod, values in data.items() if data_count.get(pod, 0) >= self.settings.points_required
        }

        if len(filtered_data) == 0:
            return ResourceRecommendation.undefined(info="Not enough data")

        info = None
        if object_data.hpa is not None and object_data.hpa.target_cpu_utilization_percentage is not None:
            info = "CPU utilization HPA detected, be wary of ever increasing recommendations"
        
        # if object_data.labels:
        #     print(object_data.labels)

        cpu_usage = self.settings.calculate_cpu_proposal(filtered_data)
        cpu_limit = (self._calculate_cpu_limit(cpu_usage))
        return ResourceRecommendation(request=cpu_usage, limit=cpu_limit, info=info)

    def __calculate_memory_proposal(
        self, history_data: MetricsPodData, object_data: K8sObjectData
    ) -> ResourceRecommendation:
        data = history_data["MaxMemoryLoader"]

        if len(data) == 0:
            return ResourceRecommendation.undefined(info="No data")

        data_count = {pod: values[0, 1] for pod, values in history_data["MemoryAmountLoader"].items()}
        # Here we filter out pods from calculation that have less than `points_required` data points
        filtered_data = {
            pod: value for pod, value in data.items() if data_count.get(pod, 0) >= self.settings.points_required
        }

        if len(filtered_data) == 0:
            return ResourceRecommendation.undefined(info="Not enough data")

        info = []
        if object_data.hpa is not None and object_data.hpa.target_memory_utilization_percentage is not None:
            info.append("Memory utilization HPA detected, be wary of ever increasing recommendations")


        if history_data.get("OOMKilledLoader", None):
            info.append("Container has been OOMKilled in the period")
            # if OOMKilled consider usage = limit
            memory_usage = object_data.allocations.limits.get(ResourceType.Memory)
        else:
            memory_usage = self.settings.calculate_memory_usage(filtered_data)

        strategy =  (
            object_data.annotations.get("argyle-krr-memory-strategy", "default") if
            object_data.annotations else "default"
        )
        min_memory_limit = (
            object_data.annotations.get("argyle-krr-memory-limit-min", None) if
            object_data.annotations else None
        )
        memory_limit = None

        if min_memory_limit and isinstance(min_memory_limit, str):
            min_memory_limit = parse(min_memory_limit)

        if strategy == "default" and isinstance(memory_usage, float):
            memory_request = memory_usage
            memory_limit = memory_usage * (1 + self.settings.memory_buffer_percentage / 100 )
        elif strategy == "guaranteed" and isinstance(memory_usage, float):
            memory_request = memory_usage * (1 + self.settings.memory_buffer_percentage / 100 )
            memory_limit = memory_usage * (1 + self.settings.memory_buffer_percentage / 100 )
        else:
            memory_request = memory_usage
            memory_limit = memory_usage * (1 + self.settings.memory_buffer_percentage / 100 )

        if isinstance(min_memory_limit, float) and memory_limit and memory_limit < min_memory_limit:
            memory_limit = min_memory_limit
            if strategy == "guaranteed":
                memory_request = min_memory_limit

        info = self.info_from_list(info)
        return ResourceRecommendation(request=memory_request, limit=memory_limit, info=info)

    def run(self, history_data: MetricsPodData, object_data: K8sObjectData) -> RunResult:
        return {
            ResourceType.CPU: self.__calculate_cpu_proposal(history_data, object_data),
            ResourceType.Memory: self.__calculate_memory_proposal(history_data, object_data),
        }


# Run it as `python ./custom_strategy.py my_strategy`
if __name__ == "__main__":
    robusta_krr.run()
