#!/usr/bin/env python

import os
import pathlib
import subprocess
import json

from functools import reduce

from typing import Optional

import typer
from typing_extensions import Annotated

from ruamel.yaml import YAML

import pprint
from rich.console import Console


from robusta_krr.formatters.table import table, NONE_LITERAL, NAN_LITERAL
from robusta_krr.core.models.result import Result
from robusta_krr.utils import resource_units

WORKLOADS = ["Deployment", "Rollout", "Job", "DaemonSet", "StatefulSet", "CronJob"]


def _format(value):
    if value is None:
        return NONE_LITERAL
    elif isinstance(value, str):
        return NAN_LITERAL
    else:
        return resource_units.format(value)
        

def get_krr_json(label, namespace="*"):
    # print(f"loading recomendations for {label} in ns {namespace}")
    if label == "no-selector":
        selector = []
    else:
        selector = ["-s", label]
    namespace_flag = []
    if namespace is not None and namespace != "*":
        namespace_flag = ["-n", namespace]
    command = [ "krr",  "argyle", "--context", "prod",
               "-f", "json", "-q", ]
    command.extend(selector)
    command.extend(namespace_flag)
    krr = subprocess.run(command, capture_output=True, encoding="utf8", check=True).stdout
    results = json.loads(krr)
    try:
        results = Result(**results)
    except:
        pprint.pprint(results)
    return(results)

def find_recommendations(build_yaml_path, namespace):
    scans = []
    if build_yaml_path == "no-selector":
        return get_krr_json("no-selector", namespace=namespace)

    with open(build_yaml_path, "r") as build_file:
        yaml = YAML(typ='safe')
        docs = yaml.load_all(build_file)
        docs = [doc for doc in docs if doc.get("kind") in WORKLOADS]
        if len(docs) == 0:
            return None
        while len(docs) > 0:
            doc = docs[0]
            labels =  doc["metadata"].get("labels")
            for label_name in [
                "app", "part-of", "app.kubernetes.io/instance", "k8s-app"
            ]:
                label = labels.get(label_name)
                if label is not None:
                    break
            if namespace is None:
                namespace = doc["metadata"].get("namespace", "*")
            results = get_krr_json(f"{label_name}={label}", namespace)
            if len(results.scans) == 0:
                docs.pop(0)
                continue
            for scan in results.scans:
                # print(f'object {scan["object"]["name"]}')
                popped = [docs.pop(i) for i, doc in enumerate(docs) if doc["metadata"]["name"] == scan.object.name]
                # print(f"popped {len(popped)} objects")
            scans.extend(results.scans)
    results.scans = scans
    return results


def find_object(name, kind, namespace, labels):
    namespace = namespace.replace("argyle-", "")
    naive_filename = pathlib.Path(f"namespaces/{namespace}/{name}-{kind}.yaml")
    if naive_filename.is_file():
        return naive_filename
    filename = next(pathlib.Path(".").glob(f"namespaces/{namespace}/**/{name}-{kind}*.yaml"), None)
    if filename is not None and filename.is_file():
        return filename
    namespace = labels.get("app")
    filename = next(pathlib.Path(".").glob(f"namespaces/{namespace}/**/{name}-{kind}*.yaml"), None)
    if filename is not None and filename.is_file():
        return filename
    filename = next(pathlib.Path(".").glob(f"namespaces/**/{name}-{kind}*.yaml"), None)
    if filename is not None and filename.is_file():
        return filename
    else:
        return None

    
def get_container_by_name(container_name, container_spec_list):
    for container in container_spec_list:
        if container["name"] == container_name:
            return container
    return None

def update_container_by_name(container_name, container_spec_list, container_updated_spec):
    for i, container in enumerate(container_spec_list):
        if container["name"] == container_name:
            container_spec_list[i] = container_updated_spec
            return container_spec_list
    return container_spec_list

def _recommended_to_resources(recommended):
    cpu_requests = _format(recommended.requests["cpu"].value)
    cpu_limits = _format(recommended.limits["cpu"].value)
    memory_requests = _format(recommended.requests["memory"].value)
    memory_limits = _format(recommended.limits["memory"].value)
    return {"requests": {"cpu": cpu_requests, "memory": memory_requests},
            "limits": {"cpu":cpu_limits, "memory": memory_limits}}


def match_objects(results: Result):
    unmatched_objects = []
    for scan in results.scans:
        kind = scan.object.kind.lower()
        name = scan.object.name
        namespace = scan.object.namespace
        labels = scan.object.labels
        file = find_object(name, kind, namespace, labels)
        if file is not None:
            with open(file, "r+") as workload_file:

                yaml = YAML()
                workload = yaml.load(workload_file)
                try:
                    containers = workload["spec"]["template"]["spec"]["containers"]
                except KeyError:  # probably a patch file
                    continue
                container = get_container_by_name(scan.object.container, containers)
                if container is None:
                    for resource in scan.recommended.info:
                        if scan.recommended.info[resource] is None:
                            scan.recommended.info[resource] = ""
                        scan.recommended.info[resource] += f"container not in {file}"
                    unmatched_objects.append(f"{kind}/{namespace}/{name}/{scan.object.container}")
                    continue
                try:
                    container["resources"] = _recommended_to_resources(scan.recommended)
                except TypeError as e:
                    pprint.pprint(f"{kind}/{namespace}/{name} - {file}")
                    pprint.pprint(container)
                    raise e

                workload["spec"]["template"]["spec"]["containers"] = update_container_by_name(scan.object.container, workload["spec"]["template"]["spec"]["containers"], container)
                
                workload_file.seek(0)
                yaml.dump(workload, workload_file)
                workload_file.truncate()
        else:
            unmatched_objects.append(f"{kind}/{namespace}/{name}/{scan.object.container}")
    return unmatched_objects


def _actual_cost(costs: dict, ondemand_ratio: float):
    return (costs["ondemand"]["price"] * ondemand_ratio + costs["committed"]["price"] * (1 - ondemand_ratio))


def total_estimate_in_table(table, results):
    total_cpu = 0
    total_ram = 0

    total_cpu_cost = 0
    total_ram_cost = 0
    total_node_cost = 0
    
    table.add_column("Node cost")
    table.add_column("CPU cost")
    table.add_column("RAM cost")

    for scan in results.scans:
        n_pods = scan.object.current_pods_count
        cpu = scan.recommended.requests["cpu"].value * n_pods or 0
        ram = scan.recommended.requests["memory"].value * n_pods or 0

        total_cpu += cpu
        total_ram += ram
        cpu_cost, ram_cost, node_cost = estimate_cost(cpu, ram)

        total_cpu_cost += cpu_cost
        total_ram_cost += ram_cost
        total_node_cost += node_cost
        table.columns[-3]._cells.append(f"USD {node_cost:.2f}")
        table.columns[-2]._cells.append(f"USD {cpu_cost:.2f}")
        table.columns[-1]._cells.append(f"USD {ram_cost:.2f}")
    table.columns[-3].footer = f"USD {total_node_cost:.2f}"
    table.columns[-2].footer = f"USD {total_cpu_cost:.2f}"
    table.columns[-1].footer = f"USD {total_ram_cost:.2f}"



def estimate_cost(cpu, ram):
    GIGABYTE = 1024 * 1024 * 1024
    HOURS_IN_MONTH = 24 * 30
    NODE_CPU = 8
    NODE_RAM = 64 * GIGABYTE
    ONDEMAND_RAM_RATIO = 0.05
    ONDEMAND_CPU_RATIO = 0.07

    ram = ram / GIGABYTE

    cpu_node_fraction = cpu / NODE_CPU
    ram_node_fraction = ram / NODE_RAM
    node_fraction = max(ram_node_fraction, cpu_node_fraction)
    
    base_node_costs = [{
        "resource": "SSD backed PD Capacity",
        "sku": "B188-61DD-52E4",
        "quantity": 256,
        "price": 0.1615
    }]
    node_cost = reduce(lambda a, b: a + b["quantity"]*b["price"], base_node_costs, 0)

    node_cost_fraction = node_fraction * node_cost

    ram_costs = {
        "committed": {
            "price": 0.001249809 * HOURS_IN_MONTH,
            "sku": "D86D-BE56-C7EB",
        },
        "ondemand": {
            "price": 0.00292353 * HOURS_IN_MONTH,
            "sku": "F449-33EC-A5EF",
        }
    }
    cpu_costs = {
        "committed": {
            "price": 0.009324454 * HOURS_IN_MONTH,
            "sku": "B4E1-097C-1E0A",
        },
        "ondemand": {
            "price": 0.02072101 * HOURS_IN_MONTH,
            "sku": "CF4E-A0C7-E3BF",
        }
    }
    
    cpu_cost = _actual_cost(cpu_costs, ONDEMAND_CPU_RATIO) * cpu
    ram_cost = _actual_cost(ram_costs, ONDEMAND_RAM_RATIO) * ram
    return(cpu_cost, ram_cost, node_cost_fraction)


def process_app(path, namespace):
    results = find_recommendations(path, namespace)
    if results is None:
        return None
    unmatched = match_objects(results)
    tbl = table(results)
    total_estimate_in_table(tbl, results)
    console = Console()
    console.print(tbl)
    if len(unmatched) > 0:
        console.print("found the following unmatched objects")
        console.print(unmatched)


def main(path: Annotated[Optional[str], typer.Argument], namespace: Annotated[Optional[str], typer.Option(None)] = None):
    if path in ["prod", "all"] :
        pull_repo()
        build_yamls()
        for path in pathlib.Path(".build").glob("*-prod.yaml"):
            process_app(path, namespace)
    else:
        process_app(path, namespace)



if __name__ == "__main__":
    typer.run(main)
