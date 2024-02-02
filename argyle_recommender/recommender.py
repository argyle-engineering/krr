#!/usr/bin/env python

import os
import platform
import pathlib
import subprocess
import json
import datetime
import logging
import sys

import structlog

from functools import reduce

from typing import Optional, Union

import typer
from typing_extensions import Annotated

from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString

import github
from rich.console import Console
import git


from robusta_krr.formatters.table import table, NONE_LITERAL, NAN_LITERAL
from robusta_krr.core.models.result import Result
from robusta_krr.utils import resource_units

WORKLOADS = ["Deployment", "Rollout", "Job", "DaemonSet", "StatefulSet", "CronJob"]
console = Console(width=500)
githubconsole = Console(record=True, no_color=True, width=500)


shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
]
if sys.stderr.isatty():
    # Pretty printing when we run in a terminal session.
    # Automatically prints pretty tracebacks when "rich" is installed
    processors = shared_processors + [
        structlog.dev.ConsoleRenderer(),
    ]
else:
    # Print JSON when we run, e.g., in a Docker container.
    # Also print structured tracebacks.
    processors = shared_processors + [
        structlog.processors.dict_tracebacks,
        structlog.processors.JSONRenderer(),
    ]

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    processors=processors
)
log = structlog.get_logger()


def _format(value):
    if value is None:
        return NONE_LITERAL
    elif isinstance(value, str):
        return NAN_LITERAL
    else:
        return resource_units.format(value)


def get_krr_json(label, namespace="*", prometheus=None, context=None, app_name=None) -> Result:
    log.debug("loading recomendations for %s in ns %s", label, namespace)
    if label == "no-selector":
        selector = []
    else:
        selector = ["-s", label]
    namespace_flag = []
    if namespace is not None and namespace != "*":
        namespace_flag = ["-n", namespace]

    prometheus_flag = []
    if prometheus is not None:
        prometheus_flag = ["-p", prometheus]

    context_flag = []
    if context is not None:
        context_flag = ["--context", context]
    command = ["krr", "argyle",
               "-f", "json", "-q"]
    command.extend(selector)
    command.extend(namespace_flag)
    command.extend(prometheus_flag)
    command.extend(context_flag)
    log.info("%s: %s", app_name, ' '.join(command))
    krr = subprocess.run(command, capture_output=True, encoding="utf8", check=True).stdout
    results = None
    try:
        results = json.loads(krr)
    except json.JSONDecodeError:
        try:
            krr = subprocess.run(command, capture_output=True, encoding="utf8", check=True).stdout
        except json.JSONDecodeError:
            log.error(krr[:300])
            raise
    assert results
    try:
        results = Result(**results)
    except Exception as e:
        log.error(results)
        log.error(e)
    return results


def find_recommendations(build_yaml_path, namespace, prometheus, context=None):
    scans = []
    results = None
    if build_yaml_path == "no-selector":
        return get_krr_json("no-selector", namespace=namespace,
                            prometheus=prometheus, context=context)

    with open(build_yaml_path, "r", encoding="utf-8") as build_file:
        yaml = YAML()
        docs = yaml.load_all(build_file)
        docs = [doc for doc in docs if doc.get("kind") in WORKLOADS]
        if len(docs) == 0:
            return None
        while len(docs) > 0:
            doc = docs[0]
            app_name = doc["metadata"].get("name")
            labels = doc["metadata"].get("labels")
            label = None
            label_name = None
            for label_name in [
                "app", "part-of", "app.kubernetes.io/instance", "k8s-app", "app.kubernetes.io/name",
            ]:
                label = labels.get(label_name)
                if label is not None:
                    break
            if label is None:
                label_name = next(iter(labels))
                label = labels.get(label_name)
            if namespace is None:
                namespace = doc["metadata"].get("namespace", "*")
            results = get_krr_json(
                f"{label_name}={label}", namespace, prometheus, app_name=app_name)
            if len(results.scans) == 0:
                docs.pop(0)
                continue
            all_popped = []
            for scan in results.scans:
                log.debug('object %s', scan.object.name)
                popped = [docs.pop(i) for i, doc in enumerate(docs) if doc["metadata"]["name"] == scan.object.name]
                all_popped.extend(popped)
                log.debug("popped %s objects", len(popped))
            if len(all_popped) == 0:
                popped_doc = docs.pop(0)
                log.debug("popped %s doc", popped_doc['metadata']['name'])
                continue
            scans.extend(results.scans)
    assert isinstance(results, Result)
    results.scans = scans
    results = Result(**results.dict())  # reinit to calculate score
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
            "limits": {"cpu": cpu_limits, "memory": memory_limits}}


def create_resource_transformers(results: Result, path: Union[str, pathlib.Path]) -> None:
    path = pathlib.Path(path)
    app_path = ""
    app_path = path.name.replace("-prod.yaml", "")
    for scan in results.scans:
        kind = scan.object.kind.lower()
        name = scan.object.name
        namespace = scan.object.namespace
        container = scan.object.container

        if app_path.startswith("scanners-application-"):
            app_path = "scanners"

        transformer_path = pathlib.Path(
            f"namespaces/{app_path}/resources/{name}-{kind}-resourcetransformer.yaml"
        )

        mode = "r+"
        if transformer_path.is_file():
            transformer_exists = True
        else:
            transformer_exists = False
            mode = "w+"
        os.makedirs(transformer_path.parent, exist_ok=True)

        with open(transformer_path, mode, encoding="utf-8") as transformer_file:
            yaml = YAML()
            transformer = None
            if transformer_exists:
                transformer = yaml.load(transformer_file)
                transformer_file.seek(0)
                if transformer is None or transformer.get("resourceQuotas") is None:
                    transformer_exists = False
                    transformer = {}
            if not transformer_exists:
                transformer = {
                    "apiVersion": "argyle.com/v1",
                    "kind": "ResourceQuotaTransformer",
                    "metadata": {
                        "name": f"{name}-{kind}",
                        "annotations": {
                            "config.kubernetes.io/function":
                            LiteralScalarString('''exec:
  path: argyle-resource-quota-transformer''')
                        },
                    },
                    "resourceQuotas":[]
                }
            assert transformer

            try:
                resource_quota =  {
                    **_recommended_to_resources(scan.recommended),
                    "workload" : name,
                    "container": container,
                    "kind": kind.title()
                }
                transformer["resourceQuotas"] = [
                    r for r in transformer["resourceQuotas"] if r["container"] != container]
                transformer["resourceQuotas"].append(resource_quota)
            except TypeError as e:
                log.error("%s/%s/%s/%s", kind, namespace, name, container)
                raise e

            yaml.dump(transformer, transformer_file)
            transformer_file.truncate()
            transformer_file.seek(0)
    handle_kustomizations(f"namespaces/{app_path}/resources/")
    return None


def _actual_cost(costs: dict, ondemand_ratio: float):
    return (
        costs["ondemand"]["price"] * ondemand_ratio
        + costs["committed"]["price"] * (1 - ondemand_ratio)
    )


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

    base_node_costs = [
        {
            "resource": "SSD backed PD Capacity",
            "sku": "B188-61DD-52E4",
            "quantity": 256,
            "price": 0.1615
        },
        {
            "resource": "Idle resource costs (estimate)",
            "sku": None,
            "quantity": 1,
            "price": 30
        },
    ]
    node_cost = reduce(lambda a, b: a + b["quantity"] * b["price"], base_node_costs, 0)

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
    return (cpu_cost, ram_cost, node_cost_fraction)


def handle_kustomizations(resources_path):
    kustomization_path = pathlib.Path(f"{resources_path}/kustomization.yaml")

    mode = "r+"
    if kustomization_path.is_file():
        kustomization_exists = True
    else:
        kustomization_exists = False
        os.makedirs(resources_path, exist_ok=True)
        mode = "w+"
    with open(kustomization_path, mode, encoding="utf-8") as kustomization_file:
        yaml = YAML()
        resources_component = None
        if kustomization_exists:
            resources_component = yaml.load(kustomization_file)
            kustomization_file.seek(0)
            if resources_component is None or "transformers" not in resources_component:
                kustomization_exists = False
        if not kustomization_exists:
            resources_component = {
                "apiVersion": "kustomize.config.k8s.io/v1alpha1",
                "kind": "Component",
                "transformers": []
            }
        assert resources_component
        resources_component["transformers"] = [
            t for t in resources_component["transformers"] if "resourcetransformer.yaml" not in t]
        resources_component["transformers"].extend(
            [t for t in os.listdir(resources_path) if "resourcetransformer.yaml" in t])
        yaml.dump(resources_component, kustomization_file)
        kustomization_file.truncate()
        kustomization_file.seek(0)
    subprocess.run([
        "argyle-kustomize-fmt", kustomization_path ],
        capture_output=True, encoding="utf8", check=True)

    parent_kustomization_path = kustomization_path.parent.parent.joinpath("kustomization.yaml")

    if str(parent_kustomization_path) == "namespaces/scanners/kustomization.yaml":
        parent_kustomization_path = pathlib.Path("namespaces/scanners/app-base/kustomization.yaml")
    mode = "r+"
    if parent_kustomization_path.is_file():
        parent_kustomization_exists = True
    else:
        parent_kustomization_exists = False
        mode = "w+"
    with parent_kustomization_path.open(mode, encoding="utf-8") as parent_kustomization_file:
        yaml = YAML()
        parent_kustomization = {}
        if parent_kustomization_exists:
            parent_kustomization = yaml.load(parent_kustomization_file)
            parent_kustomization_file.seek(0)
            if parent_kustomization is None:
                parent_kustomization_exists = False
        if not parent_kustomization_exists:
            raise ValueError(f"parent kustomization {parent_kustomization_path} does not exist")
        components = (
            parent_kustomization["components"] if
            "components" in parent_kustomization else [])
        components = [c for c in components if c != kustomization_path.parent.name]
        components.append(kustomization_path.parent.name)
        parent_kustomization["components"] = components
        yaml.dump(parent_kustomization, parent_kustomization_file)
        parent_kustomization_file.truncate()
        parent_kustomization_file.seek(0)


def process_app(path: Union[str, pathlib.Path], namespace,
                create_pr_flag: Union[bool, None] = False,
                prometheus=None, __key=None, context=None):
    if create_pr_flag is None:
        create_pr_flag = False

    repo = pull_repo(__key)
    os.chdir(repo.working_dir)
    results = find_recommendations(path, namespace, prometheus, context=context)
    if results is None:
        return None
    create_resource_transformers(results, path)
    tbl = table(results)
    total_estimate_in_table(tbl, results)
    if create_pr_flag and results.score > 70:
        log.info("Score %s is above 70, not creating PR", results.score)
        create_pr_flag = False
    pr = create_pr_func(create_pr_flag, path=path, namespace=namespace, table=tbl, __key=__key)
    if create_pr_flag and not isinstance(pr, tuple):
        log.info("PR %s created. https://github.com/argyle-systems/argyle-k8s/pull/%s ",
                 pr.number, pr.number)
    else:
        log.info("PR's content would have been %s", str(pr))


def get_github_token(__key):
    auth = auth_github_app(__key)
    gi = github.GithubIntegration(auth=auth)
    token = gi.get_access_token(45242597)
    return token


def auth_github_app(__key):
    return github.Auth.AppAuth(717966, __key)


def pull_repo(__key):
    token = get_github_token(__key)
    app_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(app_dir)
    local_path = "k8s_repo"
    username = "krr-recommender"
    password = token.token
    remote = f"https://{username}:{password}@github.com/argyle-systems/argyle-k8s.git"

    if os.path.isdir(local_path):
        repo = git.Repo(local_path)
        repo.head.reset()
        repo.heads.master.checkout()

    else:
        repo = git.Repo.clone_from(remote, local_path)
    return repo


def build_yamls(path=None):
    if path is None:
        target = "build-prod"
    else:
        target = path
    make = "make"
    if platform.system() == "Darwin":
        make = "gmake"

    subprocess.run([make, target], check=True)


def create_pr_func(create=False, path=None, namespace=None, table=None, __key=None):
    if table is None:
        raise ValueError("Need resource table to create the PR body")
    app = ""
    if str(path).startswith(".build"):
        app = str(path).replace(".build/", "").replace("-prod.yaml", "")
    repo = git.Repo(".")

    branch_suffix = ""
    if app:
        branch_suffix = f"{app}"
    else:
        branch_suffix = f"{datetime.datetime.now().timestamp():0.0f}"

    author = git.Actor("Argyle KRR recommender", "infrastructure+krrrecommender@argyle.com")
    branch_name = f"krr-recommender-{branch_suffix}"
    commit_msg = f"Adjusting resources acording to usage{f' for app {app}' if app else f' for namespace {namespace}' if namespace else ''}."
    githubconsole.print(table)

    report = githubconsole.export_text()
    g = github.Github(get_github_token(__key).token)
    ghrepo = g.get_repo("argyle-systems/argyle-k8s")
    body = f'''This is an automated PR to adjust resources{f" for {app}" if app else ""}.
{"This is created by looking at common selectors in workloads in the build output." if app else ""}
{f"This is created by looking at all workloads in the namespace {namespace}." if namespace else ""}

Resources recommendations are based on usage and paramaters that can be fine tuned.
Monthly xost estimates are based on the same strategy used to adjust resources.

```
{report}
```
    '''
    pr_title = f"Adjusting {f'{app} ' if app else ''}resources acording to usage"

    if create:
        repo.index.add(["namespaces"])
    if create and repo.is_dirty():
        try:
            # check if remote branch exists
            origin = repo.remote("origin")
            origin.fetch(branch_name)
            repo.create_head(branch_name, origin.refs[branch_name])
            repo.heads[branch_name].set_tracking_branch(origin.refs[branch_name])
            repo.git.stash('save')
            repo.heads[branch_name].checkout()
            try:
                repo.git.stash('pop')
            except git.GitCommandError: # conflict
                conflict_list = repo.git.diff(name_only=True, diff_filter="U").split("\n")
                for conflict in conflict_list:
                    repo.git.checkout("--theirs", conflict)  # resolve in favor of latest change
                    repo.git.rm(conflict, cached=True)
                    repo.git.add(conflict)
                repo.git.stash("drop")

        except git.GitCommandError: # if it does not exist, create it
            branch = repo.create_head(branch_name)
            branch.checkout()

        repo.index.commit(commit_msg, author=author, committer=author)

        try:
            repo.remote("origin").push(branch_name).raise_if_error()
        except git.GitError as e:
            log.error("Error pushing to origin")
            log.error(e)
            return (pr_title, body, branch_name)
        try:
            pr = ghrepo.create_pull(title=pr_title, body=body, head=branch_name, base="master")
            return pr
        except github.GithubException as e:
            messages = [er.get("message") for er in e.data.get("errors", dict())]
            if [m for m in messages if "A pull request already exists for" in m]:
                log.info("PR already exists, new commit pushed")
            else:
                log.error("Error creating PR")
                log.error(e)
            return (pr_title, body, branch_name)
    else:
        log.info("No PR created")
        if not repo.is_dirty():
            log.info("No changes to be made%s.", f' to {app}' if app else '')
        return (pr_title, body, branch_name)


def main(path: Annotated[str, typer.Argument],
         namespace: Annotated[Optional[str], typer.Option()] = None,
         create_pr: Annotated[Optional[bool], typer.Option()] = False,
         prometheus: Annotated[Optional[str], typer.Option("--prometheus", "-p")] = None,
         context: Annotated[Optional[str], typer.Option("--context", "-c")] = None
         ):

    key_file = os.environ.get("PRIVATE_KEY_FILE")
    if key_file:
        with open(key_file, "r", encoding="utf-8") as _f:
            __key = _f.read()

    else:
        __key = os.environ.get("PRIVATE_KEY", "")

    if not __key:
        raise ValueError(
            "Private key not found, please set either PRIVATE_KEY_FILE as a "
            " path or PRIVATE_KEY with the key contents.")
    repo = pull_repo(__key)
    os.chdir(repo.working_dir)
    if path in ["prod", "all"] or path.startswith(".build"):
        if path.startswith(".build"):
            build_yamls(path)
        else:
            build_yamls()
        for app_yaml in pathlib.Path(".build").glob("*-prod.yaml"):
            try:
                process_app(app_yaml, namespace, create_pr, prometheus, __key, context=context)
            except Exception:
                log.exception("error in processing app %s", app_yaml, stack_info=True)
    else:
        process_app(path, namespace, create_pr, prometheus, __key, context=context)


if __name__ == "__main__":
    show_locals = sys.stderr.isatty()
    app = typer.Typer(pretty_exceptions_show_locals=show_locals)
    app.command()(main)
    app()
