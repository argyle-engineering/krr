import os
import traceback
import shutil
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import Response

from argyle_recommender.recommender import (build_yamls, log,
                                            create_pr_func, pull_repo,
                                            process_app, get_krr_json, create_resource_transformers)


from robusta_krr.formatters.table import table

app = FastAPI(title="Argyle Recommender API", version="1.0.0")

log.info("Starting Argyle Recommender API")

key_file = os.environ.get("PRIVATE_KEY_FILE")
if key_file:
    with open(key_file, "r", encoding="utf-8") as _f:
        __key = _f.read()

else:
    __key = os.environ.get("PRIVATE_KEY", "")

@app.get("/")
async def root():
    return {"message": "Welcome to the Argyle Recommender API!"}

@app.get("/health")
async def health_check():
    return Response(
        content='{"status":"OK"}',
        status_code=status.HTTP_200_OK
    )

@app.get("/apps")
async def get_apps(request: Request):
    try:
        repo = pull_repo(__key)
        os.chdir(repo.working_dir)
        build_yamls("all")
        apps = os.listdir(os.path.join(repo.working_dir, ".build"))
        apps = [app.replace(".yaml", "") for app in apps]
    finally:
        shutil.rmtree(repo.working_dir, ignore_errors=True)

    return {"apps": apps}


@app.get("/app/{app_name}")
async def get_app(request: Request, app_name: str):
    try:
        repo = pull_repo(__key)
        os.chdir(repo.working_dir)
        log.info(f"Working dir: {repo.working_dir}")
        if not app_name.endswith("-prod") and not app_name.endswith("-dev"):
            app_name = f"{app_name}-prod"
        app_path = os.path.join(".build", f"{app_name}.yaml")
        build_yamls(app_path)
        context = app_name.split("-")[-1]
        if not os.path.exists(app_path):
            raise HTTPException(status_code=404, detail="App not found")
        try:
            create_pr = True
            prometheus = None
            clean_resources_flag = True
            settings = {}
            namespace = None
            process_app(app_path, namespace, create_pr, prometheus, __key, context=context,
                        clean_resources_flag=clean_resources_flag, settings=settings
                        )
        except Exception as e:
            log.error(f"Error processing app: {str(e)}")
            log.error(traceback.format_exc())
            
            raise HTTPException(status_code=500, detail=f"Error processing app: {str(e)}") from e
    finally:
        shutil.rmtree(repo.working_dir, ignore_errors=True)
    return {"app": "OK"}

@app.get("/app/{app_name}/selector/{selector_name}/{selector_value}")
async def get_app_with_selector(request: Request, app_name: Optional[str],
                                selector_name: str, selector_value: str):
    try:
        repo = pull_repo(__key)
        selector_name = selector_name.replace("_", "/")
        os.chdir(repo.working_dir)
        if not app_name.endswith("-prod") and not app_name.endswith("-dev"):
            app_name = f"{app_name}-prod"
        # app_path = os.path.join(".build", f"{app_name}.yaml")
        # build_yamls(app_path)
        context = app_name.split("-")[-1]
        # if not os.path.exists(app_path):
            # raise HTTPException(status_code=404, detail="App not found")
        try:
            create_pr = True
            prometheus = None
            settings = {}
            namespace = None
            results = get_krr_json(f"{selector_name}={selector_value}", namespace, prometheus,
                    app_name=app_name, context=context, settings=settings)
            create_resource_transformers(results, f".build/{app_name}.yaml")
            tbl = table(results)
            selector = f"{selector_name}={selector_value}"

            pr = create_pr_func(create_pr, path=f".build/{app_name}.yaml", namespace=namespace,
                                table=tbl, __key=__key, selector=selector)
            return (f"PR {pr.number} created."
                    f" https://github.com/argyle-systems/argyle-k8s/pull/{pr.number}")
        except Exception as e:
            log.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error processing app: {str(e)}")
    finally:
        shutil.rmtree(repo.working_dir, ignore_errors=True)


@app.post("/handle_alert")
async def post_handle_alert(request: Request):
    """Handle incoming alerts from the victoria metrics alertmanager.
    This endpoint expects a JSON payload with the alert details.
    Initial implementation will handle ComponentOutOfMemory alerts.
    The alert payload should be in the format specified in docs
    https://prometheus.io/docs/alerting/latest/configuration/#webhook_config

    Example payload:
    {
        "version": "4",
        "groupKey": <string>,              // key identifying the group of alerts (e.g. to deduplicate)
        "truncatedAlerts": <int>,          // how many alerts have been truncated due to "max_alerts"
        "status": "<resolved|firing>",
        "receiver": <string>,
        "groupLabels": <object>,
        "commonLabels": <object>,
        "commonAnnotations": <object>,
        "externalURL": <string>,           // backlink to the Alertmanager.
        "alerts": [
            {
            "status": "<resolved|firing>",
            "labels": <object>,
            "annotations": <object>,
            "startsAt": "<rfc3339>",
            "endsAt": "<rfc3339>",
            "generatorURL": <string>,      // identifies the entity that caused the alert
            "fingerprint": <string>        // fingerprint to identify the alert
            },
            ...
        ]
    }
    """
    if not request.headers.get("Content-Type") == "application/json":
        raise HTTPException(status_code=400, detail="Content-Type must be application/json")
    data = await request.json()
    log.info(f"Received alert: {data}")
    if not data.get("alerts"):
        log.error("No alerts found in the payload")
        raise HTTPException(status_code=400, detail="No alerts found in the payload")
    alert = data.get("alerts")[0]
    if len(data.get("alerts")) > 1:
        log.warning(f"Received multiple alerts, only processing the first one: {alert}")
    if not data.get("status") or data["status"] != "firing":
        log.info("Alert is not firing, ignoring")
        return {"message": "Alert is not firing, ignoring"}
    if not alert.get("labels") or not alert["labels"].get("alertname"):
        log.error("Alert does not have a valid alertname label")
        raise HTTPException(status_code=400, detail="Alert does not have a valid alertname label")
    alert_name = alert["labels"]["alertname"]
    if alert_name != "ComponentOutOfMemory":
        log.info(f"Alert {alert_name} is not handled by this endpoint, ignoring")
        return {"message": f"Alert {alert_name} is not handled by this endpoint, ignoring"}
    container_name = alert["labels"].get("container")
    namespace = alert["labels"].get("namespace")
    pod_name = alert["labels"].get("pod")
    # workload_name = alert["labels"].
    return {"message": f"Alert received for {container_name}/{pod_name} in {namespace} namespace"}
