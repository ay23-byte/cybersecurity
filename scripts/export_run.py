import json
import os
import sys
from mlflow.tracking import MlflowClient

def pick_latest_finished_run(client: MlflowClient):
    # gather candidate runs from all experiments and pick the most recent finished run
    latest = None
    # Prefer using a client method if available
    try:
        exps = client.list_experiments()
        exp_ids = [e.experiment_id for e in exps]
    except Exception:
        # Fallback: read experiment folders from the local `mlruns` directory
        exp_ids = []
        if os.path.isdir("mlruns"):
            for name in os.listdir("mlruns"):
                path = os.path.join("mlruns", name)
                if os.path.isdir(path):
                    exp_ids.append(name)

    for exp_id in exp_ids:
        try:
            runs = client.search_runs([exp_id], order_by=["attributes.start_time DESC"], max_results=5)
        except Exception:
            runs = []
        for r in runs:
            if getattr(r.info, "status", None) == "FINISHED":
                if latest is None or (getattr(r.info, "start_time", 0) or 0) > (getattr(latest.info, "start_time", 0) or 0):
                    latest = r
    return latest


def export_run(run_id: str, client: MlflowClient):
    out = os.path.join("out_runs", run_id)
    os.makedirs(out, exist_ok=True)
    run = client.get_run(run_id)
    info = {
        "run_id": run.info.run_id,
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "user_id": getattr(run.info, 'user_id', None),
        "lifecycle_stage": getattr(run.info, 'lifecycle_stage', None)
    }
    metadata = {
        "info": info,
        "metrics": run.data.metrics,
        "params": run.data.params,
        "tags": run.data.tags,
    }
    with open(os.path.join(out, "run.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    # download artifacts at root
    client.download_artifacts(run_id, "", out)
    print("Exported run to", out)


def main():
    c = MlflowClient()
    run_id = None
    if len(sys.argv) > 1:
        run_id = sys.argv[1]
    else:
        latest = pick_latest_finished_run(c)
        if latest is None:
            print("No finished runs found in any experiment.")
            return
        run_id = latest.info.run_id
        print("Auto-detected latest finished run:", run_id)

    export_run(run_id, c)


if __name__ == "__main__":
    main()
