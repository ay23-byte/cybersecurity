import json
from mlflow.tracking import MlflowClient # type: ignore
import mlflow # type: ignore
import datetime


c = MlflowClient()
try:
    exps = c.list_experiments()
except AttributeError:
    # Fallback for MlflowClient versions without list_experiments
    exps = mlflow.search_experiments()
output = {"experiments": []}
for e in exps:
    exp = {
        "id": e.experiment_id,
        "name": e.name,
        "artifact_location": e.artifact_location,
        "lifecycle_stage": e.lifecycle_stage,
        "runs": []
    }
    runs = c.search_runs([e.experiment_id], max_results=1000)
    for r in runs:
        st = r.info.start_time
        et = r.info.end_time
        start_readable = datetime.datetime.fromtimestamp(st/1000).isoformat() if st else None
        end_readable = datetime.datetime.fromtimestamp(et/1000).isoformat() if et else None
        exp["runs"].append({
            "run_id": r.info.run_id,
            "status": r.info.status,
            "start_time": r.info.start_time,
            "start_time_readable": start_readable,
            "end_time": r.info.end_time,
            "end_time_readable": end_readable,
            "metrics": r.data.metrics,
            "params": r.data.params,
            "tags": r.data.tags,
        })
    output["experiments"].append(exp)
print(json.dumps(output, indent=2, default=str))
