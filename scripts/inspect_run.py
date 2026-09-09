import json
from mlflow.tracking import MlflowClient

rid = '44c9fed65492492faa6638b3c027cff4'
c = MlflowClient()
run = c.get_run(rid)
out = {
    'info': {
        'run_id': run.info.run_id,
        'experiment_id': run.info.experiment_id,
        'status': run.info.status,
        'start_time': run.info.start_time,
        'end_time': run.info.end_time,
        'lifecycle_stage': getattr(run.info, 'lifecycle_stage', None)
    },
    'tags': dict(run.data.tags),
    'metrics': dict(run.data.metrics),
    'params': dict(run.data.params)
}
print(json.dumps(out, indent=2, default=str))
