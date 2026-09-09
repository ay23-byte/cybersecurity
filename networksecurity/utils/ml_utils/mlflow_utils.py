import tempfile
import contextlib
import os
import mlflow
from networksecurity.logging.logger import logging

def capture_run_logs(func):
    def wrapper(*args, **kwargs):
        tmp = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".log", prefix="component_run_")
        tmp.close()
        try:
            with mlflow.start_run(nested=True):
                with open(tmp.name, "a", encoding="utf-8") as log_f, contextlib.redirect_stdout(log_f), contextlib.redirect_stderr(log_f):
                    result = func(*args, **kwargs)

                try:
                    mlflow.log_artifact(tmp.name, artifact_path="run_logs")
                except Exception as e:
                    logging.error(f"Failed to upload component run log: {e}")
            return result
        finally:
            try:
                if os.path.exists(tmp.name):
                    os.remove(tmp.name)
            except Exception:
                pass

    return wrapper
