import os
import json
import mlflow
from contextlib import AbstractContextManager
from typing import Dict, Any, Optional, List

LOCAL_URI = "file:experiments/mlruns"

class MLflowTracker(AbstractContextManager):
    """
    Minimal MLflow wrapper to keep training scripts clean:
    - sets a local tracking URI (under experiments/mlruns)
    - creates/selects an experiment
    - starts/ends a run as a context manager
    - provides helpers to log params, metrics, dicts, and artifacts
    """

    def __init__(self,
                 experiment_name: str = "deep-learning-market-prediction",
                 run_name: Optional[str] = None,
                 tags: Optional[Dict[str, str]] = None):
        # Use a local file-based MLflow backend so runs are portable
        mlflow.set_tracking_uri(LOCAL_URI)
        mlflow.set_experiment(experiment_name)
        self.run = None
        self.run_name = run_name
        self.tags = tags or {}

    def __enter__(self):
        # Start a new run (optionally named and tagged)
        self.run = mlflow.start_run(run_name=self.run_name, tags=self.tags)
        return self

    def __exit__(self, exc_type, exc, tb):
        # Always attempt to end the run; mark as failed if an exception occurred
        if exc is not None:
            mlflow.set_tag("run_status", "failed")
        mlflow.end_run()

    # ----------------- Logging helpers -----------------
    def log_params(self, params: Dict[str, Any]):
        """
        Log (possibly nested) params by flattening dicts into 'a.b.c' keys.
        """
        flat = {}

        def _flatten(d, prefix=""):
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                if isinstance(v, dict):
                    _flatten(v, key)
                else:
                    flat[key] = v

        _flatten(params)
        mlflow.log_params(flat)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """
        Log a single numeric metric. Use 'step' to record time series metrics.
        """
        mlflow.log_metric(key, float(value), step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log multiple metrics at once, optionally at a specific step.
        """
        for k, v in metrics.items():
            self.log_metric(k, v, step=step)

    def log_artifact(self, path: str, artifact_path: Optional[str] = None):
        """
        Log a single artifact if it exists on disk.
        """
        if os.path.exists(path):
            mlflow.log_artifact(path, artifact_path=artifact_path)

    def log_artifacts(self, paths: List[str], artifact_path: Optional[str] = None):
        """
        Log multiple artifacts if they exist on disk.
        """
        for p in paths:
            self.log_artifact(p, artifact_path=artifact_path)

    def log_dict(self, d: Dict[str, Any], artifact_file: str):
        """
        Log a dictionary as a JSON artifact.
        """
        mlflow.log_dict(d, artifact_file)

    def set_tags(self, tags: Dict[str, str]):
        """
        Set or update run-level tags.
        """
        for k, v in tags.items():
            mlflow.set_tag(k, str(v))

    # ----------------- Convenience -----------------
    @staticmethod
    def maybe_log_file(path: str, artifact_path: Optional[str] = None):
        """
        If 'path' exists, log it as an artifact under 'artifact_path'.
        """
        if os.path.exists(path):
            mlflow.log_artifact(path, artifact_path=artifact_path)

    @staticmethod
    def try_read_json(path: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to load and return a JSON file. Returns None on failure.
        """
        if os.path.exists(path):
            with open(path, "r") as f:
                try:
                    return json.load(f)
                except Exception:
                    return None
        return None
