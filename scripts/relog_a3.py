import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))
    os.environ.setdefault("MLFLOW_TRACKING_USERNAME", "")
    os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", "")

    client = MlflowClient()

    model_uri = os.getenv("MODEL_URI", "").strip()
    model_name = os.getenv("MODEL_NAME", "").strip()
    to_stage = os.getenv("TO_STAGE", "Staging").strip() or "Staging"

    if not model_uri and not model_name:
        raise RuntimeError("error")
      
    if model_uri:
        source_uri = model_uri
    else:
        source_uri = f"models:/{model_name}/Staging"

    est = mlflow.sklearn.load_model(source_uri)

    register_name = model_name or os.getenv("REGISTER_NAME", "st126130-a3-model")

    with mlflow.start_run():
        mlflow.sklearn.log_model(estimator=est, artifact_path="model",
                                 registered_model_name=register_name)

    versions = client.search_model_versions(f"name='{register_name}'")
    latest = max(versions, key=lambda v: int(v.creation_timestamp))
    client.transition_model_version_stage(
        name=register_name, version=latest.version, stage=to_stage, archive_existing_versions=False
    )
    print(f"[OK] Relogged '{register_name}' to stage '{to_stage}' as version {latest.version}")

if __name__ == "__main__":
    main()
