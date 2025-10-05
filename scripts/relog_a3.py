import os, sys, traceback
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

def die(msg, code=1):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)

def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))
    os.environ.setdefault("MLFLOW_TRACKING_INSECURE_TLS", "true")

    model_uri = (os.getenv("MODEL_URI") or "").strip()
    model_name = (os.getenv("MODEL_NAME") or "").strip()
    to_stage = (os.getenv("TO_STAGE") or "Staging").strip() or "Staging"

    print("[INFO] Tracking URI:", os.getenv("MLFLOW_TRACKING_URI",""))
    print("[INFO] MODEL_URI:", "SET" if model_uri else "EMPTY")
    print("[INFO] MODEL_NAME:", model_name or "EMPTY")
    print("[INFO] Target stage:", to_stage)

    if not model_uri and not model_name:
        die("Either MODEL_URI or MODEL_NAME must be provided.")

    # เลือก source ให้แน่นอน
    source = model_uri if model_uri else f"models:/{model_name}/Staging"
    print("[INFO] Loading model from:", source)

    try:
        est = mlflow.sklearn.load_model(source)
        print("[INFO] Loaded sklearn model:", type(est).__name__)
    except Exception:
        traceback.print_exc()
        die("Failed to load model via mlflow.sklearn.load_model().

    register_name = model_name or os.getenv("REGISTER_NAME", "st126130-a3-model")
    print("[INFO] Registering as:", register_name)

    client = MlflowClient()

    with mlflow.start_run():
        mlflow.sklearn.log_model(estimator=est, artifact_path="model",
                                 registered_model_name=register_name)
        print("[INFO] Logged new version under Registered Model:", register_name)

    versions = client.search_model_versions(f"name='{register_name}'")
    latest = max(versions, key=lambda v: int(v.creation_timestamp))
    client.transition_model_version_stage(
        name=register_name, version=latest.version,
        stage=to_stage, archive_existing_versions=False
    )
    print(f"[OK] Relogged '{register_name}' to stage '{to_stage}' (version {latest.version})")

if __name__ == "__main__":
    main()
