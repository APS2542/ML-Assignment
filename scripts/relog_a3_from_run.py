import argparse, os, pickle
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

def load_sklearn_from_stage(model_name: str, stage: str):
    client = MlflowClient()
    vers = client.get_latest_versions(model_name, [stage])
    if not vers:
        raise RuntimeError(f"No version in stage '{stage}' for model '{model_name}'")
    v = vers[0]
    run_id = v.run_id
    local = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/model")
    with open(Path(local)/"model.pkl", "rb") as f:
        sk_model = pickle.load(f)
    return sk_model

def relog_and_register(sk_model, model_name: str, promote_stage: str|None):
    with mlflow.start_run():
        mlflow.sklearn.log_model(sk_model, artifact_path="model")
        run_id = mlflow.active_run().info.run_id
    client = MlflowClient()
    mv = client.create_model_version(
        name=model_name, source=f"runs:/{run_id}/model", run_id=run_id
    )
    if promote_stage:
        client.transition_model_version_stage(
            name=model_name, version=mv.version,
            stage=promote_stage, archive_existing_versions=True,
        )
    uri = f"models:/{model_name}/{promote_stage or mv.version}"
    print(f"NEW_MODEL_URI={uri}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", required=True)      # เช่น st126130-a3-model
    ap.add_argument("--from-stage", default="Staging")
    ap.add_argument("--promote-stage", default="Staging")
    args = ap.parse_args()
    assert os.getenv("MLFLOW_TRACKING_URI"), "MLFLOW_TRACKING_URI not set"
    sk_model = load_sklearn_from_stage(args.model_name, args.from_stage)
    relog_and_register(sk_model, args.model_name, args.promote_stage)

if __name__ == "__main__":
    main()
