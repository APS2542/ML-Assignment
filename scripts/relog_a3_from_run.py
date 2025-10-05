import os
import argparse
import pickle
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient


def _find_first(root: Path, filename: str) -> Path:
    for p in root.rglob(filename):
        return p
    raise FileNotFoundError(f"{filename} not found under {root}")


def load_sklearn_from_stage(model_name: str, stage: str):
    client = MlflowClient()
    vers = client.get_latest_versions(model_name, [stage])
    if not vers:
        raise RuntimeError(f"No version in stage '{stage}' for '{model_name}'")

    run_id = vers[0].run_id
    local = Path(mlflow.artifacts.download_artifacts(f"runs:/{run_id}/model"))

    candidates = [
        local / "model.pkl",                         
        local / "data" / "model_dir" / "model.pkl",     
        local / "model_dir" / "model.pkl",             
    ]
    for c in candidates:
        if c.exists():
            with open(c, "rb") as f:
                return pickle.load(f)
                
    p = _find_first(local, "model.pkl")
    with open(p, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME", "st126130-a3-model"))
    parser.add_argument("--from-stage", default="Staging")
    parser.add_argument("--to-stage", default="Staging")
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    sk_model = load_sklearn_from_stage(args.model_name, args.from_stage)

    with mlflow.start_run(run_name="relog-a3-linux") as r:
        mlflow.sklearn.log_model(
            sk_model,
            artifact_path="model",
            registered_model_name=args.model_name,
        )
        new_run_id = r.info.run_id

    client = MlflowClient()
    versions = [
        mv for mv in client.search_model_versions(f"name='{args.model_name}'")
        if mv.run_id == new_run_id
    ]
    mv = max(versions, key=lambda v: int(v.version))
    client.transition_model_version_stage(
        name=args.model_name,
        version=mv.version,
        stage=args.to_stage,
        archive_existing_versions=False,
    )

    print(f"NEW_MODEL_URI=models:/{args.model_name}/{args.to_stage}")


if __name__ == "__main__":
    main()
