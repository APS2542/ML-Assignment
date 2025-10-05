import argparse
import os
import time
import mlflow

POLL_SECS = 2
TIMEOUT_SECS = 300

def wait_until_ready(client, model_name, version):
    start = time.time()
    while True:
        mv = client.get_model_version(name=model_name, version=version)
        if mv.status == "READY":
            return mv
        if time.time() - start > TIMEOUT_SECS:
            raise TimeoutError(f"Model version {version} not READY in time (status={mv.status}).")
        time.sleep(POLL_SECS)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--from-stage", default="Staging")
    ap.add_argument("--to-stage", default="Staging")
    args = ap.parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is not set")

    client = mlflow.MlflowClient()

    vers = client.get_latest_versions(args.model_name, [args.from_stage])
    if not vers:
        raise RuntimeError(f"No versions found in stage {args.from_stage} for {args.model_name}")
    v = vers[0]

    source_uri = v.source

    if not source_uri.rstrip("/").endswith("/model"):
        source_uri = source_uri.rstrip("/") + "/model"

    print(f"Re-registering from source: {source_uri}")

    new_ver = mlflow.register_model(source=source_uri, name=args.model_name)
    print(f"Registered new version: {new_ver.version}, status={new_ver.status}")

    mv = wait_until_ready(client, args.model_name, new_ver.version)
    print(f"Model version READY: {mv.version}")

    if args.to_stage:
        client.transition_model_version_stage(
            name=args.model_name,
            version=mv.version,
            stage=args.to_stage,
            archive_existing_versions=False,
        )
        print(f"Transitioned version {mv.version} -> {args.to_stage}")

if __name__ == "__main__":
    main()

