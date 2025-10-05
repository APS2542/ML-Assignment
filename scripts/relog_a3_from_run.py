import argparse, os, time, re
import mlflow
from mlflow.tracking import MlflowClient

POLL_SECS = 2
TIMEOUT_SECS = 300

def wait_ready(client: MlflowClient, name: str, version: str):
    start = time.time()
    while True:
        mv = client.get_model_version(name=name, version=version)
        if mv.status == "READY":
            return mv
        if time.time() - start > TIMEOUT_SECS:
            raise TimeoutError(f"Model version {version} not READY (status={mv.status})")
        time.sleep(POLL_SECS)

def resolve_source(client: MlflowClient, model_uri: str | None,
                   model_name: str | None, from_stage: str):
    """
    Return (source_uri, register_name)
    """
    if model_uri:
        uri = model_uri.strip()
        if uri.startswith("runs:/"):
            if not model_name:
                raise RuntimeError("MODEL_NAME is required when using runs:/ URI")
            source = uri
            return source if source.endswith("/model") else source.rstrip("/") + "/model", model_name

        if uri.startswith("models:/"):
            # models:/<name>/<Stage|Version>
            m = re.match(r"models:/([^/]+)/(.+)$", uri)
            if not m:
                raise RuntimeError(f"Bad models URI: {uri}")
            name, ref = m.group(1), m.group(2)
            register_name = model_name or name
            if ref.isdigit():
                mv = client.get_model_version(name=name, version=ref)
                src = mv.source
            else:
                vers = client.get_latest_versions(name=name, stages=[ref])
                if not vers:
                    raise RuntimeError(f"No versions found for {name} in stage {ref}")
                src = vers[0].source
            return src if src.rstrip("/").endswith("/model") else src.rstrip("/") + "/model", register_name

        # อื่นๆ: ถือว่าเป็น artifact path ตรง ๆ
        return model_uri, model_name

    # ไม่มี model_uri → ใช้ latest version ของ stage ที่กำหนด
    if not model_name:
        raise RuntimeError("MODEL_NAME is required when --model-uri is not set")
    vers = client.get_latest_versions(model_name, [from_stage])
    if not vers:
        raise RuntimeError(f"No versions in stage {from_stage} for model {model_name}")
    src = vers[0].source
    return src if src.rstrip("/").endswith("/model") else src.rstrip("/") + "/model", model_name

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default=os.getenv("MODEL_NAME"))
    ap.add_argument("--model-uri", default=os.getenv("MODEL_URI"))  # runs:/.../model หรือ models:/name/Stage
    ap.add_argument("--from-stage", default=os.getenv("FROM_STAGE", "Staging"))
    ap.add_argument("--to-stage",   default=os.getenv("TO_STAGE",   "Staging"))
    args = ap.parse_args()

    if not os.getenv("MLFLOW_TRACKING_URI"):
        raise RuntimeError("MLFLOW_TRACKING_URI is not set")

    client = MlflowClient()

    source, register_name = resolve_source(client, args.model_uri, args.model_name, args.from_stage)
    print(f"[relog] source: {source}")
    print(f"[relog] register as: {register_name}")

    new_ver = mlflow.register_model(source=source, name=register_name)
    print(f"[relog] registered version: {new_ver.version}, status={new_ver.status}")

    mv = wait_ready(client, register_name, new_ver.version)
    print(f"[relog] READY: {mv.version}")

    if args.to_stage:
        client.transition_model_version_stage(
            name=register_name,
            version=mv.version,
            stage=args.to_stage,
            archive_existing_versions=False,
        )
        print(f"[relog] transitioned to stage: {args.to_stage}")

if __name__ == "__main__":
    main()
