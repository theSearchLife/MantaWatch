"""
Poll a RunPod Serverless streaming job until it reaches a terminal state.

Reads credentials from environment variables:
    JOB_ID              RunPod job ID
    RUNPOD_API_KEY      RunPod API key
    RUNPOD_ENDPOINT_ID  RunPod endpoint ID
"""
import datetime
import json
import os
import sys
import time
import urllib.request


POLL_INTERVAL = 30  # seconds between /stream polls


def ts() -> str:
    return datetime.datetime.utcnow().strftime("%H:%M:%S UTC")


def get_stream(endpoint: str, api_key: str, job_id: str) -> dict:
    req = urllib.request.Request(
        f"{endpoint}/stream/{job_id}",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def print_item(out: dict):
    s = out.get("status", "")
    if s == "training":
        epoch = out.get("epoch", "?")
        total = out.get("total_epochs", "?")
        loss  = out.get("loss")
        top1  = out.get("top1")
        msg   = f"[{ts()}] 🔄 Epoch {epoch}/{total}"
        if loss is not None:
            msg += f"  loss={loss:.4f}"
        if top1 is not None:
            msg += f"  top1={top1}%"
        print(msg, flush=True)
    elif s == "downloading":
        print(f"[{ts()}] 📥 Downloading dataset...", flush=True)
    elif s == "extracting":
        print(f"[{ts()}] 📦 Extracting archive...", flush=True)
    elif s == "releasing":
        print(f"[{ts()}] 🚀 Creating GitHub release...", flush=True)
    elif s not in ("done", ""):
        print(f"[{ts()}] {out}", flush=True)


def main():
    job_id      = os.environ["JOB_ID"]
    api_key     = os.environ["RUNPOD_API_KEY"]
    endpoint    = f"https://api.runpod.ai/v2/{os.environ['RUNPOD_ENDPOINT_ID']}"

    seen = 0
    while True:
        try:
            data = get_stream(endpoint, api_key, job_id)
        except Exception as e:
            print(f"[{ts()}] Poll error: {e} — retrying in 15 s", flush=True)
            time.sleep(15)
            continue

        status = data.get("status", "UNKNOWN")
        items  = data.get("stream", [])

        for item in items[seen:]:
            print_item(item.get("output", {}))
        seen = len(items)

        if status == "IN_QUEUE":
            print(f"[{ts()}] ⏳ In queue...", flush=True)

        elif status == "COMPLETED":
            final = next(
                (item.get("output", {}) for item in items
                 if isinstance(item.get("output"), dict)
                 and item["output"].get("status") == "done"),
                {},
            )
            print(f"[{ts()}] ✅ Training completed successfully.", flush=True)
            print(f"  Model size : {final.get('model_size_mb', '—')} MB", flush=True)
            print(f"  Release    : {final.get('release_url', '—')}", flush=True)
            sys.exit(0)

        elif status == "FAILED":
            print(f"[{ts()}] ❌ Training failed.", flush=True)
            print(data.get("error") or data.get("output") or str(data), flush=True)
            sys.exit(1)

        elif status == "TIMED_OUT":
            print(f"[{ts()}] ⏱️ Job timed out on RunPod (increase Execution Timeout in endpoint settings).", flush=True)
            sys.exit(1)

        elif status == "CANCELLED":
            print(f"[{ts()}] 🚫 Job was cancelled.", flush=True)
            sys.exit(1)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
