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
    return datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S UTC")


def get_json(url: str, api_key: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def find_final(output) -> dict:
    """Extract the status=='done' dict from a job output (dict or list of yields)."""
    candidates = output if isinstance(output, list) else [output]
    for item in reversed(candidates):
        if not isinstance(item, dict):
            continue
        if item.get("status") == "done":
            return item
        nested = item.get("output")
        if isinstance(nested, dict) and nested.get("status") == "done":
            return nested
    return {}


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
        if out.get("images") is not None:
            total = out.get("total") or "?"
            print(f"[{ts()}] 📥 Dataset images downloaded: {out['images']}/{total}", flush=True)
        else:
            print(f"[{ts()}] 📥 Downloading dataset...", flush=True)
    elif s == "resizing":
        if out.get("images") is not None:
            total = out.get("total") or "?"
            print(f"[{ts()}] 🪄 Resizing images: {out['images']}/{total}", flush=True)
        else:
            print(f"[{ts()}] 🪄 Resizing images...", flush=True)
    elif s == "releasing":
        print(f"[{ts()}] 🚀 Creating GitHub release...", flush=True)
    elif s == "evaluating":
        if out.get("images") is not None:
            total = out.get("total") or "?"
            print(f"[{ts()}] 📊 Downloading test set: {out['images']}/{total}", flush=True)
        else:
            print(f"[{ts()}] 📊 Running model evaluation...", flush=True)
    elif s not in ("done", ""):
        print(f"[{ts()}] {out}", flush=True)


def main():
    job_id = os.environ["JOB_ID"]
    api_key = os.environ["RUNPOD_API_KEY"]
    endpoint = f"https://api.runpod.ai/v2/{os.environ['RUNPOD_ENDPOINT_ID']}"

    terminal = {"COMPLETED", "FAILED", "TIMED_OUT", "CANCELLED"}
    printed = set()
    while True:
        try:
            data = get_json(f"{endpoint}/stream/{job_id}", api_key)
        except Exception as e:
            print(f"[{ts()}] Poll error: {e} — retrying in 15 s", flush=True)
            time.sleep(15)
            continue

        status = data.get("status", "UNKNOWN")

        for item in data.get("stream", []):
            out = item.get("output", {}) if isinstance(item, dict) else {}
            key = json.dumps(out, sort_keys=True, default=str)
            if key not in printed:
                printed.add(key)
                print_item(out)

        if status == "IN_QUEUE":
            print(f"[{ts()}] ⏳ In queue...", flush=True)

        elif status in terminal:
            # /stream can report COMPLETED even for failed jobs — /status is the truth
            try:
                final_data = get_json(f"{endpoint}/status/{job_id}", api_key)
            except Exception:
                final_data = data
            real_status = final_data.get("status", status)

            if real_status == "COMPLETED":
                final = find_final(final_data.get("output"))
                print(f"[{ts()}] ✅ Training completed successfully.", flush=True)
                print(f"  Model size : {final.get('model_size_mb', '—')} MB", flush=True)
                if final.get("released") is False and final.get("skip_reason"):
                    print(f"  Release    : SKIPPED — {final.get('skip_reason')}", flush=True)
                else:
                    print(f"  Release    : {final.get('release_url', '—')}", flush=True)
                if final.get("report_url"):
                    print(f"  Report     : {final.get('report_url')}", flush=True)
                sys.exit(0)

            if real_status == "TIMED_OUT":
                print(f"[{ts()}] ⏱️ Job timed out on RunPod "
                      "(increase Execution Timeout in endpoint settings).", flush=True)
            elif real_status == "CANCELLED":
                print(f"[{ts()}] 🚫 Job was cancelled.", flush=True)
            else:
                print(f"[{ts()}] ❌ Training failed.", flush=True)
                print(final_data.get("error") or str(final_data), flush=True)
            sys.exit(1)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
