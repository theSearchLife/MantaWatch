"""
RunPod Serverless handler — YOLO manta classification training.

Input:
    dataset_url   : Google Drive or direct URL to .zip/.tar.gz dataset
                    (default: DEFAULT_DATASET_URL)
    epochs        : int   (default 100)
    base_model    : str   (default "yolo11s-cls.pt")
    imgsz         : int   (default 640)
    batch         : int   (default 32)
    patience      : int   (default 20)
    run_name      : str   (auto-generated if omitted)
    release_tag   : str   release tag, e.g. "2025-01-15"  (skip release if omitted)
    github_token  : str   PAT with contents:write         (skip release if omitted)
    repo          : str   "owner/repo"                    (skip release if omitted)

Output (streaming):
    Yields per-epoch progress every PROGRESS_INTERVAL epochs, then final result.
    { status, run_name, model_size_mb, metrics_last_line, release_url }
"""
import datetime
import os
import pathlib
import queue
import re
import shutil
import subprocess
import threading
import zipfile

import runpod

DEFAULT_DATASET_URL  = "https://drive.google.com/uc?id=1SfFNtGMKP0dkqEqLAQNJT76_vP7LVwkC"
PROGRESS_INTERVAL    = 5   # yield a progress update every N epochs

ARCHIVE_PATH = "/tmp/dataset.archive"
EXTRACT_DIR  = "/tmp/dataset_train"
RUNS_DIR     = "/tmp/runs"


def log(msg: str):
    print(msg, flush=True)


# Dataset helpers

def download_dataset(url: str):
    if os.path.exists(ARCHIVE_PATH):
        os.remove(ARCHIVE_PATH)
    if "drive.google.com" in url:
        m = re.search(r"(?:id=|/d/)([a-zA-Z0-9_-]{20,})", url)
        file_id = m.group(1) if m else url
        log(f"  Google Drive ID: {file_id}")
        subprocess.run(["gdown", file_id, "-O", ARCHIVE_PATH], check=True)
    else:
        log(f"  Direct download: {url}")
        subprocess.run(["wget", "-q", url, "-O", ARCHIVE_PATH], check=True)
    log(f"  Downloaded: {os.path.getsize(ARCHIVE_PATH) / 1024 / 1024:.0f} MB")


def extract_archive():
    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)
    os.makedirs(EXTRACT_DIR)
    with open(ARCHIVE_PATH, "rb") as f:
        magic = f.read(4).hex()
    if magic.startswith("504b"):
        log("  Detected ZIP")
        with zipfile.ZipFile(ARCHIVE_PATH) as zf:
            zf.extractall(EXTRACT_DIR)
    elif magic.startswith("1f8b"):
        log("  Detected tar.gz")
        subprocess.run(["tar", "xzf", ARCHIVE_PATH, "-C", EXTRACT_DIR, "--no-same-owner"], check=True)
    else:
        raise RuntimeError(f"Unknown archive type (magic={magic})")


def find_dataset_root() -> str:
    result = subprocess.run(
        ["find", EXTRACT_DIR, "-maxdepth", "3", "-type", "d", "-name", "train"],
        capture_output=True, text=True,
    )
    dirs = [p.strip() for p in result.stdout.strip().splitlines() if p.strip()]
    if not dirs:
        raise RuntimeError("No 'train' directory found in archive")
    root = str(pathlib.Path(dirs[0]).parent)
    log(f"  Dataset root: {root}")
    log(f"  Classes: {os.listdir(os.path.join(root, 'train'))}")
    return root


# Training with progress streaming

def _train_with_progress(dataset_root, run_name, base_model, epochs, imgsz, batch, patience):
    """Generator: runs YOLO in a background thread, yields epoch metrics every PROGRESS_INTERVAL epochs.
    Returns the path to best.pt via StopIteration.value (i.e. `result = yield from _train_with_progress(...)`).
    """
    progress_q    = queue.Queue()
    error_holder  = [None]
    result_holder = [None]

    def on_epoch_end(trainer):
        epoch = trainer.epoch + 1  # trainer.epoch is 0-indexed
        total = trainer.epochs
        if epoch % PROGRESS_INTERVAL == 0 or epoch == total:
            loss = top1 = None
            try:
                loss = round(float(trainer.loss), 4)
            except Exception:
                pass
            try:
                top1 = round(float(trainer.metrics.get("metrics/accuracy_top1", 0)) * 100, 2)
            except Exception:
                pass
            progress_q.put({
                "status":       "training",
                "epoch":        epoch,
                "total_epochs": total,
                "loss":         loss,
                "top1":         top1,
            })

    def run():
        try:
            from ultralytics import YOLO
            model = YOLO(base_model)
            model.add_callback("on_train_epoch_end", on_epoch_end)
            model.train(
                data=dataset_root,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                patience=patience,
                project=RUNS_DIR,
                name=run_name,
                exist_ok=True,
            )
            best = pathlib.Path(RUNS_DIR) / run_name / "weights" / "best.pt"
            if not best.exists():
                raise RuntimeError(f"best.pt not found at {best}")
            result_holder[0] = str(best)
        except Exception as e:
            error_holder[0] = e
        finally:
            progress_q.put(None)  # sentinel — always signals the generator to stop waiting

    t = threading.Thread(target=run, daemon=True)
    t.start()

    while True:
        item = progress_q.get()
        if item is None:
            break
        yield item

    t.join()

    if error_holder[0]:
        raise error_holder[0]

    return result_holder[0]


# GitHub release

def last_metrics(run_name: str) -> str:
    csv = pathlib.Path(RUNS_DIR) / run_name / "results.csv"
    if csv.exists():
        lines = csv.read_text().strip().splitlines()
        return lines[-1] if lines else ""
    return ""


def create_github_release(model_path: str, token: str, repo: str, tag: str) -> str:
    """Create (or overwrite) a GitHub release and upload model.pt. Returns download URL."""
    import requests
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    base = f"https://api.github.com/repos/{repo}"

    existing = requests.get(f"{base}/releases/tags/{tag}", headers=headers, timeout=10)
    if existing.status_code == 200:
        release_id = existing.json()["id"]
        requests.delete(f"{base}/releases/{release_id}", headers=headers, timeout=10)
        requests.delete(f"{base}/git/refs/tags/{tag}", headers=headers, timeout=10)
        log(f"  Deleted existing release {tag}")

    resp = requests.post(
        f"{base}/releases",
        headers=headers,
        json={
            "tag_name": tag,
            "name": f"Model {tag}",
            "body": "YOLO11s-cls · 3 classes: manta / other_fish / non_fish\n\nAuto-generated by RunPod training job.",
            "draft": False,
            "prerelease": False,
        },
        timeout=30,
    )
    resp.raise_for_status()
    upload_url = resp.json()["upload_url"].replace("{?name,label}", "")

    with open(model_path, "rb") as f:
        up = requests.post(
            f"{upload_url}?name=model.pt",
            headers={**headers, "Content-Type": "application/octet-stream"},
            data=f,
            timeout=120,
        )
    up.raise_for_status()
    return up.json()["browser_download_url"]


# RunPod handler (generator — enables /stream endpoint)

def handler(job: dict):
    inp = job.get("input", {})

    dataset_url  = inp.get("dataset_url",  DEFAULT_DATASET_URL)
    epochs       = int(inp.get("epochs",   100))
    base_model   = inp.get("base_model",   "yolo11s-cls.pt")
    imgsz        = int(inp.get("imgsz",    640))
    batch        = int(inp.get("batch",    32))
    patience     = int(inp.get("patience", 20))
    run_name     = inp.get("run_name") or f"run-{datetime.datetime.utcnow():%Y%m%d-%H%M%S}"
    release_tag  = inp.get("release_tag")
    github_token = inp.get("github_token")
    repo         = inp.get("repo")

    log(f"=== Job start: {run_name} ===")

    try:
        yield {"status": "downloading"}
        log("==> Downloading dataset...")
        download_dataset(dataset_url)

        yield {"status": "extracting"}
        log("==> Extracting...")
        extract_archive()
        dataset_root = find_dataset_root()

        log("==> Training...")
        best_pt = yield from _train_with_progress(
            dataset_root, run_name, base_model, epochs, imgsz, batch, patience
        )
        log("==> Training complete.")

        size_mb     = round(os.path.getsize(best_pt) / 1024 / 1024, 1)
        metrics_csv = last_metrics(run_name)
        release_url = None

        if release_tag and github_token and repo:
            yield {"status": "releasing", "tag": release_tag}
            log(f"==> Creating GitHub release {release_tag}...")
            release_url = create_github_release(best_pt, github_token, repo, release_tag)
            log(f"  Release: {release_url}")
        else:
            log("  Skipping GitHub release (no tag/token/repo provided)")

        yield {
            "status":            "done",
            "run_name":          run_name,
            "model_size_mb":     size_mb,
            "metrics_last_line": metrics_csv,
            "release_url":       release_url,
        }

    except Exception as exc:
        import traceback
        log(f"ERROR: {exc}")
        log(traceback.format_exc())
        raise  # RunPod marks job as FAILED


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
