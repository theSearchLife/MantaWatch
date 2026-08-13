"""
RunPod Serverless handler — YOLO image classification training + evaluation.

Input:
    dataset_url      : Google Drive folder link (with train/, val/, test/ subfolders; the
                       target class folder is prefixed 'the_', the rest 'other_'/'non_').
    drive_sa_b64     : base64-encoded service-account JSON for authenticated Drive downloads.
    epochs           : int   (default 100)
    base_model       : str   (default "yolo11s-cls.pt")
    imgsz            : int   (default 640)
    batch            : int   (default 32)
    patience         : int   (default 20)
    run_name         : str   (auto-generated if omitted)
    release_tag      : str   e.g. "2025-01-15"  (skip release if omitted)
    github_token     : str   PAT with contents:write
    repo             : str   "owner/repo"

Output (streaming):
    Yields epoch progress every PROGRESS_INTERVAL epochs, then final result dict.
    { status, run_name, model_size_mb, release_url, report_url, released, skip_reason }
"""
import base64
import datetime
import os
import pathlib
import queue
import re
import shutil
import threading
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import runpod

# cuBLAS workspace + filter silence ultralytics' per-step determinism warning
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
warnings.filterwarnings("ignore", message="Deterministic behavior was enabled")

PROGRESS_INTERVAL = 5

DATA_DIR = "/tmp/dataset"       # train/ + val/
EVAL_DIR = "/tmp/dataset_eval"  # test/
PREV_MODEL = "/tmp/model_prev.pt"
RUNS_DIR = "/tmp/runs"

DOWNLOAD_THREADS = 16
EVAL_PREDICT_BATCH = 16  # inference batch size during eval
EVAL_LOADER_WORKERS = min(8, os.cpu_count() or 4)  # DataLoader workers for eval preprocessing
RESIZE_MAX_SIDE = 1024   # downsize long side before train/eval; Drive originals untouched
RESIZE_THREADS = os.cpu_count() or 8

# Class config is derived from the dataset folders in _apply_class_config(): the target
# is the folder prefixed 'the_' (e.g. the_manta), the rest are other_*/non_*.
CLASSES = []
TARGET_FOLDER = ""
TARGET_CLASS = ""
NON_TARGET_CLASS = ""
REPORT_CLASSES = []


def log(msg: str):
    print(msg, flush=True)


def _cgroup_mem():
    """Return (usage_mb, limit_mb) for this container's cgroup, or (None, None).

    This is the memory the OOM-killer actually watches — /proc/meminfo inside a
    container reports the host, not the cgroup limit.
    """
    pairs = [
        ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),                      # v2
        ("/sys/fs/cgroup/memory/memory.usage_in_bytes",
         "/sys/fs/cgroup/memory/memory.limit_in_bytes"),                                      # v1
    ]
    for usage_f, limit_f in pairs:
        try:
            usage = int(open(usage_f).read().strip())
            raw = open(limit_f).read().strip()
            limit = None if raw == "max" else int(raw)
            mb = 1024 * 1024
            if limit and limit > (1 << 62):  # v1 sentinel for "unlimited"
                limit = None
            return usage // mb, (limit // mb if limit else None)
        except Exception:
            continue
    return None, None


def log_mem(tag: str):
    usage, limit = _cgroup_mem()
    if usage is not None:
        detail = f" / {limit} MB limit ({100 * usage // limit}%)" if limit else ""
        log(f"  [mem] {tag}: {usage} MB used{detail}")


def _start_mem_logger(interval: int = 300):
    """Log cgroup memory every `interval`s on a daemon thread. Returns a stop Event."""
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            log_mem("sample")
            stop.wait(interval)

    threading.Thread(target=loop, daemon=True).start()
    return stop


def _free_torch_memory():
    """Run Python GC and release cached CUDA memory."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _drive_token(sa_b64: str) -> str:
    """Mint a Drive read-only access token from a base64-encoded service-account JSON.

    A service account is authenticated (12k requests/min, 750 GB/day per the Drive
    API quotas, counted per principal not per IP), so it avoids the anonymous
    API-key "downloadQuotaExceeded" wall and the RunPod shared-IP throttling that
    breaks bulk per-file downloads.
    """
    import json
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request

    info = json.loads(base64.b64decode(sa_b64))
    creds = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    creds.refresh(Request())
    return creds.token


def _remove_corrupt_images(root: str) -> int:
    """Delete images unreadable by PIL or OpenCV (what YOLO uses). Returns count."""
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    import cv2

    exts = {".jpg", ".jpeg", ".png"}
    paths = [p for p in pathlib.Path(root).rglob("*") if p.suffix.lower() in exts]

    def is_bad(p: pathlib.Path) -> bool:
        try:
            with Image.open(p) as img:
                img.load()
        except Exception:
            return True
        return cv2.imread(str(p)) is None

    with ThreadPoolExecutor(8) as pool:
        flags = list(pool.map(is_bad, paths))
    removed = 0
    for p, bad in zip(paths, flags):
        if bad:
            p.unlink(missing_ok=True)
            removed += 1
    return removed


def _resize_one(path: pathlib.Path, max_side: int) -> bool:
    from PIL import Image
    try:
        with Image.open(path) as im:
            im.load()
            w, h = im.size
            fmt = im.format
            if max(w, h) <= max_side:
                return False
            scale = max_side / float(max(w, h))
            resized = im.resize(
                (max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR
            )
        opts = {"quality": 90} if path.suffix.lower() in (".jpg", ".jpeg") else {}
        resized.save(path, format=fmt, **opts)
        return True
    except Exception:
        return False


def _resize_images_inplace(root: str, max_side: int, progress_cb=None) -> int:
    """Downscale every image under root so its long side is <= max_side, in place.

    Source photos are multi-megapixel; decoding them each epoch is what pins the
    CPU and starves the GPU. The model resizes to imgsz anyway, so shrinking once
    turns a per-epoch full-res decode into a one-off cost. Returns images rewritten.
    """
    from concurrent.futures import ThreadPoolExecutor

    exts = {".jpg", ".jpeg", ".png"}
    paths = [p for p in pathlib.Path(root).rglob("*") if p.suffix.lower() in exts]
    total = len(paths)
    if not total:
        return 0
    log(f"  Resizing {total} images to <= {max_side}px (long side)...")

    counts = {"done": 0, "shrunk": 0}
    lock = threading.Lock()

    def work(p):
        shrunk = _resize_one(p, max_side)
        with lock:
            counts["done"] += 1
            counts["shrunk"] += int(shrunk)
            done = counts["done"]
        if done % 500 == 0 and progress_cb:
            progress_cb(done, total)

    with ThreadPoolExecutor(RESIZE_THREADS) as pool:
        list(pool.map(work, paths))
    log(f"  Resized {counts['shrunk']}/{total} (others already small)")
    return counts["shrunk"]


def _gdrive_list_folder(folder_id: str, token: str) -> list:
    """Recursively list a Drive folder via the API (service-account auth).
    Returns [(file_id, relative_path), ...].
    """
    import requests

    headers = {"Authorization": f"Bearer {token}"}
    files = []

    def walk(fid: str, prefix: str):
        page_token = None
        while True:
            params = {
                "q": f"'{fid}' in parents and trashed=false",
                "fields": "nextPageToken,files(id,name,mimeType)",
                "pageSize": 1000,
                "supportsAllDrives": "true",
                "includeItemsFromAllDrives": "true",
            }
            if page_token:
                params["pageToken"] = page_token
            r = requests.get("https://www.googleapis.com/drive/v3/files",
                             params=params, headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
            for f in data.get("files", []):
                if f["mimeType"] == "application/vnd.google-apps.folder":
                    walk(f["id"], prefix + f["name"] + "/")
                else:
                    files.append((f["id"], prefix + f["name"]))
            page_token = data.get("nextPageToken")
            if not page_token:
                return

    walk(folder_id, "")
    return files


def _gdrive_download_file(file_id: str, dest: pathlib.Path, token: str, session):
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
    headers = {"Authorization": f"Bearer {token}"}
    for attempt in range(3):
        try:
            with session.get(url,
                             params={"alt": "media", "supportsAllDrives": "true"},
                             headers=headers, stream=True, timeout=(10, 30)) as r:
                r.raise_for_status()
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
                    if hasattr(os, "posix_fadvise"):
                        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def _download_drive_items(items: list, dest: str, token: str, progress_cb=None):
    """Download [(file_id, rel_path), ...] into dest, preserving the rel structure.

    fsync + fadvise after each file keep the page cache flat so the container
    memory watchdog is not tripped by the write volume.
    """
    import requests
    from concurrent.futures import ThreadPoolExecutor

    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
    if not items:
        raise RuntimeError(
            "No images to download — check the folder link and that it is shared "
            "with the service account"
        )
    log(f"  {len(items)} images to download")

    counts = {"done": 0}
    lock = threading.Lock()
    tls = threading.local()

    def work(item):
        fid, rel = item
        if not hasattr(tls, "session"):
            tls.session = requests.Session()
        parts = pathlib.PurePosixPath(rel).parts
        _gdrive_download_file(fid, pathlib.Path(dest, *parts), token, tls.session)
        with lock:
            counts["done"] += 1
            done = counts["done"]
        if done % 500 == 0 and progress_cb:
            progress_cb(done, len(items))

    with ThreadPoolExecutor(DOWNLOAD_THREADS) as pool:
        list(pool.map(work, items))
    log(f"  Downloaded {counts['done']} images -> {dest}")


def _drive_folder_id(url: str):
    """Return the folder id if url is a Drive folder link, else None."""
    m = re.search(r"drive\.google\.com/drive/(?:u/\d+/)?folders/([a-zA-Z0-9_-]+)", url)
    return m.group(1) if m else None


def _partition_drive_listing(listing: list):
    """Split a Drive listing into (train_val_items, test_items), each [(file_id, rel)]
    with the rel relative to the dataset root (any extra nesting above the splits
    is stripped, so files land directly under train/ val/ test/).
    """
    exts = {".jpg", ".jpeg", ".png"}

    prefix = ""
    for _fid, rel in listing:
        parts = pathlib.PurePosixPath(rel).parts
        if "train" in parts:
            prefix = "/".join(parts[:parts.index("train")])
            break
    if prefix:
        prefix += "/"

    train_val, test = [], []
    for fid, rel in listing:
        if not rel.startswith(prefix):
            continue
        sub = rel[len(prefix):]
        sub_path = pathlib.PurePosixPath(sub)
        if len(sub_path.parts) < 2 or sub_path.suffix.lower() not in exts:
            continue
        if sub_path.parts[0] in ("train", "val"):
            train_val.append((fid, sub))
        elif sub_path.parts[0] == "test":
            test.append((fid, sub))
    return train_val, test


def _run_with_progress(fetch_fn, status: str):
    """Run fetch_fn(progress_cb=...) in a thread, yielding {status, images, total}."""
    progress_q = queue.Queue()
    error_holder = [None]

    def on_progress(done, total):
        progress_q.put({"status": status, "images": done, "total": total})

    def run():
        try:
            fetch_fn(progress_cb=on_progress)
        except Exception as exc:
            error_holder[0] = exc
        finally:
            progress_q.put(None)

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


def _train_with_progress(dataset_root, run_name, base_model, epochs, imgsz, batch, patience):
    """Train in a background thread, yielding epoch progress dicts.

    Returns the path to best.pt via StopIteration (use `yield from`).
    """
    progress_q = queue.Queue()
    error_holder = [None]
    result_holder = [None]

    def on_epoch_end(trainer):
        epoch = trainer.epoch + 1
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
                "status": "training", "epoch": epoch,
                "total_epochs": total, "loss": loss, "top1": top1,
            })

    def run():
        model = None
        try:
            from ultralytics import YOLO
            model = YOLO(base_model)
            model.add_callback("on_train_epoch_end", on_epoch_end)
            model.train(
                data=dataset_root, epochs=epochs, imgsz=imgsz,
                batch=batch, patience=patience,
                project=RUNS_DIR, name=run_name, exist_ok=True,
            )
            best = pathlib.Path(RUNS_DIR) / run_name / "weights" / "best.pt"
            if not best.exists():
                raise RuntimeError(f"best.pt not found at {best}")
            result_holder[0] = str(best)
        except Exception as e:
            error_holder[0] = e
        finally:
            # Ultralytics retains the model/trainer/cached dataset in RAM after
            # train(); release it before eval or the container OOMs (known issue).
            try:
                del model
            except Exception:
                pass
            _free_torch_memory()
            progress_q.put(None)

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


def last_metrics(run_name: str) -> str:
    csv = pathlib.Path(RUNS_DIR) / run_name / "results.csv"
    if csv.exists():
        lines = csv.read_text().strip().splitlines()
        return lines[-1] if lines else ""
    return ""


def _gh_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _export_onnx(model_path: str, imgsz: int) -> str:
    """Export the trained .pt to ONNX (class names + imgsz embedded in metadata)."""
    from ultralytics import YOLO
    return str(YOLO(model_path).export(format="onnx", imgsz=imgsz))


def create_github_release(model_path: str, token: str, repo: str, tag: str, imgsz: int = 640) -> str:
    import requests
    headers = _gh_headers(token)
    base = f"https://api.github.com/repos/{repo}"
    existing = requests.get(f"{base}/releases/tags/{tag}", headers=headers, timeout=10)
    if existing.status_code == 200:
        release_id = existing.json()["id"]
        requests.delete(f"{base}/releases/{release_id}", headers=headers, timeout=10)
        requests.delete(f"{base}/git/refs/tags/{tag}", headers=headers, timeout=10)
        log(f"  Deleted existing release {tag}")
    resp = requests.post(
        f"{base}/releases", headers=headers,
        json={
            "tag_name": tag, "name": f"Model {tag}",
            "body": f"YOLO11s-cls · {len(CLASSES)} classes, target: {TARGET_CLASS}\n\nAuto-generated by RunPod training job.",
            "draft": False, "prerelease": False,
        }, timeout=30,
    )
    resp.raise_for_status()
    release = resp.json()
    upload_url = release["upload_url"].replace("{?name,label}", "")

    def upload(path: str, name: str):
        with open(path, "rb") as f:
            up = requests.post(
                f"{upload_url}?name={name}",
                headers={**headers, "Content-Type": "application/octet-stream"},
                data=f, timeout=180,
            )
        up.raise_for_status()

    upload(model_path, "model.pt")

    # ONNX is best-effort: the .pt release already succeeded, so a failed export
    # must not abort the release.
    try:
        onnx_path = _export_onnx(model_path, imgsz)
        upload(onnx_path, "model.onnx")
        log("  Uploaded model.onnx")
    except Exception as exc:
        log(f"  WARNING: ONNX export/upload failed — {exc}")

    return release["html_url"]


def get_previous_model(token: str, repo: str, current_tag: str):
    """Download model.pt from the most recent published release before current_tag.

    Skips drafts and prereleases (their assets 404 on the public download URL and
    they are not real released models to compare against). Downloads via the
    authenticated asset API so it also works for private repos. Never raises —
    returns (None, None) on any failure so a comparison miss does not abort the run.
    """
    import requests
    headers = _gh_headers(token)
    base = f"https://api.github.com/repos/{repo}"
    try:
        resp = requests.get(f"{base}/releases", headers=headers, timeout=15)
        if resp.status_code != 200:
            log(f"  Could not fetch releases ({resp.status_code})")
            return (None, None)
        for release in resp.json():
            if release["tag_name"] == current_tag:
                continue
            if release.get("draft") or release.get("prerelease"):
                continue
            for asset in release.get("assets", []):
                if asset["name"] == "model.pt":
                    log(f"  Previous release: {release['tag_name']} ({asset['size'] // 1024 // 1024} MB)")
                    dl = requests.get(
                        f"{base}/releases/assets/{asset['id']}",
                        headers={**headers, "Accept": "application/octet-stream"},
                        timeout=120, stream=True,
                    )
                    dl.raise_for_status()
                    with open(PREV_MODEL, "wb") as f:
                        for chunk in dl.iter_content(1024 * 1024):
                            f.write(chunk)
                    return (PREV_MODEL, release["tag_name"])
        log("  No previous published release found")
        return (None, None)
    except Exception as exc:
        log(f"  Could not fetch previous model: {exc}")
        return (None, None)


def _fig_to_b64(fig) -> str:
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def _plot_confusion_matrix(cm, classes: list, title: str = "") -> str:
    import numpy as np
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=0, ha="center")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or "Confusion matrix")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}\n({cm[i,j]})",
                    ha="center", va="center", fontsize=8,
                    color="white" if cm_norm[i, j] > 0.6 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return _fig_to_b64(fig)


def _plot_roc_curves(fpr_tpr: dict, auc_per: dict, classes: list, title: str = "") -> str:
    colors = ["steelblue", "darkorange", "seagreen"]
    fig, ax = plt.subplots(figsize=(5, 4))
    for cls, color in zip(classes, colors):
        if cls in fpr_tpr:
            fpr, tpr = fpr_tpr[cls]
            auc = auc_per.get(cls) or 0.0
            ax.plot(fpr, tpr, color=color, label=f"{cls}  AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title or "ROC curves (one-vs-rest)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return _fig_to_b64(fig)


def _plot_error_grid(errors: list, name2idx: dict) -> str:
    if not errors:
        return ""
    import numpy as np
    from PIL import Image as PILImage
    ncols = 4
    nrows = max(1, (len(errors) + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.8))
    axes = np.array(axes).flatten()
    for ax, (path, true_lbl, pred_lbl, probs) in zip(axes, errors):
        try:
            img = PILImage.open(path).convert("RGB")
            img.thumbnail((220, 220))
            ax.imshow(img)
        except Exception:
            ax.set_facecolor("#eee")
        conf = float(probs[name2idx.get(pred_lbl, 0)])
        ax.set_title(f"T: {true_lbl}\nP: {pred_lbl} ({conf:.2f})", fontsize=7)
        ax.axis("off")
    for ax in axes[len(errors):]:
        ax.axis("off")
    fig.suptitle("Top misclassified — sorted by confidence", fontsize=9)
    plt.tight_layout()
    return _fig_to_b64(fig)


def _eval_metrics(y_true, y_pred, probs_arr, classes, name2idx, error_grid) -> dict:
    """Build the evaluation result dict from accumulated predictions.

    error_grid: list of (image_path, true, pred, probs) for the report's grid (<=12).
    n_errors is the total mismatch count, computed from the arrays.
    """
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
        confusion_matrix as sk_cm, roc_auc_score, roc_curve,
    )

    accuracy = round(float(accuracy_score(y_true, y_pred)) * 100, 2)
    prec_arr = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    rec_arr = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    f1_arr = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    f2_arr = fbeta_score(y_true, y_pred, beta=2, labels=classes, average=None, zero_division=0)

    y_bin = np.stack([(y_true == c).astype(int) for c in classes], axis=1)
    y_score = np.stack([probs_arr[:, name2idx[c]] for c in classes], axis=1)
    auc_per, fpr_tpr = {}, {}
    for i, cls in enumerate(classes):
        try:
            auc_per[cls] = round(float(roc_auc_score(y_bin[:, i], y_score[:, i])), 4)
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            fpr_tpr[cls] = (fpr.tolist(), tpr.tolist())
        except Exception:
            auc_per[cls] = None

    per_class = {
        cls: {
            "precision": round(float(prec_arr[i]) * 100, 2),
            "recall": round(float(rec_arr[i]) * 100, 2),
            "f1": round(float(f1_arr[i]) * 100, 2),
            "f2": round(float(f2_arr[i]) * 100, 2),
            "auc": auc_per.get(cls),
            "support": int((y_true == cls).sum()),
        }
        for i, cls in enumerate(classes)
    }
    return {
        "accuracy": accuracy,
        "per_class": per_class,
        "classes": classes,
        "total": int(len(y_true)),
        "n_errors": int((y_true != y_pred).sum()),
        "cm": sk_cm(y_true, y_pred, labels=classes),
        "fpr_tpr": fpr_tpr,
        "auc_per": auc_per,
        "errors": error_grid,
        "name2idx": name2idx,
    }


def _eval_predict(model, paths, imgsz):
    """Predict class probabilities, parallelising preprocessing across worker procs.

    ultralytics' predict preprocesses its source serially on one core, so for
    full-resolution photos the per-image decode+resize dominates eval. We push the
    images through a torch DataLoader (num_workers) using ultralytics' own
    classification transform, then run batched GPU inference. Same transform and math
    as predict (identical probabilities) — only the heavy preprocessing is parallel.
    """
    import cv2
    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    from ultralytics.data.augment import classify_transforms

    tfm = classify_transforms(imgsz)

    class _EvalDS(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            bgr = cv2.imread(str(paths[i]))
            if bgr is None:
                bgr = np.zeros((32, 32, 3), dtype=np.uint8)
            return tfm(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))

    loader = DataLoader(_EvalDS(), batch_size=EVAL_PREDICT_BATCH, shuffle=False,
                        num_workers=EVAL_LOADER_WORKERS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = model.model.to(device).eval()
    dtype = next(net.parameters()).dtype
    out = []
    for batch in loader:
        with torch.no_grad():
            logits = net(batch.to(device=device, dtype=dtype))
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        for row in logits.softmax(1).float().cpu().numpy():
            out.append(row)
    return out


def _to_report_class(name: str) -> str:
    """Collapse the model's native classes to the 2 report classes: the target folder maps
    to the target label, anything else to the non-target label.
    """
    return TARGET_CLASS if str(name) == TARGET_FOLDER else NON_TARGET_CLASS


def run_full_eval(model_path: str, eval_root: str, imgsz: int = 640) -> dict:
    """Evaluate one model on an on-disk class-folder dataset."""
    import numpy as np
    from PIL import Image as PILImage
    from ultralytics import YOLO

    model = YOLO(model_path)

    IMG_EXTS = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    image_paths, gt_labels = [], []
    for class_dir in sorted(pathlib.Path(eval_root).iterdir()):
        if not class_dir.is_dir():
            continue
        for ext in IMG_EXTS:
            for img in class_dir.glob(ext):
                image_paths.append(img)
                gt_labels.append(class_dir.name)

    log(f"  Eval: {len(image_paths)} images, {len(set(gt_labels))} classes")

    valid_paths, valid_labels, n_skip = [], [], 0
    for p, lbl in zip(image_paths, gt_labels):
        try:
            with PILImage.open(p) as img:
                img.load()
            valid_paths.append(p)
            valid_labels.append(lbl)
        except Exception:
            n_skip += 1
    if n_skip:
        log(f"  Skipped {n_skip} corrupt image(s)")

    pred_probs = _eval_predict(model, valid_paths, imgsz)
    probs_3 = np.stack(pred_probs)  # (N, n_model_classes), model-index order

    # Report eval is binary: target vs non-target. Hard prediction is the model's
    # top-1 over its native classes, relabelled (target stays; everything else ->
    # non-target). probs_arr aggregates the native probabilities into the 2 report
    # classes only for the ROC/AUC scores.
    name2idx = {c: i for i, c in enumerate(REPORT_CLASSES)}
    probs_arr = np.zeros((probs_3.shape[0], len(REPORT_CLASSES)), dtype=probs_3.dtype)
    for j in range(probs_3.shape[1]):
        probs_arr[:, name2idx[_to_report_class(model.names[j])]] += probs_3[:, j]

    classes = REPORT_CLASSES
    valid_labels = [_to_report_class(lbl) for lbl in valid_labels]
    pred_labels = [_to_report_class(model.names[int(np.argmax(probs_3[i]))])
                   for i in range(probs_3.shape[0])]

    errors = [
        (valid_paths[i], valid_labels[i], pred_labels[i], probs_arr[i])
        for i in range(len(valid_paths)) if valid_labels[i] != pred_labels[i]
    ]
    errors.sort(key=lambda x: -float(x[3][name2idx.get(x[2], 0)]))
    ev = _eval_metrics(np.array(valid_labels), np.array(pred_labels),
                       probs_arr, classes, name2idx, errors[:12])
    log(f"  Accuracy ({TARGET_CLASS} vs {NON_TARGET_CLASS}): {ev['accuracy']}%  errors: {ev['n_errors']}/{ev['total']}")
    return ev


def _delta_fmt(new_val, prev_val):
    if prev_val is None or new_val is None:
        return "—", "#888"
    delta = round(new_val - prev_val, 2)
    if delta > 0:
        return f"+{delta:.2f}", "#1a7f37"
    if delta < 0:
        return f"{delta:.2f}", "#cf222e"
    return "±0.00", "#888"


def _verdict_html(new_f2, prev_f2, released=True) -> str:
    """Headline go/no-go verdict on the primary metric (F2 of the target class),
    including whether this model was released.
    """
    if new_f2 is None:
        return ""
    rel = "Released." if released else "Release skipped — previous model kept."
    if prev_f2 is None:
        return (f'<div class="verdict {"better" if released else "neutral"}">'
                f'<b>Primary metric · F2 ({TARGET_CLASS}): {new_f2:.2f}%</b><br>'
                f'Baseline — no previous model. {rel}</div>')
    delta = round(new_f2 - prev_f2, 2)
    if delta > 0:
        cls, msg = "better", f"✅ NEW is better (+{delta:.2f} pp)"
    elif delta < 0:
        cls, msg = "worse", f"⚠️ NEW is worse ({delta:.2f} pp)"
    else:
        cls, msg = "neutral", "➖ No change"
    return (f'<div class="verdict {cls}">'
            f'<b>Primary metric · F2 ({TARGET_CLASS}): {prev_f2:.2f}% → {new_f2:.2f}%</b><br>'
            f'{msg} · {rel}</div>')


def generate_report_html(new_eval: dict, prev_eval, new_tag: str, prev_tag, released=True) -> str:
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    prev_label = prev_tag or "—"
    classes = new_eval["classes"]

    def metric_row(label, new_val, prev_val):
        d_str, d_col = _delta_fmt(new_val, prev_val)
        prev_str = f"{prev_val:.2f}" if prev_val is not None else "—"
        nv_str = f"{new_val:.2f}" if new_val is not None else "—"
        return (
            f"<tr><td>{label}</td><td>{prev_str}</td>"
            f"<td><b>{nv_str}</b></td>"
            f'<td style="color:{d_col};font-weight:600">{d_str}</td></tr>'
        )

    def gp(ev, cls, key):
        return (ev["per_class"].get(cls, {}).get(key) if ev else None)

    overall = metric_row(
        "Accuracy (%)", new_eval["accuracy"],
        prev_eval["accuracy"] if prev_eval else None,
    )
    verdict = _verdict_html(
        gp(new_eval, TARGET_CLASS, "f2"), gp(prev_eval, TARGET_CLASS, "f2"), released
    )

    cls_rows = ""
    for cls in classes:
        sup = new_eval["per_class"].get(cls, {}).get("support", "?")
        rows = ""
        for key, label in [("precision", "Precision (%)"), ("recall", "Recall (%)"),
                            ("f1", "F1 (%)"), ("f2", "F2 (%)"), ("auc", "AUC-ROC")]:
            nv = gp(new_eval, cls, key)
            pv = gp(prev_eval, cls, key)
            if nv is not None:
                rows += metric_row(label, nv, pv)
        cls_rows += f"""
    <h3>{cls} <span style="font-weight:400;font-size:.85rem;color:#666">(n={sup})</span></h3>
    <table>
      <thead><tr><th>Metric</th><th>Prev</th><th>New</th><th>Δ</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>"""

    new_cm_b64 = _plot_confusion_matrix(new_eval["cm"], classes, f"New — {new_tag}")
    prev_cm_b64 = (
        _plot_confusion_matrix(prev_eval["cm"], classes, f"Prev — {prev_label}")
        if prev_eval else None
    )
    new_roc_b64 = _plot_roc_curves(new_eval["fpr_tpr"], new_eval["auc_per"], classes, f"New — {new_tag}")
    prev_roc_b64 = (
        _plot_roc_curves(prev_eval["fpr_tpr"], prev_eval["auc_per"], classes, f"Prev — {prev_label}")
        if prev_eval else None
    )
    err_b64 = _plot_error_grid(new_eval["errors"][:12], new_eval["name2idx"])

    def chart_pair(new_b64, prev_b64, title):
        inner = (f'<img src="data:image/png;base64,{prev_b64}">' if prev_b64 else "") + \
                f'<img src="data:image/png;base64,{new_b64}">'
        return f'<h2>{title}</h2><div class="charts">{inner}</div>'

    err_section = (
        f'<h2>Error grid — new model ({new_eval["n_errors"]} errors / {new_eval["total"]})</h2>'
        f'<img src="data:image/png;base64,{err_b64}" style="max-width:100%">'
        if err_b64 else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>MantaWatch — {new_tag}</title>
  <style>
    body {{font-family:system-ui,sans-serif;max-width:900px;margin:40px auto;padding:0 24px;color:#1f2328}}
    h1 {{font-size:1.5rem;margin-bottom:4px}}
    h2 {{font-size:1.05rem;margin-top:36px;margin-bottom:6px;color:#444;border-bottom:1px solid #d0d7de;padding-bottom:4px}}
    h3 {{font-size:.95rem;margin-top:20px;margin-bottom:4px}}
    .meta {{color:#636e7b;font-size:.875rem;margin-bottom:24px}}
    .badge {{display:inline-block;padding:1px 8px;border-radius:12px;font-size:.8rem;font-weight:600}}
    .new  {{background:#dafbe1;color:#116329}}
    .prev {{background:#eaeef2;color:#57606a}}
    .verdict {{padding:12px 16px;border-radius:8px;margin:18px 0;font-size:1rem;border:1px solid}}
    .verdict.better {{background:#dafbe1;border-color:#aceebb;color:#116329}}
    .verdict.worse {{background:#ffebe9;border-color:#ff9e94;color:#a40e26}}
    .verdict.neutral {{background:#eaeef2;border-color:#d0d7de;color:#57606a}}
    table {{border-collapse:collapse;width:100%;margin-top:6px;font-size:.9rem}}
    th,td {{padding:7px 12px;text-align:left;border-bottom:1px solid #d0d7de}}
    th {{background:#f6f8fa;font-weight:600}}
    tr:last-child td {{border-bottom:none}}
    .charts {{display:flex;gap:16px;flex-wrap:wrap;margin-top:8px}}
    .charts img {{max-width:48%;border:1px solid #d0d7de;border-radius:6px}}
    footer {{margin-top:48px;font-size:.75rem;color:#aaa}}
  </style>
</head>
<body>
  <h1>MantaWatch — Model Evaluation</h1>
  <p class="meta">
    Generated: {now}<br>
    <span class="badge new">new</span>&nbsp;{new_tag}&ensp;
    <span class="badge prev">prev</span>&nbsp;{prev_label}
  </p>

  {verdict}

  <h2>Overall accuracy</h2>
  <table>
    <thead><tr><th>Metric</th><th>Prev</th><th>New</th><th>Δ</th></tr></thead>
    <tbody>{overall}</tbody>
  </table>

  {cls_rows}

  {chart_pair(new_cm_b64, prev_cm_b64, "Confusion matrix")}
  {chart_pair(new_roc_b64, prev_roc_b64, "ROC curves")}
  {err_section}

  <footer>YOLO11s-cls · Auto-generated by RunPod</footer>
</body>
</html>"""


def _ensure_gh_pages_branch(headers: dict, base: str):
    """Create gh-pages branch if it doesn't exist, branching off main.
    Requires Contents:write permission on the token.
    """
    import requests
    check = requests.get(f"{base}/git/refs/heads/gh-pages", headers=headers, timeout=10)
    if check.status_code == 200:
        log("  gh-pages branch already exists.")
        return
    log(f"  gh-pages not found (HTTP {check.status_code}) — creating from main...")
    main_ref = requests.get(f"{base}/git/refs/heads/main", headers=headers, timeout=10)
    if main_ref.status_code != 200:
        raise RuntimeError(
            f"Cannot read main branch ref: HTTP {main_ref.status_code} — "
            "check that RELEASE_GITHUB_TOKEN has Contents:write permission. "
            f"Response: {main_ref.text[:300]}"
        )
    main_sha = main_ref.json()["object"]["sha"]
    log(f"  Branching gh-pages from main ({main_sha[:7]})...")
    ref = requests.post(
        f"{base}/git/refs", headers=headers,
        json={"ref": "refs/heads/gh-pages", "sha": main_sha}, timeout=15,
    )
    if ref.status_code not in (200, 201):
        raise RuntimeError(
            f"Cannot create gh-pages ref: HTTP {ref.status_code} — "
            "check that RELEASE_GITHUB_TOKEN has Contents:write permission. "
            f"Response: {ref.text[:300]}"
        )
    log("  gh-pages branch created.")


def _put_gh_pages_file(headers: dict, base: str, filename: str, content: str, message: str):
    import requests
    existing = requests.get(
        f"{base}/contents/{filename}", headers=headers, params={"ref": "gh-pages"}, timeout=10,
    )
    sha = existing.json().get("sha") if existing.status_code == 200 else None
    payload = {
        "message": message,
        "content": base64.b64encode(content.encode()).decode(),
        "branch": "gh-pages",
    }
    if sha:
        payload["sha"] = sha
    r = requests.put(f"{base}/contents/{filename}", headers=headers, json=payload, timeout=60)
    r.raise_for_status()


def publish_to_gh_pages(html: str, token: str, repo: str, tag: str) -> str:
    headers = _gh_headers(token)
    base = f"https://api.github.com/repos/{repo}"
    _ensure_gh_pages_branch(headers, base)
    report_file = f"report-{tag}.html"
    _put_gh_pages_file(headers, base, report_file, html, f"Report: {tag}")
    index_html = (
        f'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        f'<title>MantaWatch Reports</title>'
        f'<style>body{{font-family:system-ui,sans-serif;max-width:640px;margin:40px auto;padding:0 20px}}</style>'
        f'</head><body><h1>MantaWatch — Training Reports</h1>'
        f'<p><a href="{report_file}">Latest: {tag}</a></p></body></html>'
    )
    _put_gh_pages_file(headers, base, "index.html", index_html, f"Update index: {tag}")
    owner, reponame = repo.split("/")
    return f"https://{owner.lower()}.github.io/{reponame}/{report_file}"


def _apply_class_config(dataset_root):
    """Derive the class config from the dataset's train/ folders: the target is the folder
    prefixed 'the_' (e.g. the_manta), everything else is non-target. Fails if there isn't
    exactly one 'the_' folder.
    """
    global CLASSES, TARGET_FOLDER, TARGET_CLASS, NON_TARGET_CLASS, REPORT_CLASSES
    train_dir = pathlib.Path(dataset_root, "train")
    CLASSES = sorted(d.name for d in train_dir.iterdir() if d.is_dir())
    targets = [c for c in CLASSES if c.lower().startswith("the_")]
    if len(targets) != 1:
        raise ValueError(
            f"Expected exactly one target folder prefixed 'the_' among {CLASSES}, found "
            f"{targets}. Name the target 'the_<animal>' and the other folders 'other_*'/'non_*'."
        )
    TARGET_FOLDER = targets[0]
    TARGET_CLASS = TARGET_FOLDER[4:]  # strip 'the_' for display
    NON_TARGET_CLASS = f"non_{TARGET_CLASS}"
    REPORT_CLASSES = [TARGET_CLASS, NON_TARGET_CLASS]
    log(f"  Classes {CLASSES} → target '{TARGET_CLASS}'")


def handler(job: dict):
    inp = job.get("input", {})

    dataset_url = inp.get("dataset_url")
    if not dataset_url:
        raise ValueError("dataset_url is required (set the DATASET_URL repo variable)")
    epochs = int(inp.get("epochs", 100))
    base_model = inp.get("base_model", "yolo11s-cls.pt")
    imgsz = int(inp.get("imgsz", 640))
    batch = int(inp.get("batch", 32))
    patience = int(inp.get("patience", 20))
    run_name = inp.get("run_name") or f"run-{datetime.datetime.utcnow():%Y%m%d-%H%M%S}"
    release_tag = inp.get("release_tag")
    github_token = inp.get("github_token")
    repo = inp.get("repo")
    drive_sa_b64 = inp.get("drive_sa_b64")  # base64 service-account JSON for Drive

    log(f"=== Job start: {run_name} ===")
    log_mem("job start")
    _start_mem_logger()

    try:
        shutil.rmtree(RUNS_DIR, ignore_errors=True)

        folder_id = _drive_folder_id(dataset_url)
        if not folder_id:
            raise ValueError("dataset_url must be a Google Drive folder link")
        if not drive_sa_b64:
            raise ValueError("drive_sa_b64 (base64 service-account JSON) is required")

        yield {"status": "downloading"}
        log("==> Downloading dataset...")
        drive_token = _drive_token(drive_sa_b64)
        log("  Listing Drive folder (service account)...")
        listing = _gdrive_list_folder(folder_id, drive_token)
        train_items, test_items = _partition_drive_listing(listing)
        log(f"  train/val: {len(train_items)} images · test: {len(test_items)} images")
        yield from _run_with_progress(
            lambda progress_cb: _download_drive_items(
                train_items, DATA_DIR, drive_token, progress_cb
            ),
            "downloading",
        )
        dataset_root = DATA_DIR

        n_corrupt = _remove_corrupt_images(dataset_root)
        if n_corrupt:
            log(f"  Removed {n_corrupt} corrupt image(s)")

        _apply_class_config(dataset_root)

        yield {"status": "resizing"}
        log("==> Resizing training images...")
        yield from _run_with_progress(
            lambda progress_cb: _resize_images_inplace(
                dataset_root, RESIZE_MAX_SIDE, progress_cb
            ),
            "resizing",
        )

        have_eval = bool(test_items)

        log_mem("after download")

        log("==> Training...")
        best_pt = yield from _train_with_progress(
            dataset_root, run_name, base_model, epochs, imgsz, batch, patience
        )
        log("==> Training complete.")
        _free_torch_memory()  # collect the finished training thread's leftovers
        log_mem("after training")

        size_mb = round(os.path.getsize(best_pt) / 1024 / 1024, 1)
        metrics_csv = last_metrics(run_name)
        release_url = None
        report_url = None
        released = False
        skip_reason = None

        # Test is re-fetched separately at eval, so free the whole training set now.
        log("==> Removing training data from disk...")
        shutil.rmtree(dataset_root, ignore_errors=True)

        if release_tag and github_token and repo:
            new_eval = prev_eval = prev_tag = None
            if have_eval:
                # Evaluate before releasing so the release can be gated on the
                # primary metric. Eval is best-effort: failure here must not abort.
                try:
                    yield {"status": "evaluating"}
                    log(f"==> Downloading test split ({len(test_items)} images)...")
                    drive_token = _drive_token(drive_sa_b64)  # fresh token for this phase
                    yield from _run_with_progress(
                        lambda progress_cb: _download_drive_items(
                            test_items, EVAL_DIR, drive_token, progress_cb
                        ),
                        "evaluating",
                    )
                    test_dir = os.path.join(EVAL_DIR, "test")
                    n_corrupt = _remove_corrupt_images(test_dir)
                    if n_corrupt:
                        log(f"  Removed {n_corrupt} corrupt test image(s)")

                    log_mem("eval start")
                    log("==> Evaluating new model...")
                    new_eval = run_full_eval(best_pt, test_dir, imgsz)
                    _free_torch_memory()

                    prev_path, prev_tag = get_previous_model(github_token, repo, release_tag)
                    if prev_path:
                        log("==> Evaluating previous model...")
                        prev_eval = run_full_eval(prev_path, test_dir, imgsz)
                        _free_torch_memory()
                except Exception as eval_exc:
                    import traceback
                    log(f"  WARNING: evaluation failed — {eval_exc}")
                    log(traceback.format_exc())
            else:
                log("  No test/ split — cannot evaluate; releasing without quality gate")

            # Quality gate: skip the release only when we can prove the new model is
            # worse than the previous one on the primary metric (F2 of the target class). No
            # previous model, or a missing metric, leaves nothing to fail against, so
            # the model is released.
            mc = TARGET_CLASS
            new_f2 = (new_eval or {}).get("per_class", {}).get(mc, {}).get("f2")
            prev_f2 = (prev_eval or {}).get("per_class", {}).get(mc, {}).get("f2")
            if new_f2 is not None and prev_f2 is not None and new_f2 < prev_f2:
                skip_reason = (f"F2({TARGET_CLASS}) {new_f2:.2f}% < prev {prev_f2:.2f}% "
                               f"({prev_tag or 'prev'})")

            if skip_reason:
                log(f"==> Release SKIPPED — new model worse: {skip_reason}")
            else:
                yield {"status": "releasing", "tag": release_tag}
                log(f"==> Creating GitHub release {release_tag}...")
                release_url = create_github_release(best_pt, github_token, repo, release_tag, imgsz)
                released = True
                log(f"  Release: {release_url}")

            if new_eval is not None:
                # Publish the report regardless of the gate, so the decision is visible.
                try:
                    log("==> Generating and publishing report...")
                    html = generate_report_html(new_eval, prev_eval, release_tag, prev_tag, released)
                    report_url = publish_to_gh_pages(html, github_token, repo, release_tag)
                    log(f"  Report: {report_url}")
                except Exception as report_exc:
                    import traceback
                    log(f"  WARNING: report publish failed — {report_exc}")
                    log(traceback.format_exc())
            shutil.rmtree(EVAL_DIR, ignore_errors=True)
        else:
            log("  Skipping release and report (no tag/token/repo)")

        yield {
            "status": "done",
            "run_name": run_name,
            "model_size_mb": size_mb,
            "metrics_last_line": metrics_csv,
            "release_url": release_url,
            "report_url": report_url,
            "released": released,
            "skip_reason": skip_reason,
        }

    except Exception as exc:
        import traceback
        log(f"ERROR: {exc}")
        log(traceback.format_exc())
        raise


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
