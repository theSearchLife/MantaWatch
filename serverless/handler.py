"""
RunPod Serverless handler — YOLO manta classification training + evaluation.

Input:
    dataset_url      : Google Drive or direct URL to training dataset archive
    eval_dataset_url : URL to test dataset archive (flat class folders, optional)
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
    { status, run_name, model_size_mb, metrics_last_line, release_url, report_url }
"""
import base64
import datetime
import os
import pathlib
import queue
import re
import shutil
import subprocess
import threading
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import runpod

DEFAULT_DATASET_URL = "https://drive.google.com/uc?id=1SfFNtGMKP0dkqEqLAQNJT76_vP7LVwkC"
PROGRESS_INTERVAL   = 5  # yield progress every N epochs

TRAIN_ARCHIVE = "/tmp/dataset_train.archive"
EVAL_ARCHIVE  = "/tmp/dataset_eval.archive"
TRAIN_DIR     = "/tmp/dataset_train"
EVAL_DIR      = "/tmp/dataset_eval"
PREV_MODEL    = "/tmp/model_prev.pt"
RUNS_DIR      = "/tmp/runs"


def log(msg: str):
    print(msg, flush=True)


# Archive helpers

def _download(url: str, dest: str):
    if os.path.exists(dest):
        os.remove(dest)
    if "drive.google.com" in url:
        m = re.search(r"(?:id=|/d/)([a-zA-Z0-9_-]{20,})", url)
        file_id = m.group(1) if m else url
        log(f"  Google Drive ID: {file_id}")
        subprocess.run(["gdown", file_id, "-O", dest], check=True)
    else:
        log(f"  Direct download: {url}")
        subprocess.run(["wget", "-q", url, "-O", dest], check=True)
    log(f"  Downloaded: {os.path.getsize(dest) / 1024 / 1024:.0f} MB")


def _extract(archive: str, dest: str):
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
    with open(archive, "rb") as f:
        magic = f.read(4).hex()
    if magic.startswith("504b"):
        log("  Detected ZIP")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif magic.startswith("1f8b"):
        log("  Detected tar.gz")
        subprocess.run(["tar", "xzf", archive, "-C", dest, "--no-same-owner"], check=True)
    else:
        raise RuntimeError(f"Unknown archive type (magic={magic})")


def _stream_extract(url: str, dest: str):
    """Stream-extract a tar.gz from a URL directly — no archive stored locally.
    Halves peak disk usage vs download-then-extract (critical for large eval datasets).
    For Google Drive, uses drive.usercontent.google.com to bypass the virus-scan page.
    """
    import requests
    import tarfile as tf_lib

    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)

    if "drive.google.com" in url or "drive.usercontent.google.com" in url:
        m = re.search(r"(?:id=|/d/)([a-zA-Z0-9_-]{20,})", url)
        if not m:
            raise ValueError(f"Cannot parse Google Drive file ID from: {url}")
        file_id = m.group(1)
        stream_url = (
            f"https://drive.usercontent.google.com/download"
            f"?id={file_id}&export=download&confirm=t"
        )
        log(f"  Google Drive ID: {file_id}")
    else:
        stream_url = url

    log(f"  Streaming → extraction (archive will not be stored)...")
    resp = requests.get(stream_url, stream=True, timeout=120)
    resp.raise_for_status()
    resp.raw.decode_content = True  # handle HTTP-level Content-Encoding

    with tf_lib.open(fileobj=resp.raw, mode="r|gz") as tar:
        tar.extractall(dest)
    log(f"  Streamed and extracted to: {dest}")


def find_train_root(extract_dir: str) -> str:
    result = subprocess.run(
        ["find", extract_dir, "-maxdepth", "3", "-type", "d", "-name", "train"],
        capture_output=True, text=True,
    )
    dirs = [p.strip() for p in result.stdout.strip().splitlines() if p.strip()]
    if not dirs:
        raise RuntimeError("No 'train' directory found in archive")
    root = str(pathlib.Path(dirs[0]).parent)
    log(f"  Dataset root: {root}")
    log(f"  Classes: {os.listdir(os.path.join(root, 'train'))}")
    return root


def find_eval_root(extract_dir: str) -> str:
    """Find the dir whose immediate subdirs are class folders containing images."""
    IMG_EXTS = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    for candidate in [pathlib.Path(extract_dir)] + sorted(pathlib.Path(extract_dir).iterdir()):
        if not candidate.is_dir():
            continue
        subdirs = [d for d in candidate.iterdir() if d.is_dir()]
        if subdirs and any(img for d in subdirs for ext in IMG_EXTS for img in d.glob(ext)):
            return str(candidate)
    return extract_dir


# Training

def _train_with_progress(dataset_root, run_name, base_model, epochs, imgsz, batch, patience):
    """Generator: YOLO trains in a background thread, yields epoch dicts every PROGRESS_INTERVAL epochs.
    Return value (via StopIteration) is the path to best.pt.
    Usage: best_pt = yield from _train_with_progress(...)
    """
    progress_q    = queue.Queue()
    error_holder  = [None]
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


# GitHub helpers

def _gh_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def create_github_release(model_path: str, token: str, repo: str, tag: str) -> str:
    import requests
    headers = _gh_headers(token)
    base    = f"https://api.github.com/repos/{repo}"
    existing = requests.get(f"{base}/releases/tags/{tag}", headers=headers, timeout=10)
    if existing.status_code == 200:
        release_id = existing.json()["id"]
        requests.delete(f"{base}/releases/{release_id}", headers=headers, timeout=10)
        requests.delete(f"{base}/git/refs/tags/{tag}",  headers=headers, timeout=10)
        log(f"  Deleted existing release {tag}")
    resp = requests.post(
        f"{base}/releases", headers=headers,
        json={
            "tag_name": tag, "name": f"Model {tag}",
            "body": "YOLO11s-cls · 3 classes: manta / other_fish / non_fish\n\nAuto-generated by RunPod training job.",
            "draft": False, "prerelease": False,
        }, timeout=30,
    )
    resp.raise_for_status()
    upload_url = resp.json()["upload_url"].replace("{?name,label}", "")
    with open(model_path, "rb") as f:
        up = requests.post(
            f"{upload_url}?name=model.pt",
            headers={**headers, "Content-Type": "application/octet-stream"},
            data=f, timeout=120,
        )
    up.raise_for_status()
    return up.json()["browser_download_url"]


def get_previous_model(token: str, repo: str, current_tag: str):
    """Download model.pt from the most recent release before current_tag.
    Returns (local_path, tag_name) or (None, None).
    """
    import requests
    headers = _gh_headers(token)
    resp = requests.get(
        f"https://api.github.com/repos/{repo}/releases",
        headers=headers, timeout=15,
    )
    if resp.status_code != 200:
        log(f"  Could not fetch releases ({resp.status_code})")
        return (None, None)
    for release in resp.json():
        if release["tag_name"] == current_tag:
            continue
        for asset in release.get("assets", []):
            if asset["name"] == "model.pt":
                log(f"  Previous release: {release['tag_name']} ({asset['size']//1024//1024} MB)")
                dl = requests.get(asset["browser_download_url"], timeout=120, stream=True)
                dl.raise_for_status()
                with open(PREV_MODEL, "wb") as f:
                    for chunk in dl.iter_content(1024 * 1024):
                        f.write(chunk)
                return (PREV_MODEL, release["tag_name"])
    log("  No previous release found")
    return (None, None)


# Evaluation

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
    ax.set_xticklabels(classes, rotation=25, ha="right")
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


def run_full_eval(model_path: str, eval_root: str) -> dict:
    """Run full evaluation on a flat class-folder dataset. Returns metrics + raw data for charts."""
    import numpy as np
    from PIL import Image as PILImage
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix as sk_cm, roc_auc_score, roc_curve,
    )
    from sklearn.preprocessing import label_binarize
    from ultralytics import YOLO

    model    = YOLO(model_path)
    classes  = list(model.names.values())
    name2idx = {v: k for k, v in model.names.items()}

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

    pred_labels, pred_probs = [], []
    for result in model.predict(
        source=[str(p) for p in valid_paths], imgsz=640, verbose=False, stream=True,
    ):
        probs   = result.probs.data.cpu().numpy()
        top_idx = int(np.argmax(probs))
        pred_labels.append(model.names[top_idx])
        pred_probs.append(probs)

    probs_arr = np.stack(pred_probs)
    y_true    = np.array(valid_labels)
    y_pred    = np.array(pred_labels)

    accuracy = round(float(accuracy_score(y_true, y_pred)) * 100, 2)
    prec_arr = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    rec_arr  = recall_score  (y_true, y_pred, labels=classes, average=None, zero_division=0)
    f1_arr   = f1_score      (y_true, y_pred, labels=classes, average=None, zero_division=0)

    y_bin   = label_binarize(y_true, classes=classes)
    y_score = np.stack([probs_arr[:, name2idx[c]] for c in classes], axis=1)

    auc_per, fpr_tpr = {}, {}
    for i, cls in enumerate(classes):
        try:
            auc_per[cls] = round(float(roc_auc_score(y_bin[:, i], y_score[:, i])), 4)
            fpr, tpr, _  = roc_curve(y_bin[:, i], y_score[:, i])
            fpr_tpr[cls] = (fpr.tolist(), tpr.tolist())
        except Exception:
            auc_per[cls] = None

    per_class = {
        cls: {
            "precision": round(float(prec_arr[i]) * 100, 2),
            "recall":    round(float(rec_arr[i])  * 100, 2),
            "f1":        round(float(f1_arr[i])   * 100, 2),
            "auc":       auc_per.get(cls),
            "support":   int((y_true == cls).sum()),
        }
        for i, cls in enumerate(classes)
    }

    cm = sk_cm(y_true, y_pred, labels=classes)
    errors = [
        (valid_paths[i], valid_labels[i], pred_labels[i], probs_arr[i])
        for i in range(len(valid_paths)) if valid_labels[i] != pred_labels[i]
    ]
    errors.sort(key=lambda x: -float(x[3][name2idx.get(x[2], 0)]))
    log(f"  Accuracy: {accuracy}%  errors: {len(errors)}/{len(valid_paths)}")

    return {
        "accuracy":  accuracy,
        "per_class": per_class,
        "classes":   classes,
        "total":     len(valid_paths),
        "n_errors":  len(errors),
        "cm":        cm,
        "fpr_tpr":   fpr_tpr,
        "auc_per":   auc_per,
        "errors":    errors,
        "name2idx":  name2idx,
    }


# Report generation

def _delta_fmt(new_val, prev_val):
    if prev_val is None or new_val is None:
        return "—", "#888"
    delta = round(new_val - prev_val, 2)
    if delta > 0:
        return f"+{delta:.2f}", "#1a7f37"
    if delta < 0:
        return f"{delta:.2f}", "#cf222e"
    return "±0.00", "#888"


def generate_report_html(new_eval: dict, prev_eval, new_tag: str, prev_tag) -> str:
    now        = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    prev_label = prev_tag or "—"
    classes    = new_eval["classes"]

    def metric_row(label, new_val, prev_val):
        d_str, d_col = _delta_fmt(new_val, prev_val)
        prev_str = f"{prev_val:.2f}" if prev_val is not None else "—"
        nv_str   = f"{new_val:.2f}" if new_val is not None else "—"
        return (
            f"<tr><td>{label}</td><td><b>{nv_str}</b></td>"
            f"<td>{prev_str}</td>"
            f'<td style="color:{d_col};font-weight:600">{d_str}</td></tr>'
        )

    def gp(ev, cls, key):
        return (ev["per_class"].get(cls, {}).get(key) if ev else None)

    overall = metric_row(
        "Accuracy (%)", new_eval["accuracy"],
        prev_eval["accuracy"] if prev_eval else None,
    )

    cls_rows = ""
    for cls in classes:
        sup = new_eval["per_class"].get(cls, {}).get("support", "?")
        rows = ""
        for key, label in [("precision", "Precision (%)"), ("recall", "Recall (%)"),
                            ("f1", "F1 (%)"), ("auc", "AUC-ROC")]:
            nv = gp(new_eval, cls, key)
            pv = gp(prev_eval, cls, key)
            if nv is not None:
                rows += metric_row(label, nv, pv)
        cls_rows += f"""
    <h3>{cls} <span style="font-weight:400;font-size:.85rem;color:#666">(n={sup})</span></h3>
    <table>
      <thead><tr><th>Metric</th><th>New</th><th>Prev</th><th>Δ</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>"""

    new_cm_b64  = _plot_confusion_matrix(new_eval["cm"],  classes, f"New — {new_tag}")
    prev_cm_b64 = (_plot_confusion_matrix(prev_eval["cm"], classes, f"Prev — {prev_label}")
                   if prev_eval else None)
    new_roc_b64  = _plot_roc_curves(new_eval["fpr_tpr"],  new_eval["auc_per"],  classes, f"New — {new_tag}")
    prev_roc_b64 = (_plot_roc_curves(prev_eval["fpr_tpr"], prev_eval["auc_per"], classes, f"Prev — {prev_label}")
                    if prev_eval else None)
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

  <h2>Overall accuracy</h2>
  <table>
    <thead><tr><th>Metric</th><th>New</th><th>Prev</th><th>Δ</th></tr></thead>
    <tbody>{overall}</tbody>
  </table>

  {cls_rows}

  {chart_pair(new_cm_b64, prev_cm_b64, "Confusion matrix")}
  {chart_pair(new_roc_b64, prev_roc_b64, "ROC curves")}
  {err_section}

  <footer>YOLO11s-cls · Auto-generated by RunPod</footer>
</body>
</html>"""


# GitHub Pages publishing

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
        "branch":  "gh-pages",
    }
    if sha:
        payload["sha"] = sha
    r = requests.put(f"{base}/contents/{filename}", headers=headers, json=payload, timeout=60)
    r.raise_for_status()


def publish_to_gh_pages(html: str, token: str, repo: str, tag: str) -> str:
    headers = _gh_headers(token)
    base    = f"https://api.github.com/repos/{repo}"
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


# RunPod handler (generator — enables /stream endpoint)

def handler(job: dict):
    inp = job.get("input", {})

    dataset_url      = inp.get("dataset_url",      DEFAULT_DATASET_URL)
    eval_dataset_url = inp.get("eval_dataset_url")
    epochs           = int(inp.get("epochs",   100))
    base_model       = inp.get("base_model",   "yolo11s-cls.pt")
    imgsz            = int(inp.get("imgsz",    640))
    batch            = int(inp.get("batch",    32))
    patience         = int(inp.get("patience", 20))
    run_name         = inp.get("run_name") or f"run-{datetime.datetime.utcnow():%Y%m%d-%H%M%S}"
    release_tag      = inp.get("release_tag")
    github_token     = inp.get("github_token")
    repo             = inp.get("repo")

    log(f"=== Job start: {run_name} ===")

    try:
        yield {"status": "downloading"}
        log("==> Downloading training dataset...")
        _download(dataset_url, TRAIN_ARCHIVE)

        yield {"status": "extracting"}
        log("==> Extracting...")
        _extract(TRAIN_ARCHIVE, TRAIN_DIR)
        dataset_root = find_train_root(TRAIN_DIR)

        log("==> Training...")
        best_pt = yield from _train_with_progress(
            dataset_root, run_name, base_model, epochs, imgsz, batch, patience
        )
        log("==> Training complete.")

        size_mb     = round(os.path.getsize(best_pt) / 1024 / 1024, 1)
        metrics_csv = last_metrics(run_name)
        release_url = None
        report_url  = None

        # Free disk space: training artifacts no longer needed after training
        log("==> Cleaning up training artifacts to free disk space...")
        for _p in [TRAIN_ARCHIVE, TRAIN_DIR]:
            try:
                if os.path.isdir(_p):
                    shutil.rmtree(_p)
                elif os.path.exists(_p):
                    os.remove(_p)
                log(f"  Removed {_p}")
            except Exception as _e:
                log(f"  Warning: could not remove {_p}: {_e}")

        if release_tag and github_token and repo:
            yield {"status": "releasing", "tag": release_tag}
            log(f"==> Creating GitHub release {release_tag}...")
            release_url = create_github_release(best_pt, github_token, repo, release_tag)
            log(f"  Release: {release_url}")

            if eval_dataset_url:
                yield {"status": "evaluating"}
                log("==> Downloading & extracting eval dataset (streaming)...")
                _stream_extract(eval_dataset_url, EVAL_DIR)
                eval_root = find_eval_root(EVAL_DIR)

                log("==> Evaluating new model...")
                new_eval = run_full_eval(best_pt, eval_root)

                prev_path, prev_tag = get_previous_model(github_token, repo, release_tag)
                prev_eval = None
                if prev_path:
                    log("==> Evaluating previous model...")
                    prev_eval = run_full_eval(prev_path, eval_root)

                log("==> Generating and publishing report...")
                try:
                    html       = generate_report_html(new_eval, prev_eval, release_tag, prev_tag)
                    report_url = publish_to_gh_pages(html, github_token, repo, release_tag)
                    log(f"  Report: {report_url}")
                except Exception as pub_exc:
                    import traceback
                    log(f"  WARNING: publish_to_gh_pages failed — {pub_exc}")
                    log(traceback.format_exc())
                    log("  Check that RELEASE_GITHUB_TOKEN has Contents:write permission.")
            else:
                log("  No eval_dataset_url — skipping evaluation report")
        else:
            log("  Skipping release and report (no tag/token/repo)")

        yield {
            "status":            "done",
            "run_name":          run_name,
            "model_size_mb":     size_mb,
            "metrics_last_line": metrics_csv,
            "release_url":       release_url,
            "report_url":        report_url,
        }

    except Exception as exc:
        import traceback
        log(f"ERROR: {exc}")
        log(traceback.format_exc())
        raise


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
