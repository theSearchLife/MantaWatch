# Architecture

Developer-facing overview of how the pipeline works.

## Components

| Component | Responsibility |
|---|---|
| `.github/workflows/train.yml` | Reads `training_config.yml`, gathers secrets/variables, submits a job to RunPod, and streams progress. |
| `.github/scripts/stream_poll.py` | Polls the RunPod `/stream` + `/status` endpoints and prints progress into the Actions log; exits non-zero on failure so the Action turns red. |
| `serverless/handler.py` | The worker. Runs end-to-end: download → train → evaluate → gate → release + report. |
| `serverless/Dockerfile` | Image for the worker (Ultralytics, ONNX export deps, google-auth, etc.). |
| `.github/workflows/build-image.yml` | Builds and pushes that image to GHCR. |
| `training_config.yml` | Hyperparameters (epochs, imgsz, batch, patience, base model). |

The animal-specific configuration is **not** in the code: classes are derived from the
dataset's folder names, the dataset from `DATASET_URL`, credentials from secrets.

## Flow of one run

```
train.yml (GitHub Actions)
  └─ POST /run to RunPod with { dataset_url, drive_sa_b64, epochs, … }
       │
       ▼
handler.py (RunPod Serverless GPU worker, a streaming generator)
  1. derive classes from the train/ folders (the `the_` one = target) → TARGET_CLASS / NON_TARGET_CLASS
  2. mint a Drive token from the service-account JSON
  3. list the Drive folder, partition into train/val and test items
  4. download train/val to disk (parallel, authenticated)
  5. drop corrupt images; require exactly one `the_` target folder (fail fast otherwise)
  6. downscale train images once (long side ≤ 1024)
  7. train YOLO11 classifier (Ultralytics), streaming epoch progress
  8. free training data from disk
  9. download the test split; evaluate the NEW model
 10. fetch the previous released model; evaluate it on the same test set
 11. quality gate: compare F2(target); decide release vs skip
 12. if not worse → create GitHub Release (model.pt + model.onnx)
 13. publish the comparison report to gh-pages (always)
 14. yield the final result dict
```

`stream_poll.py` turns the yielded dicts into log lines and decides the Action's
exit code from the terminal `/status`.

## Key design decisions

**Dataset over Google Drive, via a service account.** Anonymous / API-key access hits
Drive's per-file download quota at scale. A service account is authenticated
(per-principal quotas), so bulk per-file downloads work. The trade-off is the setup
(sharing the folder with the SA email) — see [setup.md](setup.md). Researchers keep a
normal Drive workflow (folders, drag-and-drop); originals are never modified.

**Train images downscaled once on the worker.** Source photos are multi-megapixel;
decoding them every epoch pins a single CPU core and starves the GPU. The worker
resizes them once (long side ≤ 1024) before training. The model trains at `imgsz`
(640) regardless, so results are unchanged — only decode cost drops. Drive originals
are untouched.

**Evaluation preprocessing is parallelised.** Ultralytics' `predict` preprocesses a
source serially on one core, which dominates eval on full-resolution photos. Eval
instead runs images through a torch `DataLoader` (`num_workers`) using Ultralytics'
**own** classification transform, then does batched GPU inference — identical numbers,
parallel preprocessing (eval dropped from tens of minutes to seconds). See
`_eval_predict` in `handler.py`.

**The report is a binary target-vs-rest comparison.** The model trains on N classes,
but the report collapses them to **target vs non-target** (all non-target classes
merged) by the model's top-1 prediction. The primary metric is **F2 of the target
class** (recall-weighted: missing the target costs more than a false alarm). The
report shows the new vs previous model with a single go/no-go verdict.

**Releases are gated.** The worker evaluates the new model *and* the previous released
model on the same test set, and **skips the release if the new model's F2(target) is
worse**. If there's no previous model, or eval can't run, it releases (nothing to fail
against). The report is published either way so a rejection is visible with its reason.

**Two model formats per release.** `model.pt` (Ultralytics/PyTorch) and `model.onnx`
(exported via `model.export(format="onnx")`, with class names + imgsz embedded in the
ONNX metadata). The ONNX is what the local Rust sorter consumes.

**Report hosting.** The report is HTML with inline (base64) charts, written to a
`gh-pages` branch the worker creates if missing, then served by GitHub Pages.

## Memory & cost notes

- The worker is memory-bound by the container's **cgroup** limit (not host RAM); the
  code reads `/sys/fs/cgroup/memory.*` and logs usage periodically.
- Training leftovers are freed (`del model` + CUDA cache) and the training set is
  deleted from disk before eval; the test set is fetched separately at eval time.
- Downloads use `fsync` + `posix_fadvise(DONTNEED)` so the page cache from writes
  doesn't count against the cgroup limit.
- Runs on a RunPod Serverless 4090 with min-0 workers (no idle cost) and a high
  execution timeout.

## Extending / gotchas

- **Changing `handler.py` or the `Dockerfile` requires rebuilding the worker image**
  (Build & Push Docker Image), or the endpoint runs stale code.
- Classes are read from the dataset's `train/` subfolders. Exactly one must be prefixed
  `the_` (the target); the worker fails the job otherwise.
- The target class is the `the_`-prefixed folder (e.g. `the_manta` → target `manta`); the
  rest are merged into non-target for the report.
- `RELEASE_GITHUB_TOKEN` needs **Contents: write** for both Releases and `gh-pages`.
- Report not visible ⇒ GitHub Pages not enabled on `gh-pages` (one-time setting).
