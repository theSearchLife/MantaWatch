# MantaWatch

Automated image-classification training pipeline. Trigger one GitHub Action and a GPU
worker (RunPod Serverless) trains a [YOLO11](https://docs.ultralytics.com/tasks/classify/)
classifier on images from a Google Drive folder, evaluates it against the previous model,
publishes a visual report, and — only if the new model is not worse — releases it.

It's generic: the classes come from your dataset's folder names, so the same pipeline
retrains on any animal without code changes (currently configured for **manta rays**).

## Companion tool — RunML

MantaWatch is the **training half** of a two-part system. The models it releases are used
with **[RunML](https://github.com/theSearchLife/RunML)** — a desktop tool that runs a
released model locally to sort a folder of images into per-class folders. You train a model
here, then use its release (`model.onnx`) with RunML to classify new images.

## Documentation

- **[Setup](docs/setup.md)** — one-time configuration: secrets, variables, the RunPod endpoint, GitHub Pages.
- **[Usage](docs/usage.md)** — adding images, running training, reading the report, getting the model.
- **[Architecture](docs/architecture.md)** — how the pipeline works under the hood.
