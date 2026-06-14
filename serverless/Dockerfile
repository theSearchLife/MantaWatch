FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN pip install --no-cache-dir \
    ultralytics gdown runpod requests scikit-learn \
    "huggingface_hub[hf_transfer]"

# Faster parallel downloads from the HF Hub
ENV HF_HUB_ENABLE_HF_TRANSFER=1

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
