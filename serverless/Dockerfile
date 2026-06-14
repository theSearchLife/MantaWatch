FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN pip install --no-cache-dir \
    ultralytics gdown runpod requests scikit-learn \
    huggingface_hub

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
