# Dockerizing the STT API

This folder contains Dockerfiles to run the FastAPI app at
`src/services/python/apps/stt_api.py` using the v2 STT service.

- `Dockerfile.cpu`: CPU-only image (Python slim + ffmpeg)
- `Dockerfile.gpu`: GPU-enabled image (PyTorch CUDA runtime + ffmpeg)
- `requirements.txt`: minimal Python deps for the service

## Build

Build from the REPO ROOT as context (Dockerfiles live in a subfolder):

```
# CPU
docker build -f src/services/python/docker/Dockerfile.cpu -t stt-api:cpu .

# GPU
docker build -f src/services/python/docker/Dockerfile.gpu -t stt-api:gpu .
```

## Run

```
# CPU
docker run --rm -p 8000:8000 stt-api:cpu

# GPU (requires NVIDIA Container Toolkit)
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

docker run --rm -p 8000:8000 --gpus=all stt-api:gpu
```

API will be available at: `http://localhost:8000/transcribe`

## Why uv instead of pip?

- Faster installs and dependency resolution, great Docker layer caching
- Simple: `uv pip install --system -r requirements.txt`
- Deterministic if you later adopt a lockfile

## Notes

- The Dockerfiles copy the entire repo into `/app` and set `PYTHONPATH=/app` so imports like `src.services...` work.
- For non-WAV formats, ffmpeg is used by faster-whisper’s backend; WAV bytes can be decoded fully in memory.
- GPU image includes CUDA/cuDNN via the PyTorch runtime; make sure your host drivers are compatible.
