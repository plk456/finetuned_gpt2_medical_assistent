# fine_tuned_peft

## Docker

This repository includes a Dockerfile that supports both CPU and GPU runs.

### Build

- CPU build (faster, no CUDA):

  docker build --build-arg BUILD_CPU=1 -t finetuned_peft:cpu .

- GPU build (requires NVIDIA GPU and nvidia-docker / Docker with GPU support):

  docker build -t finetuned_peft:gpu .

Note: the Dockerfile uses an official PyTorch CUDA runtime image when BUILD_CPU is 0.

### Run

- Run (mounting .env and model dirs if needed):

  docker run --rm -p 7860:7860 \
    -e MONGO_URI="${MONGO_URI}" \
    -v "$PWD":/app \
    finetuned_peft:gpu

For GPU access, run with the Docker runtime that exposes GPUs, for example:

  docker run --gpus all --rm -p 7860:7860 -e MONGO_URI="${MONGO_URI}" -v "$PWD":/app finetuned_peft:gpu

### Tips

- Keep secrets out of the image; pass via `-e` or use a secrets mechanism.
- Use Git LFS for model binaries or download model weights at runtime from a model repo.
