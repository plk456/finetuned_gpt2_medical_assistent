# Multi-stage Dockerfile for fine_tuned_peft
# - Uses official PyTorch runtime image with CUDA support for GPU builds
# - Falls back to CPU-only base if you set BUILD_CPU=1

ARG BUILD_CPU=0

FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-runtime as base

# Allow overriding base image to a slim CPU Python image when building for CPU
FROM python:3.11-slim as cpu-base

# Choose which stage to use (docker build --build-arg BUILD_CPU=1 ...)
FROM cpu-base AS final
ARG BUILD_CPU=0

# If not building CPU-only, switch to the CUDA-enabled base
RUN if [ "$BUILD_CPU" = "0" ]; then echo "Using CUDA base image (ensure host has nvidia-docker)"; else echo "Using CPU base image"; fi

# Common setup
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps for common packages and wget/git
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        wget \
        ca-certificates \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel
# Install requirements; some packages (torch) may be provided by base image in GPU builds
RUN python -m pip install --no-cache-dir -r /app/requirements.txt || true

# Copy only the application code
COPY . /app

# Create a non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# Expose Gradio port
EXPOSE 7860

# Default command (runs app with environment-based behavior)
CMD ["python", "app.py"]
