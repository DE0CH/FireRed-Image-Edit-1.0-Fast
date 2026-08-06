"""Modal deployment for FireRed-Image-Edit-1.0-Fast.

Runs the existing Gradio app (app.py) unchanged on a Modal GPU container,
with Hugging Face model weights cached in a persistent Modal Volume so
cold starts only load from cache instead of re-downloading ~60 GB.

Usage:
    modal run modal_app.py::download_models   # one-time: warm the weights cache
    modal deploy modal_app.py                 # deploy the web UI
"""

import os
import subprocess

import modal

APP_NAME = "firered-image-edit-fast"
PORT = 7860
CACHE_DIR = "/cache"

hf_cache = modal.Volume.from_name("firered-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.11.0",
        "torchvision==0.26.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "transformers==5.14.1",
        "accelerate==1.14.0",
        "diffusers==0.39.0",
        "peft==0.19.1",
        "gradio[mcp]==6.22.0",
        "av==17.1.0",
        "spaces==0.51.1",
        "huggingface-hub==1.24.0",
        "kernels==0.16.0",
    )
    .env({"HF_HOME": CACHE_DIR})
    .add_local_dir(
        os.path.dirname(os.path.abspath(__file__)),
        remote_path="/app",
        ignore=[".git", "uv.lock", "__pycache__", "*.pyc"],
    )
)

app = modal.App(APP_NAME)


@app.function(image=image, volumes={CACHE_DIR: hf_cache}, timeout=3600)
def download_models():
    from huggingface_hub import snapshot_download

    # The base pipeline's own transformer folder is never used (app.py swaps in
    # the Rapid-AIO transformer), so skip its ~40 GB of weights.
    snapshot_download(
        "FireRedTeam/FireRed-Image-Edit-1.1",
        ignore_patterns=["transformer/*"],
    )
    snapshot_download("prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19")
    hf_cache.commit()


@app.function(
    image=image,
    gpu="H100",
    volumes={CACHE_DIR: hf_cache},
    scaledown_window=300,
    max_containers=1,
    timeout=3600,
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=PORT, startup_timeout=1800)
def ui():
    subprocess.Popen(
        ["python", "app.py"],
        cwd="/app",
        env={
            **os.environ,
            "GRADIO_SERVER_NAME": "0.0.0.0",
            "GRADIO_SERVER_PORT": str(PORT),
        },
    )
