# apt update -y && apt upgrade -y && apt install -y tmux


# export UV_CACHE_DIR = ~/users/patrick/
## Using `uv` package manager
# uv init --bare --python 3.12
# uv sync --python 3.12
# source .venv/bin/activate

## If decide not to use the available pyproject.toml
# uv add torch numpy torchvision pillow datasets huggingface-hub transformers wandb hf_transfer einops
## OR
pip install datasets einops hf_transfer huggingface_hub ipykernel ipywidgets jupyter numpy pillow pycocoevalcap torch torchvision transformers wandb lmms_eval
