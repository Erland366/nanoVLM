uv init --bare --python 3.12
uv sync --python 3.12
source .venv/bin/activate

## If decide not to use available pyproject.toml
# uv add torch numpy torchvision pillow datasets huggingface-hub transformers wandb hf_transfer einops
## OR
# pip install torch numpy torchvision pillow datasets huggingface-hub transformers wandb hf_transfer einops

export WANDB_API_KEY=
export HF_TOKEN=
export HF_HOME=
