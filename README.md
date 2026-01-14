# DualTowerVLM

DualTowerVLM is a compact dual-tower vision-language model. The left tower (vision encoder + modality projector + language decoder) processes image tokens and produces KV cache. The right tower is a language model that consumes the full text sequence while reusing the cached image KV.

## Quick Start

Setup:
```bash
uv init --bare --python 3.12
uv sync --python 3.12
source .venv/bin/activate
uv add torch numpy torchvision pillow datasets huggingface-hub transformers wandb
```

Train:
```bash
bash train.sh
```

Generate:
```bash
python generate.py --checkpoint path/or/hf-repo --image path/to/image.png --prompt "Describe the image."
```

Evaluate (MMStar):
```bash
bash evaluate.sh
```

## Environment Setup

We use `uv` by default, but any environment manager works. If you prefer pip:
```bash
pip install torch numpy torchvision pillow datasets huggingface-hub transformers wandb
```

Dependencies:
- `torch` <3
- `numpy` <3
- `torchvision` for image preprocessing
- `pillow` for image loading
- `datasets` for datasets
- `huggingface-hub` & `transformers` for pretrained backbones
- `wandb` for logging

## Training

Defaults live in `configs/config_dual_tower_cauldron.py` and are loaded by `train_dual_tower.py`.

```bash
bash train.sh
```

Notes:
- `train.sh` exports environment variables (W&B + HF), sets `CUDA_VISIBLE_DEVICES`, and logs to `logs/train_dual_tower_*.log`.
- Edit `configs/config_dual_tower_cauldron.py` for datasets, packing, LR schedules, and eval cadence.

## Generate

```bash
python generate.py \
  --checkpoint path/or/hf-repo \
  --image path/to/image.png \
  --prompt "Describe the image."
```

Sampling:
```bash
python generate.py \
  --checkpoint path/or/hf-repo \
  --image path/to/image.png \
  --prompt "Describe the image." \
  --sampling --top_k 50 --top_p 0.9 --temperature 0.7
```

If your checkpoint is a weights file, pass `--config path/to/config.json`.

## Evaluation with lmms-eval

Install lmms-eval from source:
```bash
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

Default MMStar eval:
```bash
bash evaluate.sh
```

Custom checkpoint + task:
```bash
CHECKPOINT=path/or/hf-repo TASKS=mmstar BATCH_SIZE=32 bash evaluate.sh
```

Direct call:
```bash
python evaluation.py --model path/or/hf-repo --tasks mmstar --batch_size 32 --process_with_media
```

## Hub Integration

```python
from models.dual_tower.dual_tower import DualTowerVLM

model = DualTowerVLM.from_pretrained("path/or/hf-repo", load_backbone=False)
model.save_pretrained("local/output")
```

## Directory
0. End-to-end notebook (data + model) -> `e2e_dual_tower.ipynb`
1. Dataset & collators -> `data/datasets.py`, `data/collators.py`
2. DualTower model -> `models/dual_tower/dual_tower.py`, `models/dual_tower/dual_language_model.py`
3. Configs -> `configs/config_*.py` (`vanilla` / `dual_tower`)
4. Training -> `train_dual_tower.py`, `train_vanilla.py`
5. Evaluation -> `evaluation.py`, `evaluate.sh`, `eval/lmms_eval_wrapper.py`
6. Task-specific eval helpers -> `evaluation/*.py` (e.g. `ocr_vqa`, `coco_captions`)

## Changelog
- > 8/1/26 -- Fix decoding for dual tower evaluation (cider + accuracy)
- > 18/12/25 -- Add `lm_changes.patch` to make it easier to look at the diff between `dual_language_model.py` vs original `language_model.py`
- > 18/12/25 -- Add `cider_utils.py` for in-training evaluation
- > 13/12/25 -- Revamp DualTowerVLM dataset packing, training, generation code
- > 15/10/25 -- Revamp and tidy up `nanoVLM` repo, trimming out unused codes

## Citation
This implementation is adapted from:
```bibtex
@misc{wiedmann2025nanovlm,
  author = {Luis Wiedmann and Aritra Roy Gosthipaty and Andrés Marafioti},
  title = {nanoVLM},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/nanoVLM}}
}
```
