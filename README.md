# DualTowerVLM
## Setup
1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
2. Run `setup.sh`

## Train
1. Run `train.sh` after filling in the environment variables.

## Directory
0. End-to-end Implementation (Data + Model) -> `e2e_dual_tower.ipynb`
1. Dataset & Collators -> `data/datasets.py` & `data/collators.py`
2. DualTower model -> `models/dual_tower/dual_tower.py` & `models/dual_tower/dual_language_model.py` (RoPE & Causal Mask Modification)
3. Configs -> `configs/config_*.py` (`vanilla` / `dual_tower`)
4. Training Loop -> `train_*.py` (`vanilla` / `dual_tower`)
5. Evaluation -> `evaluation/cider_utils.py`

## Changelog
- > 18/12/25 -- Add `lm_changes.patch` to make it easier to look at the diff between `dual_language_model.py` vs original `language_model.py`
- > 18/12/25 -- Add `cider_utils.py` for in-training evaluation
- > 13/12/25 -- Revamp DualTowerVLM dataset packing, training, generation code
- > 15/10/25 -- Revamp and tidy up `nanoVLM` repo, trimming out unused codes

## Citation
This implementation is adapted from:
```
@misc{wiedmann2025nanovlm,
  author = {Luis Wiedmann and Aritra Roy Gosthipaty and Andrés Marafioti},
  title = {nanoVLM},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/nanoVLM}}
}
```
