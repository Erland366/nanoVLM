# DualTowerVLM
## Setup
1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
2. Simply run `setup.sh`

## Train
1. Run `train_dual_tower.sh` after filling in the environment variables.

## Directory
1. Training code -> `train_*.py` (vanilla / dual_tower)
2. TwinTower model -> `models/dual_tower.py`
3. Configs -> `configs/config_*.py` (vanilla / dual_tower)
4. Evaluation -> `generate_dual_tower.py` & `cider.py`

## Changelog
> 15/10/25 -- Revamp and tidy up `nanoVLM` repo, trimming out unused codes
> 13/12/25 -- Revamp DualTowerVLM dataset packing, training, generation code

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
