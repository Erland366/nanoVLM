"""
Debug/fast iteration config for torch.compile optimization testing.

Usage:
    # In train.py, change import:
    # from configs.config_debug import VLMConfig, TrainConfig, GlobalConfig

    # Or run with env var to select config (if you modify train.py)
    # CONFIG=debug python train.py

Model sizes:
    - Original SmolLM2-135M: 32 blocks, 960 hidden, ~135M params
    - Debug: 4 blocks, 256 hidden, ~5M params (27x smaller)
    - ViT: 4 blocks instead of 12

Expected speedup: ~10-20x faster per step
"""

from dataclasses import dataclass, field


@dataclass
class GlobalConfig:
    seed: int = 42
    hf_home: str = None
    log_dir: str = "logs"
    model_dir: str = "checkpoints"
    eval_dir: str = "evals"
    prefix_run_name: str = "debug"


@dataclass
class VLMConfig:
    # Vision Transformer - SMALLER (keep img_size=512 for data pipeline compatibility)
    vit_hidden_dim: int = 256  # was 768
    vit_inter_dim: int = 4 * 256  # was 4*768
    vit_patch_size: int = 16
    vit_img_size: int = 512  # Keep 512 for data pipeline compatibility
    vit_n_heads: int = 4  # was 12
    vit_dropout: float = 0.0
    vit_n_blocks: int = 4  # was 12
    vit_ln_eps: float = 1e-6
    vit_cls_flag: bool = False
    vit_model_type: str = "debug/vit-tiny"  # Placeholder name, random init

    # Language Model - SMALLER
    lm_hidden_dim: int = 256  # was 960
    lm_inter_dim: int = 512  # was 2560
    lm_rms_eps: float = 1e-5
    lm_re_base: int = 100000
    lm_max_position_embeddings: int = 2048  # was 8192
    lm_base_vocab_size: int = 49152
    extra_token_amount: int = 66
    lm_vocab_size: int = lm_base_vocab_size + extra_token_amount
    lm_n_heads: int = 4  # was 15
    lm_n_kv_heads: int = 2  # was 5
    lm_dropout: float = 0.0
    lm_n_blocks: int = 4  # was 32
    lm_attn_scaling: float = 1.0
    lm_max_length: int = 512  # was 4096
    lm_use_tokens: bool = False
    lm_tie_weights: bool = True
    lm_model_type: str = "debug/lm-tiny"  # Placeholder name, random init
    lm_tokenizer: str = 'HuggingFaceTB/SmolLM2-135M-Instruct'  # Still need tokenizer
    lm_chat_template: str = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    # Modality Projection
    # mp_image_token_length = (vit_img_size/vit_patch_size)^2 / mp_pixel_shuffle_factor^2
    # = (512/16)^2 / 4^2 = 32^2 / 16 = 64
    mp_pixel_shuffle_factor: int = 4
    mp_image_token_length: int = 64  # Must match: (img_size/patch_size)^2 / shuffle^2

    max_img_size: int = 512  # was 2048
    resize_to_max_side_len: bool = False

    vlm_extra_tokens: dict[str, str] = field(default_factory=lambda: {
        "image_token": "<|image|>",
        "global_image_token": "<|global_image|>",
        "r1c1": "<row_1_col_1>", "r1c2": "<row_1_col_2>", "r1c3": "<row_1_col_3>", "r1c4": "<row_1_col_4>",
        "r2c1": "<row_2_col_1>", "r2c2": "<row_2_col_2>", "r2c3": "<row_2_col_3>", "r2c4": "<row_2_col_4>",
        "r3c1": "<row_3_col_1>", "r3c2": "<row_3_col_2>", "r3c3": "<row_3_col_3>", "r3c4": "<row_3_col_4>",
        "r4c1": "<row_4_col_1>", "r4c2": "<row_4_col_2>", "r4c3": "<row_4_col_3>", "r4c4": "<row_4_col_4>",
    })
    vlm_load_backbone_weights: bool = False  # Random init for debug
    vlm_checkpoint_path: str = 'checkpoints/debug'
    hf_repo_name: str = None  # Don't push


@dataclass
class TrainConfig:
    # Model/Optimizer
    lr_mp: float = 1e-3
    lr_vision_backbone: float = 1e-4
    lr_language_backbone: float = 1e-4
    compile: bool = True  # Test torch.compile
    resume_from_vlm_checkpoint: bool = False

    # Data - minimal for debug
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    train_dataset_path: str = 'HuggingFaceM4/the_cauldron'
    train_dataset_name: tuple[str, ...] = ("textcaps",)  # Just one small dataset

    stream_dataset: bool = True
    interleave_datasets: bool = False
    interleave_probabilities: tuple[float, ...] | None = None
    interleave_stopping_strategy: str = "all_exhausted"
    streaming_shuffle_buffer: int = 100  # Small buffer
    stratified_val_split: bool = False
    data_cutoff_idx: int | None = 100  # Only 100 samples for debug

    # Quality filtering
    relevance_min_rating: int = 1
    image_correspondence_min_rating: int = 1
    visual_dependency_min_rating: int = 1
    formatting_min_rating: int = 1

    # Packing
    max_images_per_example: int = 2
    max_images_per_knapsack: int = 4
    pack_sequences: bool = False
    max_sample_length: int = 512

    # Training - short for debug
    max_training_steps: int = 20  # Very short
    max_grad_norm: float = 1.0

    # Evaluation
    val_size: int = 50
    max_val_batches: int = 5
    eval_in_epochs: bool = False  # Disable eval for speed
    eval_interval: int = 1000  # Won't trigger with 20 steps

    # Logging - minimal
    log_wandb: bool = False  # Disable for debug
    wandb_entity: str = ""
    wandb_project: str = 'debug'
    prefix_run_name: str = "debug"
    save_code_cfg: bool = False
    save_model_every_n_steps: int = 10000  # Won't trigger
    stats_log_interval: int = 5

    # Model saving - disabled
    save_local: bool = False
    local_model_cp_path: str = "checkpoints/debug"
    save_hf: bool = False
    hf_model_cp_path: str = ""

    # Extended eval - disabled
    use_lmms_eval: bool = False
    lmms_eval_tasks: str = ''
    lmms_eval_limit: float = None
    lmms_eval_batch_size: int = 1
