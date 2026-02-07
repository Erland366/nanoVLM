from dataclasses import dataclass, field


@dataclass
class VLMConfig:
    vit_hidden_dim: int = 768
    vit_inter_dim: int = 4 * vit_hidden_dim
    vit_patch_size: int = 16
    vit_img_size: int = 512
    vit_n_heads: int = 12
    vit_dropout: float = 0.0
    vit_n_blocks: int = 12
    vit_ln_eps: float = 1e-6
    vit_cls_flag: bool = False
    vit_model_type: str = "google/siglip2-base-patch16-512"

    lm_hidden_dim: int = 960
    lm_inter_dim: int = 2560
    lm_rms_eps: float = 1e-5
    lm_re_base: int = 100000
    lm_max_position_embeddings: int = 4096
    lm_base_vocab_size: int = 49152
    extra_token_amount: int = 66  # Number of extra tokens for the VLM (image start, image end, image token)
    lm_vocab_size: int = lm_base_vocab_size + extra_token_amount
    lm_n_heads: int = 15
    lm_n_kv_heads: int = 5
    lm_dropout: float = 0.0
    lm_n_blocks: int = 32
    lm_attn_scaling: float = 1.0
    lm_max_length: int = 4096
    lm_use_tokens: bool = False  # Use tokens or embeddings as LM input.
    lm_tie_weights: bool = True  # Tie LM head weight to token embeddings.
    lm_model_type: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    lm_tokenizer: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    lm_chat_template: str = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    mp_pixel_shuffle_factor: int = 4
    mp_image_token_length: int = 64

    # Mixture of Modality Heads (MoMH) config
    momh_enabled: bool = False
    momh_head_pct_vision: float = 0.2  # 20% of heads for V->V only
    momh_head_pct_text: float = 0.3  # 30% of heads for T->T only
    # Remaining 50% (1 - vision - text) for VT->VT cross-modal

    # Activation checkpointing for LM/ViT blocks during training.
    activation_checkpointing: bool = True

    max_img_size: int = 2048
    resize_to_max_side_len: bool = False

    vlm_extra_tokens: dict[str, str] = field(
        default_factory=lambda: {
            "image_token": "<|image|>",
            "global_image_token": "<|global_image|>",
            "r1c1": "<row_1_col_1>",
            "r1c2": "<row_1_col_2>",
            "r1c3": "<row_1_col_3>",
            "r1c4": "<row_1_col_4>",
            "r1c5": "<row_1_col_5>",
            "r1c6": "<row_1_col_6>",
            "r1c7": "<row_1_col_7>",
            "r1c8": "<row_1_col_8>",
            "r2c1": "<row_2_col_1>",
            "r2c2": "<row_2_col_2>",
            "r2c3": "<row_2_col_3>",
            "r2c4": "<row_2_col_4>",
            "r2c5": "<row_2_col_5>",
            "r2c6": "<row_2_col_6>",
            "r2c7": "<row_2_col_7>",
            "r2c8": "<row_2_col_8>",
            "r3c1": "<row_3_col_1>",
            "r3c2": "<row_3_col_2>",
            "r3c3": "<row_3_col_3>",
            "r3c4": "<row_3_col_4>",
            "r3c5": "<row_3_col_5>",
            "r3c6": "<row_3_col_6>",
            "r3c7": "<row_3_col_7>",
            "r3c8": "<row_3_col_8>",
            "r4c1": "<row_4_col_1>",
            "r4c2": "<row_4_col_2>",
            "r4c3": "<row_4_col_3>",
            "r4c4": "<row_4_col_4>",
            "r4c5": "<row_4_col_5>",
            "r4c6": "<row_4_col_6>",
            "r4c7": "<row_4_col_7>",
            "r4c8": "<row_4_col_8>",
            "r5c1": "<row_5_col_1>",
            "r5c2": "<row_5_col_2>",
            "r5c3": "<row_5_col_3>",
            "r5c4": "<row_5_col_4>",
            "r5c5": "<row_5_col_5>",
            "r5c6": "<row_5_col_6>",
            "r5c7": "<row_5_col_7>",
            "r5c8": "<row_5_col_8>",
            "r6c1": "<row_6_col_1>",
            "r6c2": "<row_6_col_2>",
            "r6c3": "<row_6_col_3>",
            "r6c4": "<row_6_col_4>",
            "r6c5": "<row_6_col_5>",
            "r6c6": "<row_6_col_6>",
            "r6c7": "<row_6_col_7>",
            "r6c8": "<row_6_col_8>",
            "r7c1": "<row_7_col_1>",
            "r7c2": "<row_7_col_2>",
            "r7c3": "<row_7_col_3>",
            "r7c4": "<row_7_col_4>",
            "r7c5": "<row_7_col_5>",
            "r7c6": "<row_7_col_6>",
            "r7c7": "<row_7_col_7>",
            "r7c8": "<row_7_col_8>",
            "r8c1": "<row_8_col_1>",
            "r8c2": "<row_8_col_2>",
            "r8c3": "<row_8_col_3>",
            "r8c4": "<row_8_col_4>",
            "r8c5": "<row_8_col_5>",
            "r8c6": "<row_8_col_6>",
            "r8c7": "<row_8_col_7>",
            "r8c8": "<row_8_col_8>",
        }
    )
    vlm_load_backbone_weights: bool = False
    vlm_checkpoint_path: str = "checkpoints"
    hf_repo_name: str = "nanoVLM"


@dataclass
class TrainConfig:
    # Optimizer selection
    optimizer: str = "adamw"  # "adamw" or "muon"
    weight_decay: float = 0.01
    adamw_betas: tuple[float, float] = (0.9, 0.999)

    # Muon optimizer (microsoft/dion)
    muon_mu: float = 0.95
    muon_adjust_lr: str | None = "spectral_norm"  # "spectral_norm", "rms_norm", or None
    muon_nesterov: bool = False
    muon_cautious_wd: bool = False
    muon_epsilon: float = 1e-8
    muon_use_triton: bool = False
    muon_scalar_algorithm: str = "adamw"  # "adamw" or "lion"

    lr_mp: float = 5e-5
    lr_vision_backbone: float = 1e-5
    lr_language_backbone: float = 1e-5

    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0

    eval_in_epochs: bool = False
    eval_interval: int = 500
    stats_log_interval: int = 10

    max_training_steps: int = 30000
    max_training_tokens: int | None = None

    compile: bool = False
    compile_mode: str | None = "default"
    activation_memory_budget: float | None = 0.5

    # Training-time activation checkpointing behavior (controlled via VLMConfig.activation_checkpointing).
    # This flag enables cross-rank sync when computing token-efficiency stats.
    sync_token_efficiency: bool = False

    max_images_per_example: int = 1
    max_images_per_knapsack: int = 18
    max_sample_length: int = 4096
    pack_sequences: bool = True

    train_dataset_path: str = "patrickamadeus/the_cauldron"
    train_dataset_name: tuple[str, ...] = ("sample_1pct",)
    stream_dataset: bool = False
    data_num_workers: int = 4
    val_num_workers: int = 4

    interleave_datasets: bool = False
    interleave_probabilities: tuple[float, ...] | None = None
    interleave_stopping_strategy: str = "all_exhausted"
    streaming_shuffle_buffer: int = 0
    stratified_val_split: bool = False
    data_cutoff_idx: int | None = None

    relevance_min_rating: int = 1
    image_correspondence_min_rating: int = 1
    visual_dependency_min_rating: int = 1
    formatting_min_rating: int = 1

    enable_validation: bool = False
    max_val_batches: int = 5000

    log_wandb: bool = True
    wandb_entity: str | None = None
    wandb_project: str = "momH"
    wandb_xaxis_tokens: bool = False
    prefix_run_name: str | None = None

    save_code_cfg: bool = True
    save_model_every_n_steps: int = 500
    save_local: bool = False
    local_model_cp_path: str = "checkpoints/vanilla-cauldron"
    save_hf: bool = False
    hf_model_cp_path: str = "patrickamadeus/vanilla-cauldron"
    checkpoint_every_n_steps: int = 500
    checkpoint_dir: str = "checkpoints"
    checkpoint_format: str = "dcp"  # "dcp" or "torch"
    resume_from_checkpoint: str | None = None
    resume_from_vlm_checkpoint: str | None = None

    effective_token_lr_scale: bool = True
    effective_token_lr_exponent: float = 1

    use_lmms_eval: bool = False
    lmms_eval_tasks: str = (
        "mmstar,mmmu_val,ocrbench,textvqa_val,docvqa_val,scienceqa,mme,infovqa_val,chartqa"
    )
    lmms_eval_limit: float | None = None
    lmms_eval_batch_size: int = 64

    val_size: float = 0.1


@dataclass
class GlobalConfig:
    seed: int = 42
    hf_home: str | None = None
    log_dir: str = "logs"
    model_dir: str = "checkpoints"
    eval_dir: str = "evals"
    prefix_run_name: str | None = None
