from dataclasses import dataclass, field

@dataclass
class GlobalConfig:
    seed: int = 42
    hf_home: str = None # if none, defaults to ~/.cache/huggingface
    log_dir: str = "logs"
    model_dir: str = "checkpoints"
    eval_dir: str = "evals"
    prefix_run_name: str = None


@dataclass
class VLMConfig:
    vit_hidden_dim: int = 768
    vit_inter_dim: int = 3072
    vit_patch_size: int = 16
    vit_img_size: int = 512
    vit_n_heads: int = 12
    vit_dropout: float = 0.0
    vit_n_blocks: int = 12
    vit_ln_eps: float = 1e-6
    vit_cls_flag: bool = False
    vit_model_type: str = 'google/siglip2-base-patch16-512'

    lm_hidden_dim: int = 960
    lm_inter_dim: int = 2560
    lm_rms_eps: float = 1e-5
    lm_re_base: int = 100000
    lm_max_position_embeddings: int = 8192
    lm_base_vocab_size: int = 49152
    extra_token_amount: int = 66
    lm_vocab_size: int = 49218
    lm_n_heads: int = 15
    lm_n_kv_heads: int = 5
    lm_dropout: float = 0.0
    lm_n_blocks: int = 32
    lm_attn_scaling: float = 1.0
    lm_max_length: int = 4096
    lm_use_tokens: bool = False
    lm_tie_weights: bool = True
    lm_model_type: str = 'HuggingFaceTB/SmolLM2-135M-Instruct'
    lm_tokenizer: str = 'HuggingFaceTB/SmolLM2-135M-Instruct'
    lm_chat_template: str = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"

    mp_pixel_shuffle_factor: int = 4
    mp_image_token_length: int = 64

    max_img_size: int = 2048
    resize_to_max_side_len: bool = True

    vlm_extra_tokens: dict[str, str] = field(default_factory=lambda: {
        "image_token": "<|image|>", "global_image_token": "<|global_image|>",
        **{f"r{r}c{c}": f"<row_{r}_col_{c}>" for r in range(1, 9) for c in range(1, 9)}
    })

    vlm_load_backbone_weights: bool = True
    vlm_checkpoint_path: str = 'lusxvr/nanoVLM-230M-8k'
    hf_repo_name: str = 'nanoVLM'


@dataclass
class TrainConfig:
    # =========================
    # Model/Optimizer Related
    # =========================
    lr_mp: float = 5e-3                     # Learning rate for multimodal projection layers
    lr_vision_backbone: float = 0.0         # Learning rate for vision backbone
    lr_language_backbone: float = 1e-5      # Learning rate for language backbone
    freeze_right_tower: bool = True         # Whether or not to freeze the "right" tower (e.g. language tower)
    lr_right_tower: float = 0.0             # If not frozen, learning rate for right tower
    compile: bool = False                   # Use torch.compile for model/training
    resume_from_vlm_checkpoint: bool = True # Resume full VLM training from checkpoint

    # =========================
    # Data Related
    # =========================
    batch_size: int = 4                     # Per-device batch size
    gradient_accumulation_steps: int = 32   # Gradient accumulation steps (to simulate larger effective batch)
    train_dataset_path: str = 'HuggingFaceM4/the_cauldron'  # HF dataset repo/id for training
    train_dataset_name: tuple[str, ...] = (
        
        # # Subset 1
        "textcaps", "textvqa", "screen2words", "vistext",
        "iam", "cocoqa", "visual7w", "iconqa",

        # # Subset 2
        # "tallyqa", "raven", "spot_the_diff", "vsr", "geomverse", "ai2d", 
        # "diagram_image_to_text", "infographic_vqa", "st_vqa", "okvqa", "intergps",

        # # Subset 3
        # "plotqa", "dvqa", "figureqa", "chart2text", "chartqa",
        # "robut_wikisql", "robut_wtq", "tabmwp", "docvqa",
        # "multihiertt", "tat_qa", "hitab", "tqa", "mapqa",

        # # Subset 4 (grouped)
        # "ocrvqa", "clevr_math", "datikz", "aokvqa", "finqa", "scienceqa", 
        # "robut_sqa", "websight", "visualmrc", "vqarad", "hateful_memes"
    )

    relevance_min_rating: int = 3           # Subset: minimum relevance rating for filtering
    image_correspondence_min_rating: int = 3# Subset: minimum image-text correspondence rating
    visual_dependency_min_rating: int = 3   # Subset: minimum visual dependency rating
    formatting_min_rating: int = 3          # Subset: minimum formatting quality rating
    max_images_per_example: int = 10
    max_images_per_knapsack: int = 18
    max_sample_length: int = 4096
    val_size: int = 50_000                  # Validation/evaluation split sizing (global or per-rank)
    val_ratio: float | None = 0.05          # Optional: use ratio to compute val size
    stream_dataset: bool = True             # Stream dataset from remote source/storage
    interleave_datasets: bool = True        # Enable/disable dataset interleaving for training (if relevant)
    interleave_probabilities: tuple[float, ...] | None = None  # Interleaving: probabilities per dataset
    interleave_stopping_strategy: str = "all_exhausted"        # Interleaving: stop after all exhausted, or on first exhaustion
    streaming_shuffle_buffer: int = 10_000  # Stream buffer for shuffling (higher = more randomness, more memory)
    stratified_val_split: bool = True       # Stratify/split validation set in a balanced way
    data_cutoff_idx: int | None = None      # If not None, cut off dataset at this index for debugging/prototyping

    # =========================
    # Training Loop Related
    # =========================
    max_training_steps: int = 50_000        # Total optimizer steps for training
    max_grad_norm: float = 1.0              # Gradient clipping

    # =========================
    # Evaluation Related
    # =========================
    max_val_batches: int = 128              # Hard limit on batches during evaluation
    eval_in_epochs: bool = True
    eval_interval: int = 500                # Number of steps/epochs between evaluations
    stats_log_interval: int = 100           # Log loss/metrics stats every N steps

    # =========================
    # Logging & Model Saving
    # =========================
    log_wandb: bool = True                  # Enable/disable logging to WandB
    wandb_entity: str = ""                  # Indicate the entity to log to in wandb
    wandb_project: str = 'dualtower-cauldron' # Project name for WandB
    prefix_run_name: str = "dualtower-cauldron" # Prefix for experiment/run
    save_code_cfg: bool = True              # Save configuration/code snapshot with checkpoint
    save_model_every_n_steps: int = 2500    # How often to save model snapshot
    save_local: bool = False                # Save model locally for backup/debugging
    local_model_cp_path: str = "checkpoints/dualtower"  # Checkpoint save path (local)
    save_hf: bool = True                    # Save model to HuggingFace Hub (push_to_hub)
    hf_model_cp_path: str = "patrickamadeus/dualtower"  # HuggingFace Hub repo name/path

    # =========================
    # Extended Evaluation
    # =========================
    use_lmms_eval: bool = False             # Use lmms-eval for evaluation
    lmms_eval_tasks: str = 'mmstar,mmmu_val,ocrbench,textvqa_val,docvqa_val,scienceqa,mme,infovqa_val,chartqa'
    lmms_eval_limit: float = None
    lmms_eval_batch_size: int = 64
