from dataclasses import dataclass, field

@dataclass
class GlobalConfig:
    seed: int = 42
    hf_home: str = None # if none use default ~/.cache/huggingface
    log_dir: str = "logs"
    model_dir: str = "checkpoints"
    eval_dir: str = "evals"
    prefix_run_name: str = None


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
    vit_model_type: str = 'google/siglip2-base-patch16-512'

    lm_hidden_dim: int = 960
    lm_inter_dim: int = 2560
    lm_rms_eps: float = 1e-5
    lm_re_base: int = 100000
    lm_max_position_embeddings: int = 8192
    lm_base_vocab_size: int = 49152
    extra_token_amount: int = 66  # Number of extra tokens for the VLM (image start, image end, image token)
    lm_vocab_size: int = lm_base_vocab_size + extra_token_amount # Not a great way to do this, but it works for now (vlm_extra_tokens cannot be a dict, since this is mutable, and a Field has no len() function)
    lm_n_heads: int = 15
    lm_n_kv_heads: int = 5
    lm_dropout: float = 0.0
    lm_n_blocks: int = 32
    lm_attn_scaling: float = 1.0
    lm_max_length: int = 4096
    lm_use_tokens: bool = False # Decide if the LM expects tokens or embeddings as input (if using as a backbone for the VLM, set to False)
    lm_tie_weights: bool = True # Decide if you want to tie the LM Head weight to the token embedding weights
    lm_model_type: str = 'HuggingFaceTB/SmolLM2-360M-Instruct'
    lm_tokenizer: str = 'HuggingFaceTB/SmolLM2-360M-Instruct'
    lm_chat_template: str = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    mp_pixel_shuffle_factor: int = 4
    mp_image_token_length: int = 64

    max_img_size: int = 2048
    resize_to_max_side_len: bool = False

    vlm_extra_tokens: dict[str, str] = field(default_factory=lambda: {"image_token": "<|image|>", "global_image_token": "<|global_image|>",
      "r1c1": "<row_1_col_1>", "r1c2": "<row_1_col_2>", "r1c3": "<row_1_col_3>", "r1c4": "<row_1_col_4>", "r1c5": "<row_1_col_5>", "r1c6": "<row_1_col_6>", "r1c7": "<row_1_col_7>", "r1c8": "<row_1_col_8>",
      "r2c1": "<row_2_col_1>", "r2c2": "<row_2_col_2>", "r2c3": "<row_2_col_3>", "r2c4": "<row_2_col_4>", "r2c5": "<row_2_col_5>", "r2c6": "<row_2_col_6>", "r2c7": "<row_2_col_7>", "r2c8": "<row_2_col_8>",
      "r3c1": "<row_3_col_1>", "r3c2": "<row_3_col_2>", "r3c3": "<row_3_col_3>", "r3c4": "<row_3_col_4>", "r3c5": "<row_3_col_5>", "r3c6": "<row_3_col_6>", "r3c7": "<row_3_col_7>", "r3c8": "<row_3_col_8>",
      "r4c1": "<row_4_col_1>", "r4c2": "<row_4_col_2>", "r4c3": "<row_4_col_3>", "r4c4": "<row_4_col_4>", "r4c5": "<row_4_col_5>", "r4c6": "<row_4_col_6>", "r4c7": "<row_4_col_7>", "r4c8": "<row_4_col_8>",
      "r5c1": "<row_5_col_1>", "r5c2": "<row_5_col_2>", "r5c3": "<row_5_col_3>", "r5c4": "<row_5_col_4>", "r5c5": "<row_5_col_5>", "r5c6": "<row_5_col_6>", "r5c7": "<row_5_col_7>", "r5c8": "<row_5_col_8>",
      "r6c1": "<row_6_col_1>", "r6c2": "<row_6_col_2>", "r6c3": "<row_6_col_3>", "r6c4": "<row_6_col_4>", "r6c5": "<row_6_col_5>", "r6c6": "<row_6_col_6>", "r6c7": "<row_6_col_7>", "r6c8": "<row_6_col_8>",
      "r7c1": "<row_7_col_1>", "r7c2": "<row_7_col_2>", "r7c3": "<row_7_col_3>", "r7c4": "<row_7_col_4>", "r7c5": "<row_7_col_5>", "r7c6": "<row_7_col_6>", "r7c7": "<row_7_col_7>", "r7c8": "<row_7_col_8>",
      "r8c1": "<row_8_col_1>", "r8c2": "<row_8_col_2>", "r8c3": "<row_8_col_3>", "r8c4": "<row_8_col_4>", "r8c5": "<row_8_col_5>", "r8c6": "<row_8_col_6>", "r8c7": "<row_8_col_7>", "r8c8": "<row_8_col_8>"})
    vlm_load_backbone_weights: bool = True

    momh_enabled: bool = True
    momh_head_pct_vision: float = 0.3  # 40% of heads for V->V only
    momh_head_pct_text: float = 0.2    # 40% of heads for T->T only
    # Remaining 20% (1 - vision - text) for VT->VT cross-modal

    vlm_checkpoint_path: str = 'lusxvr/nanoVLM-230M-8k'
    hf_repo_name: str = 'nanoVLM'

@dataclass
class TrainConfig:
    # =========================
    # Model/Optimizer Related
    # =========================
    lr_mp: float = 0.00512                            # Learning rate for multimodal projection layers
    lr_vision_backbone: float = 0                 # Learning rate for vision backbone
    lr_language_backbone: float = 5e-5              # Learning rate for language backbone
    compile: bool = False                            # Use torch.compile for model/training
    resume_from_vlm_checkpoint: bool = False       # Resume full VLM training from checkpoint

    # =========================
    # Data Related
    # =========================
    batch_size: int = 1                           # Per-device batch size
    gradient_accumulation_steps: int = 32            # Gradient accumulation steps (to simulate larger effective batch)
    train_dataset_path: str = 'HuggingFaceM4/the_cauldron'
    train_dataset_name: tuple[str, ...] = (

        # # Use All
        # "all"

        # # Subset 1 (Basic QA + Text Understanding): 
        "textcaps", "textvqa", "screen2words", "vistext",
         "iam", "cocoqa", "visual7w", "iconqa",

        # # Subset 2: (Spatial Grounding + Reasoning QA)
        # "tallyqa", "raven", "spot_the_diff", "vsr", 
        # "geomverse", "ai2d", "diagram_image_to_text", 
        # "infographic_vqa", "st_vqa", "okvqa", "intergps",

        # # Subset 3: (OCR + Doc / Screen Understanding)
        # "plotqa", "dvqa", "figureqa", "chart2text", "chartqa", 
        # "robut_wikisql", "robut_wtq", "tabmwp", "docvqa", 
        # "multihiertt", "tat_qa", "hitab", "tqa", "mapqa",

        # # Subset 4: (Domain Specific, Science, Math)
        # "ocrvqa", "clevr_math", "datikz", "aokvqa", "finqa", 
        # "scienceqa", "robut_sqa", "websight", "visualmrc", 
        # "vqarad", "hateful_memes"

    )                 

    stream_dataset: bool = True  # Whether to stream the dataset (for large datasets)
    interleave_datasets: bool = True
    interleave_probabilities: tuple[float, ...] | None = None
    interleave_stopping_strategy: str = "all_exhausted"
    streaming_shuffle_buffer: int = 10_000
    stratified_val_split: bool = True
    data_cutoff_idx: int | None = None # limit number of data to be used in train run            # If not None, cut off dataset at this index for debugging/prototyping

    # Subset quality filtering
    relevance_min_rating: int = 1                   # Subset: minimum relevance rating for filtering
    image_correspondence_min_rating: int = 1        # Subset: minimum image-text correspondence rating
    visual_dependency_min_rating: int = 1           # Subset: minimum visual dependency rating
    formatting_min_rating: int = 1                  # Subset: minimum formatting quality rating

    # Packing
    max_images_per_example: int = 10
    max_images_per_knapsack: int = 18
    pack_sequences: bool = False
    max_sample_length: int = 4096

    # Sync Token Efficiency

    # =========================
    # Training Loop Related
    # =========================
    max_training_steps: int = 20_000                   # Total optimizer steps for training
    max_grad_norm: float = 1.0                      # Gradient clipping

    # =========================
    # Evaluation Related
    # =========================
    # Validation/evaluation split sizing (global or per-rank depending on strategy)
    val_size: int = 50_000
    max_val_batches: int = 128                      # Hard limit on batches during evaluation
    eval_in_epochs: bool = True                     # Evaluate in epochs or by global steps
    eval_interval: int = 250                        # Number of steps/epochs between evaluations

    # =========================
    # Logging & Model Saving
    # =========================
    log_wandb: bool = True                          # Enable/disable logging to WandB
    wandb_entity: str = "erlandpg"                          # Indicate the entity to log to in wandb
    wandb_project: str = 'nanoVLM'
    prefix_run_name: str = "momh"
    save_code_cfg: bool = True                # Save configuration/code snapshot with checkpoint
    save_model_every_n_steps: int = 500            # How often to save model snapshot
    stats_log_interval: int = 100                   # Log loss/metrics stats every N steps

    # Model saving configuration
    save_local: bool = False                        # Save model locally for backup/debugging
    local_model_cp_path: str = "checkpoints/vanilla-c1"  # Checkpoint save path (local)
    save_hf: bool = True                           # Save model to HuggingFace Hub (push_to_hub)
    hf_model_cp_path: str = "patrickamadeus/vanilla-c1"  # HuggingFace Hub repo name/path

    # =========================
    # Extended Evaluation
    # =========================
    use_lmms_eval: bool = False                     # Use lmms-eval for evaluation
    lmms_eval_tasks: str = 'mmstar,mmmu_val,ocrbench,textvqa_val,docvqa_val,scienceqa,mme,infovqa_val,chartqa'
    lmms_eval_limit: float = None
    lmms_eval_batch_size: int = 64

