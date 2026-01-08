from dataclasses import dataclass, field

@dataclass
class GlobalConfig:
    seed: int = 42
    hf_home: str = '/workspace/huggingface/'  # if none, defaults to ~/.cache/huggingface
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
    lm_max_length: int = 256
    lm_use_tokens: bool = False
    lm_tie_weights: bool = True
    lm_model_type: str = 'HuggingFaceTB/SmolLM2-135M-Instruct'
    lm_tokenizer: str = 'HuggingFaceTB/SmolLM2-360M-Instruct'
    lm_chat_template: str = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"

    mp_pixel_shuffle_factor: int = 4
    mp_image_token_length: int = 64

    max_img_size: int = 512
    resize_to_max_side_len: bool = False

    vlm_extra_tokens: dict[str, str] = field(default_factory=lambda: {
        "image_token": "<|image|>", "global_image_token": "<|global_image|>",
        **{f"r{r}c{c}": f"<row_{r}_col_{c}>" for r in range(1, 9) for c in range(1, 9)}
    })

    vlm_load_backbone_weights: bool = True
    vlm_checkpoint_path: str = 'lusxvr/nanoVLM-230M-8k'
    hf_repo_name: str = 'nanoVLM'


@dataclass
class TrainConfig:
    # model related
    # 1. learning rate
    # 2. model compiling
    # 3. checkpoint resuming
    lr_mp: float = 1e-04
    lr_vision_backbone: float = 0.0
    lr_language_backbone: float = 5e-05
    freeze_right_tower: bool = True
    lr_right_tower: float = 0.0
    compile: bool = False
    resume_from_vlm_checkpoint: bool = True

    # data related
    # 0. batch size + grad. accum (effective batch size)
    batch_size: int = 64
    gradient_accumulation_steps: int = 4
    # 1. using the_cauldron preset / custom data source
    train_dataset_path: str = 'howard-hou/OCR-VQA'
    train_dataset_name: tuple[str, ...] = ("tqa",)  # e.g., ("ocrvqa", ) or ("tqa", "ocrvqa")
    # 1.1. dataset subset for the the_cauldron needed, including the relevance min_rating, etc.
    relevance_min_rating: int = 1
    image_correspondence_min_rating: int = 1
    visual_dependency_min_rating: int = 1
    formatting_min_rating: int = 1
    # 1.2 custom data source
    use_custom_dataset: bool = True
    custom_dataset_id: str = "ocr_vqa_dual"
    train_split_name: str = "train"
    val_split_name: str = "validation"
    # 2. automatic validation set generation or direct from the path subset
    generate_val_data: bool = False # if true, generate val data from train data
    val_ratio: float = 0.2
    # 3. data packing related including samples length + others
    use_packing: bool = False
    max_images_per_example: int = 1
    max_images_per_knapsack: int = 1
    max_sample_length: int = 256
    stream_dataset: bool = False  # Whether to stream the dataset (for large datasets)
    data_cutoff_idx: int | None = None # limit number of data to be used in train run
    

    # training loop related
    # 1. using epochs / training steps
    # 2. max_training steps + max epochs defined
    # 3. max_grad_norm
    use_epochs: bool = True
    max_epochs: int = 5
    max_training_steps: int = 200
    max_grad_norm: float = 1.0


    # evaluation related
    # 1. evaluation dataloader preset
    # 2. evaluation metric preset
    # 3. evaluation steps
    # 4. evaluation

    custom_eval_dataset_id: str = "ocr_vqa_dual"
    custom_eval_dataset_path: str = "howard-hou/OCR-VQA"
    # custom_eval_dataset_id: str = "coco_captions_dual"
    # custom_eval_dataset_path: str = "patrickamadeus/coco_caption_val_unique"
    custom_eval_dataset_split_name: str = "validation"
    eval_metric: str = "accuracy"
    eval_every_n_steps: int = 500 # evaluate every x steps
    eval_batch_size: int = 128 # batch size for evaluation
    max_val_batches: int = 10000 # hard limit on number of batches to evaluate on
    max_cider_batches: int = 1000
    generate_first_n_samples: int = 10
    max_gen_samples: int = 50


    # logging & model saving related
    # 1. wandb logging
    log_wandb: bool = True
    wandb_project: str = 'dual-tower-ocrvqa'
    prefix_run_name: str = "dual-tower-ocrvqa"
    save_code_cfg: bool = True
    # 2. save model & generate every x steps
    save_model_every_n_steps: int = 1500
    generate_every_n_steps: int = 1500
    # 4. model saving
    save_local: bool = False
    local_model_cp_path: str = "checkpoints/dual-tower"
    save_hf: bool = False
    hf_model_cp_path: str = "patrickamadeus/dual-tower"