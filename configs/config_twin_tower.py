from dataclasses import dataclass, field

@dataclass
class GlobalConfig:
    seed: int = 42
    hf_home: str = '/workspace/huggingface/'  # if none, defaults to ~/.cache/huggingface
    same_dir_as_nanovlm_repo: bool = False    # if true, sys.path.append(os.path.join(os.getcwd(), 'nanoVLM'))


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
    lr_mp: float = 5e-05
    lr_vision_backbone: float = 0.0
    lr_language_backbone: float = 1e-05
    freeze_right_tower: bool = False
    lr_right_tower: float = 5e-05

    data_cutoff_idx: int | None = None
    generate_val_data: bool = False
    val_ratio: float = 0.2
    batch_size: int = 64

    use_epochs: bool = True
    max_training_steps: int = 200
    max_epochs: int = 3

    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    max_images_per_example: int = 2
    max_images_per_knapsack: int = 8
    max_sample_length: int = 256

    compile: bool = True
    resume_from_vlm_checkpoint: bool = True

    direct_train_dataset_path: str = "jxie/coco_captions"
    split_name: str = "train"

    train_dataset_path: str = 'HuggingFaceM4/the_cauldron'
    train_dataset_name: tuple[str, ...] = ("tqa",)

    log_wandb: bool = True
    wandb_project: str = 'nanoVLM-twin-tower'
    prefix_run_name: str = "twin-tower"

    eval_every_n_steps: int = 500
    eval_batch_size: int = 64
    max_val_batches: int = 1000
    max_cider_batches: int = 1000
    eval_metric: str = "cider"
    generate_first_n_samples: int = 10
    save_model_every_n_steps: int = 1500

    save_local: bool = False
    local_model_cp_path: str = "checkpoints/nanoVLM-230M-8k-twin-full-ft-sanity"

    save_hf: bool = True
    hf_model_cp_path: str = "patrickamadeus/nanoVLM-230M-8k-twin-full-ft-sanity"