from dataclasses import dataclass, field

@dataclass
class GlobalConfig:
    seed: int = 42
    hf_home: str = '/workspace/huggingface/' # if none use default ~/.cache/huggingface
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
    lm_max_length: int = 256
    lm_use_tokens: bool = False # Decide if the LM expects tokens or embeddings as input (if using as a backbone for the VLM, set to False)
    lm_tie_weights: bool = True # Decide if you want to tie the LM Head weight to the token embedding weights
    lm_model_type: str = 'HuggingFaceTB/SmolLM2-135M-Instruct'
    lm_tokenizer: str = 'HuggingFaceTB/SmolLM2-360M-Instruct'
    lm_chat_template: str = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    mp_pixel_shuffle_factor: int = 4
    mp_image_token_length: int = 64

    max_img_size: int = 512
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
    vlm_checkpoint_path: str = 'lusxvr/nanoVLM-460M-8k'
    hf_repo_name: str = 'nanoVLM'

@dataclass
class TrainConfig:
    # model related
    # 1. learning rate
    # 2. model compiling
    # 3. checkpoint resuming
    lr_mp: float = 1e-5
    lr_vision_backbone: float = 0.0
    lr_language_backbone: float = 5e-6
    compile: bool = True
    resume_from_vlm_checkpoint: bool = True



    # data related
    # 0. batch size + grad. accum (effective batch size)
    batch_size: int = 64
    gradient_accumulation_steps: int = 4
    # 1. using the_cauldron preset / custom data source
    train_dataset_path: str = 'HuggingFaceM4/the_cauldron'
    train_dataset_name: tuple[str, ...] = ("ocrvqa", )  # e.g., ("ocrvqa", ) or ("tqa", "ocrvqa")
    # 1.1. dataset subset for the the_cauldron needed, including the relevance min_rating, etc.
    relevance_min_rating: int = 1
    image_correspondence_min_rating: int = 1
    visual_dependency_min_rating: int = 1
    formatting_min_rating: int = 1
    # 1.2 custom data source
    use_custom_dataset: bool = True
    train_dataset_path: str = 'jxie/coco_captions'
    custom_dataset_id: bool = "coco_captions"
    train_split_name: str = "train"
    val_split_name: str = "validation"
    # 2. automatic validation set generation or direct from the path subset
    generate_val_data: bool = False # if true, generate val data from train data
    val_ratio: float = 0.2
    # 3. data packing related including samples length + others
    use_packing: bool = True
    max_images_per_example: int = 1
    max_images_per_knapsack: int = 1
    max_sample_length: int = 256
    stream_dataset: bool = False  # Whether to stream the dataset (for large datasets)
    data_cutoff_idx: int = None # limit number of data to be used in train run
    
    

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
    custom_eval_dataset_id: str = "coco_captions"
    custom_eval_dataset_path: str = "patrickamadeus/coco_caption_val_uniq"
    eval_metric: str = "cider"
    eval_every_n_steps: int = 500 # evaluate every x steps
    eval_batch_size: int = 128 # batch size for evaluation
    max_val_batches: int = 1000 # hard limit on number of batches to evaluate on


    # logging & model saving related
    # 1. wandb logging
    log_wandb: bool = True
    wandb_project: str = 'dual-tower'
    prefix_run_name: str = "vanilla-460m"
    save_code_cfg: bool = True
    # 2. save model & generate every x steps
    save_model_every_n_steps: int = 1500
    generate_every_n_steps: int = 1500
    max_gen_batches: int = 10
    # 4. model saving
    save_local: bool = False
    local_model_cp_path: str = "checkpoints/vanilla-460m"
    save_hf: bool = True
    hf_model_cp_path: str = "patrickamadeus/vanilla-460m"