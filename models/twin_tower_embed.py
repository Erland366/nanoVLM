from typing import Optional
from data.processors import get_tokenizer
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.language_model import LanguageModel
from models.vision_language_model import VisionLanguageModel
from models.config import VLMConfig
from models.utils import top_k_top_p_filtering

def get_best_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

class CacheLanguageModel(LanguageModel):
    def __init__(self, cfg: VLMConfig, *, load_backbone: bool = True, freeze_decoder: bool = False):
        if load_backbone:
            lm = LanguageModel.from_pretrained(cfg)
            super().__init__(cfg)
            self.load_state_dict(lm.state_dict())
            del lm
        else:
            super().__init__(cfg)
        self.embd_cache = []
        if freeze_decoder:
            for p in self.parameters():
                p.requires_grad = False
        
    def reset_embd_cache(self):
        self.embd_cache = []
                
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor=None, kv_cache: list[dict]=None, start_pos: int=0):
        if self.lm_use_tokens:
            x = self.token_embedding(x)

        B, T_curr, _ = x.size()
        current_position_ids = torch.arange(start_pos, start_pos + T_curr, device=x.device).unsqueeze(0).expand(B, -1)
        cos, sin = self.rotary_embd(current_position_ids) # Get rotary position embeddings for current tokens

        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)
        
        # store per layer embedding output
        self.embd_cache = []
        for i, block in enumerate(self.blocks):
            x, kv_cache[i] = block(x, cos, sin, attention_mask, kv_cache[i])
            self.embd_cache.append(x)
        x = self.norm(x)

        if self.lm_use_tokens: 
            x = self.head(x) 

        return x, kv_cache




class LeftTower(VisionLanguageModel):
    '''
    Left tower model, only used for the vision understanding.
    Arguments:
        cfg: VLMConfig
        load_backbone: bool -> whether to load the backbone weights
        freeze_vision_encoder: bool -> whether to freeze the vision encoder
        freeze_modality_projector: bool -> whether to freeze the modality projector
        freeze_decoder: bool -> whether to freeze the decoder
    Returns:
        LeftTower -> LeftTower object
    '''
    def __init__(
        self,
        cfg: VLMConfig,
        *,
        load_backbone: bool = True,
        freeze_vision_encoder: bool = False,
        freeze_modality_projector: bool = False,
        freeze_decoder: bool = False,
    ):
        super().__init__(cfg, load_backbone=load_backbone)
        self.decoder = CacheLanguageModel(cfg, load_backbone=load_backbone, freeze_decoder=freeze_decoder)

        if freeze_vision_encoder:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
        if freeze_modality_projector:
            for p in self.MP.parameters():
                p.requires_grad = False
    
    def forward(self, input_ids: torch.Tensor, images, attention_mask=None, targets=None):
        images_tensor = self._process_images(images, input_ids.device)
        token_embd = self.decoder.token_embedding(input_ids)

        if images_tensor is not None:
            image_embd = self.vision_encoder(images_tensor)
            image_embd = self.MP(image_embd)
            token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)

        self.initial_token_embd = token_embd
        logits, _ = self.decoder(token_embd, attention_mask=attention_mask)

        loss = None
        if targets is not None:
            logits = self.decoder.head(logits)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)

        return logits, loss
    
    def get_initial_token_embd(self):
        return self.initial_token_embd
    
    def get_decoder_embd_cache(self):
        return self.decoder.embd_cache





class RightTower(LanguageModel):
    '''
    Right tower model, only used for the language understanding.
    Arguments:
        cfg: VLMConfig
        load_backbone: bool -> whether to load the backbone weights
        freeze_decoder: bool -> whether to freeze the decoder
    Returns:
        RightTower -> RightTower object
    '''
    def __init__(self, cfg: VLMConfig, *, load_backbone: bool = True, freeze_decoder: bool = False):
        if load_backbone:
            lm = LanguageModel.from_pretrained(cfg)
            super().__init__(cfg)
            self.load_state_dict(lm.state_dict())
            del lm
        else:
            super().__init__(cfg)
        if freeze_decoder:
            for p in self.parameters():
                p.requires_grad = False
        self.embd_cache = []

    def reset_embd_cache(self):
        self.embd_cache = []

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        kv_cache: list[dict] = None,
        start_pos: int = 0,
        left_tower_embds: list[torch.Tensor] = None,
        left_tower_img_mask: torch.Tensor = None,
        mask_img_on_first_layer: bool = True
    ):
        if self.lm_use_tokens or x.dim() == 2:
            x = self.token_embedding(x)

        B, T, D = x.size()
        pos_ids = torch.arange(start_pos, start_pos + T, device=x.device).unsqueeze(0).expand(B, -1)
        cos, sin = self.rotary_embd(pos_ids)

        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)

        is_decode_phase = kv_cache[0] is not None
        self.reset_embd_cache()
        for i, block in enumerate(self.blocks):
            if not is_decode_phase:
                # first layer, mask the image tokens (optionally)
                if i == 0 and mask_img_on_first_layer:
                    # create a masked image tokens version only for the first layer
                    first_layer_mask = attention_mask.clone() if attention_mask is not None else torch.ones_like(left_tower_img_mask, dtype=torch.long)
                    first_layer_mask[left_tower_img_mask] = 0
                    x, kv_cache[i] = block(x, cos, sin, first_layer_mask, kv_cache[i])
                else:
                    # for all other layers, replace the image tokens with the left tower embeddings outputs per layer
                    if i > 0:
                        x[left_tower_img_mask] = left_tower_embds[i-1][left_tower_img_mask]
                    
                    # use the original attention mask for other layers
                    x, kv_cache[i] = block(x, cos, sin, attention_mask, kv_cache[i])
            else:
                # during decoding phase, just use the original attention mask
                x, kv_cache[i] = block(x, cos, sin, attention_mask, kv_cache[i])
            self.embd_cache.append(x)
            
        x = self.norm(x)

        if self.lm_use_tokens:
            x = self.head(x)

        return x, kv_cache


class TwinTowerModel(nn.Module):
    '''
    Twin tower model, used for the vision and language understanding.
    Arguments:
        cfg: VLMConfig
        load_backbone: bool -> whether to load the backbone weights
        freeze_vision_encoder: bool -> whether to freeze the vision encoder
        freeze_modality_projector: bool -> whether to freeze the modality projector
        freeze_left_tower_decoder: bool -> whether to freeze the left tower decoder
        freeze_right_tower_decoder: bool -> whether to freeze the right tower decoder
        mask_img_on_first_layer: bool -> whether to mask the image tokens on the first layer
    Returns:
        TwinTowerModel -> TwinTowerModel object
    '''
    def __init__(
        self,
        cfg: VLMConfig,
        *,
        load_backbone: bool = True,
        freeze_vision_encoder: bool = True,
        freeze_modality_projector: bool = False,
        freeze_left_tower_decoder: bool = False,
        freeze_right_tower_decoder: bool = True,
        mask_img_on_first_layer: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        device = get_best_device()
        
        self.left_tower = LeftTower(
            cfg=cfg,
            load_backbone=load_backbone,
            freeze_vision_encoder=freeze_vision_encoder,
            freeze_modality_projector=freeze_modality_projector,
            freeze_decoder=freeze_left_tower_decoder,
        ).to(device)
        
        self.right_tower = RightTower(
            cfg=cfg,
            load_backbone=load_backbone,
            freeze_decoder=freeze_right_tower_decoder,
        ).to(device)
        
        self.tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template)
    
    def _process_images(self, images: torch.Tensor | list[torch.Tensor], device: torch.device):
        if isinstance(images, list):
            if images and isinstance(images[0], list):
                images = [img for sublist in images for img in sublist]
            if not images:
                return None
            return torch.cat(images, dim=0).to(device)
        return images.to(device)

    def forward(
        self, 
        input_ids: torch.Tensor,
        images: torch.Tensor | list[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        mask_img_on_first_layer: bool = True,
    ):
        device = next(self.left_tower.vision_encoder.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if targets is not None:
            targets = targets.to(device)

        # forward pass left tower (capture embds and <image> mask)
        self.left_tower.decoder.reset_embd_cache()
        self.right_tower.reset_embd_cache()
        left_tower_img_mask = (input_ids == self.tokenizer.image_token_id)
        
        # TODO: discuss with Yova, [CLS]-like should we mask or no?
        # if hasattr(self.tokenizer, "global_image_token_id") and self.tokenizer.global_image_token_id is not None:
        #     left_tower_img_mask = left_tower_img_mask | (input_ids == self.tokenizer.global_image_token_id)

        # Don't pass targets to left tower - we only need embeddings, not loss
        logits_left, _ = self.left_tower(input_ids, images, attention_mask=attention_mask, targets=None)
        left_tower_embds = self.left_tower.get_decoder_embd_cache()
        token_embd = self.left_tower.get_initial_token_embd()

        # forward with the right tower
        logits_right, _ = self.right_tower(
            x=token_embd,
            attention_mask=attention_mask,
            left_tower_embds=left_tower_embds,
            left_tower_img_mask=left_tower_img_mask,
            mask_img_on_first_layer=mask_img_on_first_layer,
        )
        
        # calculate loss if targets are provided
        loss = None
        if targets is not None:
            logits_right = self.right_tower.head(logits_right)
            loss = F.cross_entropy(logits_right.reshape(-1, logits_right.size(-1)), targets.reshape(-1), ignore_index=-100)
        
        return logits_right, loss
    
    @torch.inference_mode()
    def generate(self, input_ids, images, attention_mask=None, max_new_tokens=5, top_k=50, top_p=0.9, temperature=0.5, greedy=False, mask_img_on_first_layer=False):
        # Process images
        images_tensor = self._process_images(images, input_ids.device)
        token_embd = self.left_tower.decoder.token_embedding(input_ids)  # [B, T_prompt_text, D_lm]

        # Left tower processing for images
        if images_tensor is not None:
            # Process image with left tower
            image_embd = self.left_tower.vision_encoder(images_tensor)  # [B, T_img_feat, D_model]
            image_embd = self.left_tower.MP(image_embd)  # [B, mp_image_token_length, D_lm]
            # Combine image and text embeddings
            token_embd = self.left_tower._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)

        left_tower_img_mask = (input_ids == self.tokenizer.image_token_id)

        # TODO: discuss with Yova, [CLS]-like should we mask or no?
        # if hasattr(self.tokenizer, "global_image_token_id") and self.tokenizer.global_image_token_id is not None:
        #     left_tower_img_mask = left_tower_img_mask | (input_ids == self.tokenizer.global_image_token_id)
        
        # Left tower forward pass to get embeddings
        self.left_tower.decoder.reset_embd_cache()
        _, _ = self.left_tower.decoder(token_embd, attention_mask=attention_mask)
        left_tower_embds = self.left_tower.decoder.embd_cache

        current_total_seq_len = token_embd.size(1)
        batch_size = input_ids.size(0)
        
        # --- Multimodal Prefill Phase with right tower ---
        self.right_tower.reset_embd_cache()
        prefill_output, kv_cache_list = self.right_tower(
            x=token_embd,                     # embeddings, not input_ids
            attention_mask=attention_mask,
            left_tower_embds=left_tower_embds,
            left_tower_img_mask=left_tower_img_mask,
            mask_img_on_first_layer=mask_img_on_first_layer
        )
        
        last_token_output_from_prefill = prefill_output[:, -1, :]
        
        if not self.right_tower.lm_use_tokens:
            current_logits = self.right_tower.head(last_token_output_from_prefill)
        else:
            current_logits = last_token_output_from_prefill
        
        # Store newly generated token IDs
        newly_generated_ids_list = []
        
        # --- Decode Phase by sampling tokens autoregressively using the kv-cache ---
        for _ in range(max_new_tokens):
            if greedy:
                next_token_id = torch.argmax(current_logits, dim=-1, keepdim=True)
            else:
                filtered_logits = top_k_top_p_filtering(current_logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(filtered_logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            
            newly_generated_ids_list.append(next_token_id)
            
            # Embed the newly generated token
            next_token_embed = self.right_tower.token_embedding(next_token_id)  # [B, 1, D_lm]
            
            # The start_pos for the new token is the current total sequence length *before* adding this new token
            current_token_start_pos = current_total_seq_len
            current_total_seq_len += 1
            
            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat((attention_mask, torch.ones((batch_size, 1), device=attention_mask.device, dtype=attention_mask.dtype)), dim=1)
            
            # With KV cache: only process the new token through right tower
            # Decode loop
            step_input = next_token_embed if not self.right_tower.lm_use_tokens else next_token_id

            decode_step_output, kv_cache_list = self.right_tower(
                x=step_input,
                attention_mask=attention_mask,
                kv_cache=kv_cache_list,
                start_pos=current_token_start_pos,
                left_tower_embds=None,
                left_tower_img_mask=None,
                mask_img_on_first_layer=mask_img_on_first_layer
            )

            last_token_output = decode_step_output[:, -1, :]
            
            # Apply head to get logits (if model is in embedding mode)
            if not self.right_tower.lm_use_tokens:
                current_logits = self.right_tower.head(last_token_output)
            else:
                current_logits = last_token_output
        
        if not newly_generated_ids_list:  # Handle case where max_new_tokens might be 0
            return torch.empty((batch_size, 0), dtype=torch.long, device=input_ids.device)
        
        generated_ids = torch.cat(newly_generated_ids_list, dim=1)
        
        # Post-process to handle EOS token
        if self.tokenizer.eos_token_id is not None and generated_ids.numel() > 0:
            seq_len = generated_ids.size(1)
            device = generated_ids.device
            
            eos_mask = (generated_ids == self.tokenizer.eos_token_id)  # Create a boolean mask for EOS tokens
            
            col_indices_for_min = torch.arange(seq_len, device=device)  # Create column indices [0, 1, ..., seq_len-1]
            
            # In eos_mask, mark positions with actual col_idx, others with a large number
            masked_col_indices = torch.where(eos_mask, col_indices_for_min.unsqueeze(0).expand_as(generated_ids), seq_len + 1)
            
            first_eos_indices_values = torch.min(masked_col_indices, dim=1).values
            
            # Clamp values to seq_len (if no EOS found, min will be seq_len + 1, clamp brings it to seq_len)
            actual_first_eos_indices = torch.clamp(first_eos_indices_values, max=seq_len)
            
            # Create column indices for comparison, shape [batch_size, seq_len]
            col_indices_for_comparison = torch.arange(seq_len, device=device).unsqueeze(0).expand_as(generated_ids)
            
            # Tokens are replaced if their column index is greater than the index of the first EOS token
            replace_mask = col_indices_for_comparison > actual_first_eos_indices.unsqueeze(1)
            
            generated_ids[replace_mask] = self.tokenizer.eos_token_id
        
        return generated_ids
    
    @classmethod
    def _load_twin_tower_checkpoint(
        cls,
        cfg: VLMConfig,
        *,
        freeze_vision_encoder: bool = False,
        freeze_modality_projector: bool = False,
        freeze_left_tower_decoder: bool = False,
        freeze_right_tower_decoder: bool = True,
        mask_img_on_first_layer: bool = True
    ):
        import os
        import json
        from safetensors.torch import load_model
        from huggingface_hub import hf_hub_download
        
        repo_id_or_path = cfg.vlm_checkpoint_path
        
        if os.path.exists(repo_id_or_path):
            config_path = os.path.join(repo_id_or_path, "config.json")
            weights_path = os.path.join(repo_id_or_path, "model.safetensors")
            print("THIS IS CONFIG PATH!")
            print(config_path)
            
            if not os.path.exists(config_path):
                raise ValueError(f"Config file not found at {config_path}")
            if not os.path.exists(weights_path):
                raise ValueError(f"Weights file not found at {weights_path}")
        else:
            
            config_path = hf_hub_download(repo_id=repo_id_or_path, filename="config.json")
            weights_path = hf_hub_download(repo_id=repo_id_or_path, filename="model.safetensors")
        
        with open(config_path, "r") as f:
            checkpoint_cfg = VLMConfig(**json.load(f))
        
        twin_tower = cls(
            checkpoint_cfg,
            load_backbone=False,
            freeze_vision_encoder=freeze_vision_encoder,
            freeze_modality_projector=freeze_modality_projector,
            freeze_left_tower_decoder=freeze_left_tower_decoder,
            freeze_right_tower_decoder=freeze_right_tower_decoder,
            mask_img_on_first_layer=mask_img_on_first_layer
        )
        
        load_model(twin_tower, weights_path)
        
        return twin_tower

    @classmethod
    def from_pretrained(
        cls, 
        cfg: VLMConfig,
        *,
        freeze_vision_encoder: bool = True,
        freeze_modality_projector: bool = False,
        freeze_left_tower_decoder: bool = False,
        freeze_right_tower_decoder: bool = True,
        mask_img_on_first_layer: bool = True,
    ):
        try:
            return cls._load_twin_tower_checkpoint(
                cfg, 
                freeze_vision_encoder=freeze_vision_encoder,
                freeze_modality_projector=freeze_modality_projector,
                freeze_left_tower_decoder=freeze_left_tower_decoder,
                freeze_right_tower_decoder=freeze_right_tower_decoder,
                mask_img_on_first_layer=mask_img_on_first_layer
            )
        except Exception as e:
            # print(f"Failed to load as twin tower checkpoint: {e}")
            print("Falling back to loading from separate VLM and LM checkpoints...")
            vlm_model = VisionLanguageModel.from_pretrained(cfg.vlm_checkpoint_path)
            language_model = LanguageModel.from_pretrained(cfg)
            
            twin_tower = cls(
                cfg,
                load_backbone=False,
                freeze_vision_encoder=freeze_vision_encoder,
                freeze_modality_projector=freeze_modality_projector,
                freeze_left_tower_decoder=freeze_left_tower_decoder,
                freeze_right_tower_decoder=freeze_right_tower_decoder,
                mask_img_on_first_layer=mask_img_on_first_layer
            )
            
            twin_tower.left_tower.vision_encoder.load_state_dict(vlm_model.vision_encoder.state_dict())
            twin_tower.left_tower.MP.load_state_dict(vlm_model.MP.state_dict())
            twin_tower.left_tower.decoder.load_state_dict(vlm_model.decoder.state_dict())
            twin_tower.right_tower.load_state_dict(language_model.state_dict())
            
            # Re-apply freezing after loading weights (load_state_dict resets requires_grad)
            if freeze_vision_encoder:
                for p in twin_tower.left_tower.vision_encoder.parameters():
                    p.requires_grad = False
            if freeze_modality_projector:
                for p in twin_tower.left_tower.MP.parameters():
                    p.requires_grad = False
            if freeze_left_tower_decoder:
                for p in twin_tower.left_tower.decoder.parameters():
                    p.requires_grad = False
            if freeze_right_tower_decoder:
                for p in twin_tower.right_tower.parameters():
                    p.requires_grad = False
            
            del vlm_model
            del language_model
            
            return twin_tower

    def save_pretrained(self, save_directory):
        import json
        import os
        from safetensors.torch import save_model
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(self.cfg.__dict__, f, indent=4)
        save_model(self, os.path.join(save_directory, "model.safetensors"))
    
    def push_to_hub(self, repo_id: str, private: bool = False):
        import tempfile
        from huggingface_hub import create_repo, upload_folder
        
        repo_url = create_repo(repo_id=repo_id, private=private, exist_ok=True)
        repo_id = repo_url.repo_id
        print(f"Created repo: {repo_url}")
        
        with tempfile.TemporaryDirectory() as save_path:
            self.save_pretrained(save_path)
            
            with open(os.path.join(save_path, "README.md"), "w") as f:
                f.write(self._get_model_card_template().format(repo_id=repo_id))
            
            return upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=save_path,
                commit_message="Upload twin-tower VLM using push_to_hub",
            )
    
    def _get_model_card_template(self) -> str:
        """Get the model card template for twin-tower model."""
        return """
---
library_name: nanovlm
license: mit
pipeline_tag: image-text-to-text
tags:
  - vision-language
  - multimodal
  - research
  - twin-tower
---

**Twin-Tower VLM** is a vision-language model based on the twin-tower architecture. This model uses a separate vision tower to process images and generate per-layer contexts, which are then integrated with a frozen language tower for text generation.

## Architecture

The twin-tower architecture consists of:

1. **Vision Tower**: Processes images through vision encoder → modality projector → decoder layers to create per-layer contexts
2. **Language Tower**: Frozen language model that receives vision contexts and generates text

## Key Features

- **Twin-Tower Design**: Separate processing of vision and language with per-layer context integration
- **Frozen Language Tower**: Language model parameters are frozen, gradients flow through vision contexts
- **Per-Layer Contexts**: Vision tower generates contexts for each language model layer
- **Efficient Training**: Only vision tower components are trainable

## Usage

```python
from twin_tower import VisionLanguageTwinTowerModel
from config import VLMConfig

# Load the model
cfg = VLMConfig()
model = VisionLanguageTwinTowerModel.from_pretrained(cfg)

# Generate text from image
from PIL import Image
image = Image.open("your_image.jpg")
result = model.generate_from_text("What is in this image?", image)
print(result)
```

## Model Details

- **Base Model**: {repo_id}
- **Architecture**: Twin-Tower VLM
- **Vision Encoder**: SigLIP-based
- **Language Model**: SmolLM2-based
- **Parameters**: ~230M total (vision tower trainable, language tower frozen)

For more information, check out the base nanoVLM model: https://huggingface.co/lusxvr/nanoVLM-222M.
"""


