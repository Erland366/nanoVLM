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

class LeftTower(VisionLanguageModel):
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
        if freeze_vision_encoder:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
        if freeze_modality_projector:
            for p in self.MP.parameters():
                p.requires_grad = False
        if freeze_decoder:
            for p in self.decoder.parameters():
                p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, images, attention_mask=None):
        images_tensor = self._process_images(images, input_ids.device)
        token_embd = self.decoder.token_embedding(input_ids)

        if images_tensor is not None:
            image_embd = self.vision_encoder(images_tensor)
            image_embd = self.MP(image_embd)
            token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)

        logits, kv_cache = self.decoder(token_embd, attention_mask=attention_mask)

        return logits, kv_cache



class RightTower(LanguageModel):
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

    def _make_image_causal_mask(self, B, T_curr, img_seq_len, x_dtype, x_device):
        total_kv_len = img_seq_len + T_curr
        mask = torch.zeros((B, 1, T_curr, total_kv_len), dtype=x_dtype, device=x_device)
        text_causal = torch.tril(torch.ones((T_curr, T_curr), dtype=torch.bool, device=x_device))
        mask[:, :, :, :img_seq_len] = 0  # attend all image tokens
        mask[:, :, :, img_seq_len:] = torch.where(
            text_causal,
            torch.tensor(0.0, dtype=x_dtype, device=x_device),
            torch.tensor(float('-inf'), dtype=x_dtype, device=x_device)
        )
        return mask

    def _combine_causal_and_padding_mask(self, causal_mask, attention_mask, total_kv_len, x_dtype, x_device):
        orig_mask = attention_mask[:, :total_kv_len]  # [B, total_kv_len]
        if orig_mask.dim() == 2:
            padding_mask = orig_mask.unsqueeze(1).unsqueeze(2)
        else:
            padding_mask = orig_mask
        padding_additive = torch.where(
            padding_mask == 0,
            torch.tensor(float('-inf'), dtype=x_dtype, device=x_device),
            torch.tensor(0.0, dtype=x_dtype, device=x_device)
        )
        return causal_mask + padding_additive

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        img_kv_cache: list[dict] = None,
        kv_cache: list[dict] = None,
        start_pos: int = 0,
    ):
        if self.lm_use_tokens:
            x = self.token_embedding(x)

        B, T_curr, _ = x.size()
        if img_kv_cache is not None and any(cache is not None for cache in img_kv_cache):
            img_seq_len = img_kv_cache[0]['key'].size(2)
            text_start_pos = start_pos + img_seq_len
            # dont start from 0, because we already have image tokens in the cache
            current_position_ids = torch.arange(
                text_start_pos, 
                text_start_pos + T_curr, 
                device=x.device
            ).unsqueeze(0).expand(B, -1)

            # we need novel causal masking since the T_kv no longer == T_curr, bcs we hack it to add the kv sequence.
            # this is still hacky way and needs modification on @language_model.py, so we gotta override it
            # future work is to modify the sequence / language model  / the mask handling in more elegant way to keep the causal going
            # for now, this is a hacky way to get the causal masking to work
            if T_curr > 1:
                total_kv_len = img_seq_len + T_curr
                causal_mask = self._make_image_causal_mask(B, T_curr, img_seq_len, x.dtype, x.device)
                if attention_mask is not None:
                    attention_mask = self._combine_causal_and_padding_mask(
                        causal_mask, attention_mask, total_kv_len, x.dtype, x.device
                    )
                else:
                    attention_mask = causal_mask
        else:
            current_position_ids = torch.arange(
                start_pos, 
                start_pos + T_curr, 
                device=x.device
            ).unsqueeze(0).expand(B, -1)
        
        cos, sin = self.rotary_embd(current_position_ids)

        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)

        for i, block in enumerate(self.blocks):
            # combine image KV cache with the current KV cache if available
            combined_kv_cache = None
            if img_kv_cache is not None and img_kv_cache[i] is not None:
                if kv_cache[i] is not None:
                    combined_kv_cache = {
                        'key': torch.cat([img_kv_cache[i]['key'], kv_cache[i]['key']], dim=2),
                        'value': torch.cat([img_kv_cache[i]['value'], kv_cache[i]['value']], dim=2)
                    }
                else:
                    # just use the image KV cache if no text KV cache yet
                    combined_kv_cache = img_kv_cache[i]
            else:
                combined_kv_cache = kv_cache[i]
            
            x, new_kv_cache = block(x, cos, sin, attention_mask, combined_kv_cache)
            
            # only store the part of the KV cache that corresponds to the text tokens
            # this so that we don't duplicate the image tokens in the cache
            if img_kv_cache is not None and img_kv_cache[i] is not None and new_kv_cache is not None:
                img_seq_len = img_kv_cache[i]['key'].size(2)
                if new_kv_cache['key'].size(2) > img_seq_len:
                    kv_cache[i] = {
                        'key': new_kv_cache['key'][:, :, img_seq_len:],
                        'value': new_kv_cache['value'][:, :, img_seq_len:]
                    }
                else:
                    kv_cache[i] = None
            else:
                kv_cache[i] = new_kv_cache

        x = self.norm(x)

        if self.lm_use_tokens: 
            x = self.head(x) 

        return x, kv_cache


class TwinTowerModel(nn.Module):
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

        # get last image token position
        image_token_mask = (input_ids == self.tokenizer.image_token_id)
        positions = torch.where(image_token_mask[0])[0]
        split_pos = positions[-1].item() + 1 if len(positions) > 0 else 0
        
        # split input_ids and attention_mask
        left_input_ids = input_ids[:, :split_pos]
        right_input_ids = input_ids[:, split_pos:]
        left_attention_mask = attention_mask[:, :split_pos] if attention_mask is not None else None
        
        # forward pass through left tower to get KV cache
        _, img_kv_cache = self.left_tower(left_input_ids, images, attention_mask=left_attention_mask)
        
        # prepare right tower input - handle both token and embedding modes
        if self.right_tower.lm_use_tokens:
            right_input = right_input_ids
        else:
            right_input = self.right_tower.token_embedding(right_input_ids)
        
        logits_right, _ = self.right_tower(
            x=right_input,
            attention_mask=attention_mask,  # full mask, including the mask for kv cache (pad 0 + look 1)
            img_kv_cache=img_kv_cache,
            kv_cache=None,
            start_pos=0,
        )
        
        loss = None
        if targets is not None:
            if not self.right_tower.lm_use_tokens:
                logits_right = self.right_tower.head(logits_right)
            # create full logits tensor with vocab size  
            full_logits = torch.zeros(input_ids.size(0), input_ids.size(1), logits_right.size(-1), device=device, dtype=logits_right.dtype)
            # fill in right tower logits
            full_logits[:, split_pos:, :] = logits_right
            # Loss: targets already have -100 for image/padding positions, so they're ignored
            loss = F.cross_entropy(full_logits.reshape(-1, full_logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        
        return logits_right, loss
    
    @torch.inference_mode()
    def generate(self, input_ids, images, attention_mask=None, max_new_tokens=5, top_k=50, top_p=0.9, temperature=0.5, greedy=False, mask_img_on_first_layer=False):
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Find split position - image tokens are aligned across batch
        image_token_mask = (input_ids == self.tokenizer.image_token_id)
        positions = torch.where(image_token_mask[0])[0]
        split_pos = positions[-1].item() + 1 if len(positions) > 0 else 0
        
        # Split input_ids
        left_input_ids = input_ids[:, :split_pos]
        right_input_ids = input_ids[:, split_pos:]
        left_attention_mask = attention_mask[:, :split_pos] if attention_mask is not None else None
        
        # Process images through left tower components
        images_tensor = self._process_images(images, device)
        # Left tower uses its own token embedding
        left_token_embd = self.left_tower.decoder.token_embedding(left_input_ids)

        if images_tensor is not None:
            # Process image with left tower
            image_embd = self.left_tower.vision_encoder(images_tensor)  # [B, T_img_feat, D_model]
            image_embd = self.left_tower.MP(image_embd)  # [B, mp_image_token_length, D_lm]
            # Combine image and text embeddings
            left_token_embd = self.left_tower._replace_img_tokens_with_embd(left_input_ids, left_token_embd, image_embd)
        
        # --- Prefill Phase: Process left part through left tower decoder to get image KV cache ---
        _, img_kv_cache = self.left_tower.decoder(left_token_embd, attention_mask=left_attention_mask)
        
        # Right tower uses its own token embedding
        right_token_embd = self.right_tower.token_embedding(right_input_ids)
        current_right_seq_len = right_token_embd.size(1)
        
        # --- Multimodal Prefill Phase with right tower ---
        # Right tower processes the right part of prompt with image KV cache
        # Pass the FULL attention mask (not split) so right tower can see the complete sequence
        prefill_input = right_token_embd if not self.right_tower.lm_use_tokens else right_input_ids
        prefill_output, text_kv_cache = self.right_tower(
            x=prefill_input,
            attention_mask=attention_mask,  # Full mask, not split
            img_kv_cache=img_kv_cache,
            kv_cache=None,
            start_pos=0,
        )
        
        last_token_output_from_prefill = prefill_output[:, -1, :]
        
        if not self.right_tower.lm_use_tokens:
            current_logits = self.right_tower.head(last_token_output_from_prefill)
        else:
            current_logits = last_token_output_from_prefill
        
        newly_generated_ids_list = []
        
        # --- Decode Phase: Generate tokens autoregressively using KV cache ---
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
            
            # The start_pos for the new token in the right tower context
            current_token_start_pos = current_right_seq_len
            current_right_seq_len += 1
            
            # Update attention mask to include the new token (extend the full mask)
            if attention_mask is not None:
                attention_mask = torch.cat((attention_mask, torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)), dim=1)
            
            # Process the new token through right tower with both image and text KV caches
            step_input = next_token_embed if not self.right_tower.lm_use_tokens else next_token_id

            decode_step_output, text_kv_cache = self.right_tower(
                x=step_input,
                attention_mask=attention_mask,  # Full mask, extended with new token
                img_kv_cache=img_kv_cache,
                kv_cache=text_kv_cache,
                start_pos=current_token_start_pos,
            )

            last_token_output = decode_step_output[:, -1, :]
            
            # Apply head to get logits (if model is in embedding mode)
            if not self.right_tower.lm_use_tokens:
                current_logits = self.right_tower.head(last_token_output)
            else:
                current_logits = last_token_output
        
        if not newly_generated_ids_list:  # Handle case where max_new_tokens might be 0
            return torch.empty((batch_size, 0), dtype=torch.long, device=device)
        
        generated_ids = torch.cat(newly_generated_ids_list, dim=1)
        
        # Post-process to handle EOS token
        if self.tokenizer.eos_token_id is not None and generated_ids.numel() > 0:
            seq_len = generated_ids.size(1)
            
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


