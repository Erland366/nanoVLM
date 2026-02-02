import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import tempfile
from dataclasses import asdict
from safetensors.torch import load_model as load_safetensors, save_model
from models.dual_tower.dual_language_model import LanguageModel
from models.vision_language_model import VisionLanguageModel
from models.config import VLMConfig
from models.utils import top_k_top_p_filtering
from huggingface_hub import create_repo, hf_hub_download, upload_folder



class LeftTower(VisionLanguageModel):
    def __init__(
        self,
        cfg: VLMConfig,
        *,
        load_backbone: bool = True,
        freeze_vision_encoder: bool = False,
        freeze_modality_projector: bool = False,
        freeze_language_decoder: bool = False,
    ):
        super().__init__(cfg, load_backbone=load_backbone)
        if freeze_vision_encoder:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
        if freeze_modality_projector:
            for p in self.MP.parameters():
                p.requires_grad = False
        if freeze_language_decoder:
            for p in self.decoder.parameters():
                p.requires_grad = False

    def forward(
        self, 
        input_ids: torch.Tensor, 
        images, 
        attention_mask: torch.Tensor = None,
    ):
        images_tensor = self._process_images(images, input_ids.device)
        token_embd = self.decoder.token_embedding(input_ids)

        if images_tensor is not None:
            image_embd = self.vision_encoder(images_tensor)
            image_embd = self.MP(image_embd)
            token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)
        
        _, kv_cache = self.decoder(token_embd, attention_mask=attention_mask)
        
        return None, kv_cache


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
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        kv_cache: list[dict] = None,
        start_pos = 0,
    ):
        if kv_cache is None:
            raise ValueError("kv_cache must be provided from LeftTower result to RightTower.forward(). It cannot be None.")
        
        B, T_text, _ = x.size()
        
        # Create position_ids for the current text sequence based on start_pos
        if isinstance(start_pos, torch.Tensor):
             if start_pos.dim() == 1:
                 start_pos = start_pos.unsqueeze(1) # (B, 1)
             
             # Create offsets [0, 1, ..., T_text-1]
             offsets = torch.arange(T_text, device=x.device).unsqueeze(0) # (1, T_text)
             current_position_ids = start_pos + offsets # (B, T_text)
        else:
            current_position_ids = torch.arange(start_pos, start_pos + T_text, device=x.device).unsqueeze(0).expand(B, -1)
        
        # For rotary embeddings, we need the attention mask to match the current sequence length being processed
        # During autoregressive decoding (T_text=1):
        # We must NOT pass the sliced mask to RoPE because RoPE's skippable logic (cumsum) requires full history.
        # Passing a slice [1] resets position to 0.
        # Instead, we rely on the pre-calculated current_position_ids (via start_pos) and pass None.
        
        rope_mask = attention_mask
        if attention_mask is not None and attention_mask.size(1) != T_text:
             # We are in decoding phase (likely), and mask is full history but T_text is 1.
             # We must NOT pass the full mask because RoPE expects mask shape == pos_ids shape.
             # We must NOT pass the sliced mask because it breaks cumsum logic for skippable RoPE.
             # So we pass None, and trust that current_position_ids is correct.
             rope_mask = None

        cos, sin = self.rotary_embd(current_position_ids, rope_mask)
        
        # Process through all transformer blocks
        # kv_cache is List[N] for N layers
        # Each layer i uses kv_cache[i] for its own attention mechanism
        for i, block in enumerate(self.blocks):
            x, kv_cache[i] = block(x, cos, sin, attention_mask, kv_cache[i])
        
        # Final normalization
        x = self.norm(x)
        
        # Compute logits if we are using tokens, otherwise stay in the embedding space
        if self.lm_use_tokens:
            x = self.head(x)
        
        return x, kv_cache

    @torch.inference_mode()
    def generate(self, inputs: torch.Tensor, max_new_tokens: int=20, attention_mask: torch.Tensor=None):
        """
        Generate tokens autoregressively from a given input sequence.

        Args:
            inputs (torch.Tensor): Input tensor containing token indices or embeddings.
                Shape: (batch_size, sequence_length) or (sequence_length,) for a single sequence.
            max_new_tokens (int): Number of new tokens to generate after the input sequence.
            attention_mask (torch.Tensor, optional): Attention mask for the input sequence.
                Shape: (batch_size, sequence_length). 1 for valid tokens, 0 for padding.

        Returns:
            torch.Tensor: The generated sequence, including the original inputs and newly generated tokens.
                Shape: (batch_size, sequence_length + max_new_tokens)
        """
        # Add batch dimension if needed
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        generated_outputs = inputs.clone()
        
        # Handle attention mask
        if attention_mask is None:
            # If no mask provided, assume all tokens are valid
            attention_mask = torch.ones_like(inputs, dtype=torch.long)
        elif attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        
        # Clone the attention mask so we can extend it during generation
        current_attention_mask = attention_mask.clone()

        #  -- Prefill phase --
        prompt_output, kv_cache_list = self.forward(
            generated_outputs, 
            attention_mask=current_attention_mask,
            kv_cache=None,
            start_pos=0
        )
        last_output = prompt_output[:, -1, :]
        
        # count non <pad> area, this is the valid final position ID before autoregressive increment++
        # Skippable: track actual valid tokens count per sample
        # Note: Even though we pass single-token mask during generation loop to forward(), 
        # this `current_token_start_pos` tensor maintains the correct cumulative position (accounting for skipped pads)
        # which we pass as `start_pos` to correctly generate position_ids in forward().
        current_token_start_pos = current_attention_mask.sum(dim=1) - 1 # Shape (B,)

        # Autoregressive generation loop up to `max_new_tokens`
        for i in range(max_new_tokens):
            if self.lm_use_tokens:
                # Now the model outputs logits
                next_output = torch.argmax(last_output, dim=-1, keepdim=True)
            else:
                # Now the model outputs embeddings
                next_output = last_output.unsqueeze(1)

            generated_outputs = torch.cat((generated_outputs, next_output), dim=1)
            
            # Extend attention mask for the new token (it's always valid, not padding, so extend [...,1])
            new_token_mask = torch.ones((current_attention_mask.size(0), 1), 
                                        dtype=current_attention_mask.dtype, 
                                        device=current_attention_mask.device)
            current_attention_mask = torch.cat([current_attention_mask, new_token_mask], dim=1)
            
            # increment pos ID for next token
            current_token_start_pos += 1
            if i == max_new_tokens - 1: 
                break

            decode_step_output, kv_cache_list = self.forward(
                next_output, 
                attention_mask=current_attention_mask,
                kv_cache=kv_cache_list,
                start_pos=current_token_start_pos
            )
            last_output = decode_step_output[:, -1, :] 
    
        return generated_outputs


class DualTowerVLM(nn.Module):
    def __init__(
        self,
        cfg: VLMConfig,
        *,
        load_backbone: bool = True,
        freeze_left_vision: bool = False,
        freeze_left_projector: bool = False,
        freeze_left_decoder: bool = False,
        freeze_right_decoder: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.left_tower = LeftTower(
            cfg,
            load_backbone=load_backbone,
            freeze_vision_encoder=freeze_left_vision,
            freeze_modality_projector=freeze_left_projector,
            freeze_language_decoder=freeze_left_decoder,
        )
        self.right_tower = RightTower(
            cfg,
            load_backbone=load_backbone,
            freeze_decoder=freeze_right_decoder,
        )
        self.tokenizer = self.left_tower.tokenizer

    def forward(
        self,
        input_ids: torch.Tensor,
        images,
        last_img_idx: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        targets: torch.Tensor = None
    ):
        # Process left tower to grab kv_cache
        _, kv_cache = self.left_tower(
            input_ids= input_ids[:, :last_img_idx + 1],
            images=images,
            attention_mask=attention_mask[:, :last_img_idx + 1],
        )

        # Process to right tower and use left's kv_cache
        # we need to embed this first (why?) because input_ids being passed is pure token ID, see @train.py / generate.py
        full_embd = self.right_tower.token_embedding(input_ids)
        logits, _ = self.right_tower(
            x=full_embd,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            start_pos=0
        )

        # Loss calculation (if any)
        loss = None
        if targets is not None:
            logits = self.right_tower.head(logits)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        
        return logits, loss
    
    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        images,
        last_img_idx: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        max_new_tokens: int = 50,
        top_k: int = 50,
        top_p: float = 0.9,
        temperature: float = 0.7,
        greedy: bool = False,
    ):
        """
        Generate tokens autoregressively using the dual tower architecture.
        
        Args:
            input_ids (torch.Tensor): Input token IDs of shape (B, T)
            images: Images to process (can be list or tensor)
            last_img_idx (torch.Tensor): Index of the last image token in each sequence
            attention_mask (torch.Tensor, optional): Attention mask of shape (B, T)
            max_new_tokens (int): Number of new tokens to generate
            top_k (int): Top-k filtering parameter for sampling
            top_p (float): Top-p (nucleus) filtering parameter for sampling
            temperature (float): Temperature for sampling (higher = more random)
            greedy (bool): If True, use greedy decoding (argmax), otherwise use sampling
        
        Returns:
            torch.Tensor: Generated token IDs of shape (B, max_new_tokens)
        """
        B = input_ids.size(0)
        device = input_ids.device
        
        # Add batch dimension if needed
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            B = 1
        
        # Handle attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        elif attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        
        # Process left tower to get image KV cache
        _, kv_cache = self.left_tower(
            input_ids=input_ids[:, :last_img_idx + 1],
            images=images,
            attention_mask=attention_mask[:, :last_img_idx + 1],
        )
        
        # Prefill phase: process the FULL sequence with image KV cache
        # The right tower will internally handle replacing image K/V with cached values
        # See modified GQA implementation above for this `LanguageModelGroupedQueryAttention`

        # TODO: below token embedding code is the bypass attempt on self.cfg.lm_use_token, future work may include `if self.cfg.lm_use_tokens` for better conditional in various cases.
        full_embd = self.right_tower.token_embedding(input_ids)
        prompt_output, kv_cache = self.right_tower.forward(
            x=full_embd,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            start_pos=0  # we start prefill from pos 0
        )
        
        last_output = prompt_output[:, -1, :]
        
        # Get logits from the last token output
        if not self.right_tower.lm_use_tokens:
            current_logits = self.right_tower.head(last_output)
        else:
            current_logits = last_output
        
        current_token_start_pos = attention_mask.sum(dim=1) - 1 # Shape (B,)

        newly_generated_ids_list = []
        current_attention_mask = attention_mask.clone()
        
        # Autoregressive generation loop
        for _ in range(max_new_tokens):
            # Sample next token
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
            current_token_start_pos += 1
            
            # Update attention mask
            new_token_mask = torch.ones((B, 1), 
                                        dtype=current_attention_mask.dtype, 
                                        device=device)
            current_attention_mask = torch.cat([current_attention_mask, new_token_mask], dim=1)
            
            # With KV cache: only process the new token
            decode_step_output, kv_cache = self.right_tower.forward(
                x=next_token_embed,
                attention_mask=current_attention_mask,
                kv_cache=kv_cache,
                start_pos=current_token_start_pos
            )
            
            last_token_output = decode_step_output[:, -1, :]
            
            # Apply head to get logits (if model is in embedding mode)
            if not self.right_tower.lm_use_tokens:
                current_logits = self.right_tower.head(last_token_output)
            else:
                current_logits = last_token_output
        
        # Concatenate all generated tokens
        if not newly_generated_ids_list:
            return torch.empty((B, 0), dtype=torch.long, device=device)
        
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
    
    def save_pretrained(self, save_directory: str) -> None:
        os.makedirs(save_directory, exist_ok=True)

        with open(os.path.join(save_directory, "config.json"), "w") as f:
            f.write(json.dumps(asdict(self.cfg), indent=4))

        save_model(self, os.path.join(save_directory, "model.safetensors"))

    @classmethod
    def from_pretrained(
        cls,
        source: str,
        *,
        config_path: str | None = None,
        device: torch.device | str | None = None,
        load_backbone: bool = False,
        **model_kwargs,
    ):
        """Load a DualTowerVLM from a local checkpoint directory/weights file or HF repo."""
        if device is None:
            device = torch.device("cpu")

        if os.path.isdir(source):
            resolved_config_path = config_path or os.path.join(source, "config.json")
            weights_path = os.path.join(source, "model.safetensors")
        elif os.path.isfile(source):
            weights_path = source
            resolved_config_path = config_path or os.path.join(os.path.dirname(source), "config.json")
        else:
            resolved_config_path = hf_hub_download(repo_id=source, filename="config.json")
            weights_path = hf_hub_download(repo_id=source, filename="model.safetensors")

        if not os.path.exists(resolved_config_path):
            raise FileNotFoundError("config.json not found; pass config_path for local weights.")
        if not os.path.exists(weights_path):
            raise FileNotFoundError("model.safetensors not found for the provided source.")

        with open(resolved_config_path, "r") as f:
            cfg_dict = json.load(f)
        cfg = VLMConfig(**cfg_dict)

        model = cls(cfg, load_backbone=load_backbone, **model_kwargs)

        load_safetensors(model, weights_path)
        model = model.to(device)
        return model

    def push_to_hub(self, repo_id: str, private: bool = False) -> None:
        repo_url = create_repo(repo_id=repo_id, private=private, exist_ok=True)
        repo_id = repo_url.repo_id
        print("Created repo: ", repo_url)

        with tempfile.TemporaryDirectory() as save_path:
            self.save_pretrained(save_path)

            with open(os.path.join(save_path, "README.md"), "w") as f:
                f.write(DUAL_TOWER_MODEL_CARD_TEMPLATE.format(repo_id=repo_id))

            return upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=save_path,
                commit_message="Upload DualTowerVLM using push_to_hub",
            )


DUAL_TOWER_MODEL_CARD_TEMPLATE = """---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
library_name: dualtowervlm
license: mit
pipeline_tag: image-text-to-text
tags:
  - vision-language
  - multimodal
  - dual-tower
  - research
---

**DualTowerVLM** is a dual-tower Vision-Language Model (VLM) architecture that processes images and text through separate towers before combining their representations.

For more information, check out the repository.

**Usage:**

```python
from models.dual_tower.dual_tower import DualTowerVLM
from models.config import VLMConfig

cfg = VLMConfig()
model = DualTowerVLM.from_pretrained("{repo_id}")
```
"""
