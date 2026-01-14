from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from tqdm import tqdm
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms

from data.processors import get_image_processor, get_image_string
from models.dual_tower.dual_tower import DualTowerVLM


class DualTowerWrapper(lmms):
    def __init__(
        self,
        model: str | DualTowerVLM = "patrickamadeus/dualtower-cauldron",
        device: str = "cuda",
        batch_size: int = 32,
        config_path: Optional[str] = None,
        load_backbone: bool = False,
        max_length: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        if isinstance(model, str):
            self.model = DualTowerVLM.from_pretrained(
                model,
                config_path=config_path,
                device=device,
                load_backbone=load_backbone,
                freeze_left_vision=True,
                freeze_left_projector=True,
                freeze_left_decoder=True,
                freeze_right_decoder=True,
            )
        else:
            self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.batch_size = batch_size

        if dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            self._rank = 0
            self._world_size = 1

        self.tokenizer = self.model.tokenizer
        resize_to_max_side_len = getattr(self.model.cfg, "resize_to_max_side_len", False)
        self.image_processor = get_image_processor(
            self.model.cfg.max_img_size,
            self.model.cfg.vit_img_size,
            resize_to_max_side_len,
        )
        self._max_length = (
            max_length
            if max_length is not None
            else getattr(self.model.cfg, "lm_max_length", self.model.cfg.lm_max_position_embeddings)
        )

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _prepare_visual_input(self, visual_list: List[Image.Image]) -> Tuple[Optional[list], Optional[list]]:
        if not visual_list or visual_list[0] is None:
            return None, None

        images = []
        split_ratios = []
        for visual in visual_list:
            if isinstance(visual, Image.Image):
                image = visual
            elif isinstance(visual, str):
                image = Image.open(visual).convert("RGB")
            elif isinstance(visual, np.ndarray):
                image = Image.fromarray(visual)
            else:
                raise ValueError(f"Unsupported visual type: {type(visual)}.")

            processed_images, split_ratio = self.image_processor(image)
            if (
                not hasattr(self.tokenizer, "global_image_token")
                and split_ratio[0] * split_ratio[1] == len(processed_images) - 1
            ):
                processed_images = processed_images[1:]

            images.append(processed_images)
            split_ratios.append(split_ratio)

        if images:
            return images, split_ratios
        return None, None

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for DualTowerVLM")

    def flatten(self, input_list):
        new_list = []
        for sublist in input_list:
            if sublist is None:
                new_list.append(None)
            else:
                for item in sublist:
                    new_list.append(item)
        return new_list

    def get_benchmark_formatting(self, task_name: str) -> dict:
        benchmark_formats = {
            ("ai2d", "mmstar", "seedbench", "scienceqa"): {
                "text_replacements": {
                    "\nOptions:": "\nChoices:",
                    "\nA. ": "\nChoices:\nA. ",
                    "Please select the correct answer from the options above.": "Answer with the letter.",
                    "Answer with the option's letter from the given choices directly": "Answer with the letter directly",
                },
                "assistant_prefix": "Answer:",
                "user_prefix": "",
                "user_suffix": "",
            },
            ("docvqa_val", "docvqa_test"): {
                "text_replacements": {},
                "assistant_prefix": "",
                "user_prefix": (
                    "Give a short and terse answer to the following question. "
                    "Do not paraphrase or reformat the text you see in the image. "
                    "Do not include any full stops. Just give the answer without additional explanation. Question: "
                ),
                "user_suffix": "",
            },
            "chartvqa": {
                "text_replacements": {},
                "assistant_prefix": "",
                "user_prefix": (
                    "For the question below, follow the following instructions:\n"
                    "-The answer should contain as few words as possible.\n"
                    "-Don't paraphrase or reformat the text you see in the image.\n"
                    "-Answer a binary question with Yes or No.\n"
                    "-When asked to give a numerical value, provide a number like 2 instead of Two.\n"
                    "-If the final answer has two or more items, provide it in the list format like [1, 2].\n"
                    "-When asked to give a ratio, give out the decimal value like 0.25 instead of 1:4.\n"
                    "-When asked to give a percentage, give out the whole value like 17 instead of decimal like 0.17%.\n"
                    "-Don't include any units in the answer.\n"
                    "-Do not include any full stops at the end of the answer.\n"
                    "-Try to include the full label from the graph when asked about an entity.\n"
                    "Question: "
                ),
                "user_suffix": "",
            },
            ("textvqa_val", "textvqa_test"): {
                "text_replacements": {},
                "assistant_prefix": "",
                "user_prefix": (
                    "Answer the following question about the image using as few words as possible. "
                    "Follow these additional instructions:\n"
                    "-Always answer a binary question with Yes or No.\n"
                    "-When asked what time it is, reply with the time seen in the image.\n"
                    "-Do not put any full stops at the end of the answer.\n"
                    "-Do not put quotation marks around the answer.\n"
                    "-An answer with one or two words is favorable.\n"
                    "-Do not apply common sense knowledge. The answer can be found in the image.\n"
                    "Question: "
                ),
                "user_suffix": "",
            },
            ("mmmu_val", "mmmu_test"): {
                "text_replacements": {
                    "Question:": "",
                    "Answer with the option's letter from the given choices directly.": "Answer with the letter directly.",
                    "\nA. ": "\nChoices:\nA. ",
                },
                "assistant_prefix": "Answer:",
                "user_prefix": "",
                "user_suffix": "",
            },
            ("infovqa_val", "mme", "ocrbench"): {
                "text_replacements": {},
                "assistant_prefix": "",
                "user_prefix": "",
                "user_suffix": "\nGive a very brief answer.",
            },
        }

        if task_name in benchmark_formats:
            return benchmark_formats[task_name]
        for key, formatting in benchmark_formats.items():
            if isinstance(key, (list, tuple)) and task_name in key:
                return formatting
        return {"text_replacements": {}, "assistant_prefix": "", "user_prefix": "", "user_suffix": ""}

    def apply_benchmark_formatting(self, context_str: str, prompt: str, task_name: str) -> Tuple[str, str]:
        formatting = self.get_benchmark_formatting(task_name)

        if formatting["user_prefix"]:
            context_str = formatting["user_prefix"] + context_str
        for old_text, new_text in formatting["text_replacements"].items():
            context_str = context_str.replace(old_text, new_text)
        if formatting["user_suffix"]:
            context_str = context_str + formatting["user_suffix"]
        if formatting["assistant_prefix"]:
            prompt = prompt + formatting["assistant_prefix"]

        return context_str, prompt

    def _left_pad(self, seq: torch.Tensor, max_len: int, pad_value: int) -> torch.Tensor:
        if seq.numel() >= max_len:
            return seq
        return torch.nn.functional.pad(seq, (max_len - seq.numel(), 0), value=pad_value)

    def _right_pad(self, seq: torch.Tensor, max_len: int, pad_value: int) -> torch.Tensor:
        if seq.numel() >= max_len:
            return seq
        return torch.nn.functional.pad(seq, (0, max_len - seq.numel()), value=pad_value)

    def _center_pad_batch(
        self,
        input_ids_list: List[torch.Tensor],
        attention_mask_list: List[torch.Tensor],
        split_points: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        left_ids, right_ids = [], []
        left_masks, right_masks = [], []
        max_left, max_right = 0, 0
        for ids, mask, split in zip(input_ids_list, attention_mask_list, split_points):
            left = ids[:split]
            right = ids[split:]
            left_mask = mask[:split]
            right_mask = mask[split:]
            left_ids.append(left)
            right_ids.append(right)
            left_masks.append(left_mask)
            right_masks.append(right_mask)
            max_left = max(max_left, left.numel())
            max_right = max(max_right, right.numel())

        left_ids_padded = [self._right_pad(seq, max_left, self.tokenizer.pad_token_id) for seq in left_ids]
        right_ids_padded = [self._left_pad(seq, max_right, self.tokenizer.pad_token_id) for seq in right_ids]
        left_masks_padded = [self._right_pad(seq, max_left, 0) for seq in left_masks]
        right_masks_padded = [self._left_pad(seq, max_right, 0) for seq in right_masks]

        input_ids = [torch.cat([l, r], dim=0) for l, r in zip(left_ids_padded, right_ids_padded)]
        attention_mask = [torch.cat([l, r], dim=0) for l, r in zip(left_masks_padded, right_masks_padded)]

        return torch.stack(input_ids), torch.stack(attention_mask), max_left - 1

    def _find_last_image_token_pos(self, input_ids: torch.Tensor) -> int:
        image_token_id = self.tokenizer.encode(self.tokenizer.image_token, add_special_tokens=False)[0]
        positions = (input_ids == image_token_id).nonzero(as_tuple=False)
        if positions.numel() == 0:
            raise ValueError("No image token found in the prompt.")
        return int(positions[-1].item())

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            try:
                contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
                visuals = [dtv(self.task_dict[t][s][i]) for dtv, i, t, s in zip(doc_to_visual, doc_id, task, split)]
                images, split_ratios = self._prepare_visual_input(self.flatten(visuals))
            except Exception as exc:
                print(f"Error preparing visual input: {exc}")
                if len(contexts) > 0:
                    pbar.update(len(contexts))
                    res.extend([""] * len(contexts))
                continue

            messages = []
            split_idx = 0
            for i in range(len(contexts)):
                current_context_str = contexts[i]
                current_context_str, _ = self.apply_benchmark_formatting(current_context_str, "", task[i])

                if visuals[i] is None:
                    image_count = 0
                else:
                    image_count = len(visuals[i])
                image_string = ""
                for _ in range(image_count):
                    image_string += get_image_string(
                        self.tokenizer, [split_ratios[split_idx]], self.model.cfg.mp_image_token_length
                    )
                    split_idx += 1

                prompt_content = image_string + current_context_str
                messages.append([{"role": "user", "content": prompt_content}])

            prompts = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for i in range(len(prompts)):
                _, prompts[i] = self.apply_benchmark_formatting("", prompts[i], task[i])

            tokenized = self.tokenizer(
                prompts,
                return_attention_mask=True,
                padding=False,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )
            input_ids_list = [torch.tensor(ids, dtype=torch.long) for ids in tokenized["input_ids"]]
            attention_mask_list = [torch.tensor(mask, dtype=torch.long) for mask in tokenized["attention_mask"]]

            split_points = [self._find_last_image_token_pos(ids) + 1 for ids in input_ids_list]
            input_ids, attention_mask, last_img_idx = self._center_pad_batch(
                input_ids_list, attention_mask_list, split_points
            )
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            current_gen_kwargs = all_gen_kwargs[0] if all_gen_kwargs else {}
            max_new_tokens = current_gen_kwargs.get("max_new_tokens", 50)
            temperature = current_gen_kwargs.get("temperature", 0.0)
            top_p = current_gen_kwargs.get("top_p", 1.0)
            top_k = current_gen_kwargs.get("top_k", 50)
            greedy = current_gen_kwargs.get("do_sample", False) is False or temperature == 0.0

            generated_ids_batch = self.model.generate(
                input_ids=input_ids,
                images=images,
                attention_mask=attention_mask,
                last_img_idx=last_img_idx,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                greedy=greedy,
            )

            generated_texts = self.tokenizer.batch_decode(generated_ids_batch, skip_special_tokens=True)
            res.extend(generated_texts)
            pbar.update(len(contexts))

        pbar.close()
        return re_ords.get_original(res)

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for DualTowerVLM")

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size_per_gpu(self):
        return self.batch_size
