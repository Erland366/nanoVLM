import argparse

import torch
from PIL import Image
from data.processors import get_image_processor, get_image_string
from models.dual_tower.dual_tower import DualTowerVLM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a caption with a DualTowerVLM checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a checkpoint directory or weights file.",
        default="patrickamadeus/dualtower-cauldron",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe the image.",
        help="Text prompt to feed the model.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (ignored for greedy decoding).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k filtering for sampling (ignored for greedy decoding).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p filtering for sampling (ignored for greedy decoding).",
    )
    parser.add_argument(
        "--sampling",
        action="store_true",
        help="Use top-k/top-p sampling instead of greedy decoding.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config.json path if --checkpoint points to weights.",
    )
    return parser.parse_args()


def pick_device(device_arg):
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_inputs(tokenizer, image_processor, cfg, image_path, prompt, device):
    image = Image.open(image_path).convert("RGB")
    processed_image, split_ratio = image_processor(image)

    if (
        not hasattr(tokenizer, "global_image_token")
        and split_ratio[0] * split_ratio[1] == len(processed_image) - 1
    ):
        processed_image = processed_image[1:]

    image_string = get_image_string(tokenizer, [split_ratio], cfg.mp_image_token_length)
    messages = [{"role": "user", "content": image_string + prompt}]
    conv = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_special_tokens=False,
        add_generation_prompt=True,
        return_dict=True,
    )

    input_ids = torch.tensor(conv["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.tensor(conv["attention_mask"], dtype=torch.long, device=device).unsqueeze(0)

    image_token_id = tokenizer.encode(tokenizer.image_token, add_special_tokens=False)[0]
    image_positions = (input_ids[0] == image_token_id).nonzero(as_tuple=False)
    if image_positions.numel() == 0:
        raise ValueError("No image token found in the prompt; cannot compute last_img_idx.")
    last_img_idx = int(image_positions[-1].item())

    return input_ids, attention_mask, [processed_image], last_img_idx


def main():
    args = parse_args()
    device = pick_device(args.device)
    print(f"Using device: {device}")
    model = DualTowerVLM.from_pretrained(
        args.checkpoint,
        config_path=args.config,
        device=device,
        load_backbone=False,
        freeze_left_vision=True,
        freeze_left_projector=True,
        freeze_left_decoder=True,
        freeze_right_decoder=True,
    )
    model.eval()
    cfg = model.cfg
    tokenizer = model.tokenizer
    resize_to_max_side_len = getattr(cfg, "resize_to_max_side_len", False)
    image_processor = get_image_processor(
        cfg.max_img_size,
        cfg.vit_img_size,
        resize_to_max_side_len,
    )

    input_ids, attention_mask, images, last_img_idx = build_inputs(
        tokenizer, image_processor, cfg, args.image, args.prompt, device
    )

    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=input_ids,
            images=images,
            attention_mask=attention_mask,
            last_img_idx=last_img_idx,
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            greedy=not args.sampling,
        )

    caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(caption)


if __name__ == "__main__":
    main()
