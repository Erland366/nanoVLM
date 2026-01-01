import torch


class BaseCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _pad_batch(self, batch, max_length):
        batch["input_ids"] = [torch.nn.functional.pad(ids, (max_length - len(ids), 0), value=self.tokenizer.pad_token_id) for ids in batch["input_ids"]]
        batch["labels"]    = [torch.nn.functional.pad(labels, (max_length - len(labels), 0), value=self.tokenizer.pad_token_id) for labels in batch["labels"]]
        batch["attention_mask"] = [torch.nn.functional.pad(attention_mask, (max_length - len(attention_mask), 0), value=0) for attention_mask in batch["attention_mask"]]

    def prepare_batch(self, batch, max_length=None):
        # 1) Handle empty
        if not batch:
            return {"input_ids": [], "labels": [], "attention_mask": [], "images": []}

        # 2) Drop None rows
        batch = [s for s in batch if s is not None]
        if not batch:
            return {"input_ids": [], "labels": [], "attention_mask": [], "images": []}

        # batch is a list of dicts, each containing "input_ids", "attention_mask", "labels", "images"
        # let's convert it to a dict of lists of tensors
        batch = {k: [item[k] for item in batch] for k in batch[0]}

        if max_length is not None:
            batch = self._discard_samples_that_are_too_long(batch, max_length)

        if len(batch["input_ids"]) == 0:
            return batch

        # Pad samples to max length
        if max_length is not None:
            max_len = max_length
        else:
            max_len = max(map(len, batch["input_ids"]))
        self._pad_batch(batch, max_len) #  dictionaries in Python are mutable and passed by reference

        return {
            "input_ids": torch.stack(batch["input_ids"]),
            "attention_mask": torch.stack(batch["attention_mask"]),
            "images": batch["images"],
            "labels": torch.stack(batch["labels"]),
        }

    def _discard_samples_that_are_too_long(self, batch, max_length):
        filtered = [
            (ids, label, attn, img)
            for ids, label, attn, img in zip(batch["input_ids"], batch["labels"], batch["attention_mask"], batch["images"])
            if len(ids) <= max_length
        ]
        if not filtered:
            return {"input_ids": [], "labels": [], "attention_mask": [], "images": []}
        batch_token_ids, batch_labels, batch_attentions, batch_images = zip(*filtered)
        return {"input_ids": list(batch_token_ids), "labels": list(batch_labels), "attention_mask": list(batch_attentions), "images": list(batch_images)}


class VQACollator(BaseCollator):  # Visual Question Answering Collator
    def __init__(self, tokenizer, max_length):
        self.max_length = max_length
        super().__init__(tokenizer)

    def _pad_batch(self, batch, max_length):  # Reimplementing to use -100 as the pad value for labels, so that it's ignored by the loss
        batch["input_ids"] = [torch.nn.functional.pad(ids, (max_length - len(ids), 0), value=self.tokenizer.pad_token_id) for ids in batch["input_ids"]]
        batch["labels"]    = [torch.nn.functional.pad(labels, (max_length - len(labels), 0), value=-100) for labels in batch["labels"]]
        batch["attention_mask"] = [torch.nn.functional.pad(attention_mask, (max_length - len(attention_mask), 0), value=0) for attention_mask in batch["attention_mask"]]

    def __call__(self, batch):
        batch = self.prepare_batch(batch, max_length=self.max_length)
        return batch


class VQADualCollator(BaseCollator):
    """
    Collator for Dual Tower VLM that performs center padding.
    
    Center padding strategy:
    - Right-pad the left part (up to last image token)
    - Left-pad the right part (after last image token)
    - This ensures all samples have the same split point for dual tower processing
    """
    
    def __init__(self, tokenizer, max_length):
        super().__init__(tokenizer)
        self.max_length = max_length
    
    def _left_pad_batch(self, batch, max_length, pad_value):
        """Left pad sequences to max_length."""
        return [torch.nn.functional.pad(seq, (max_length - len(seq), 0), value=pad_value) for seq in batch]
    
    def _right_pad_batch(self, batch, max_length, pad_value):
        """Right pad sequences to max_length."""
        return [torch.nn.functional.pad(seq, (0, max_length - len(seq)), value=pad_value) for seq in batch]

    def _split_and_center_pad(self, batch, split_points, pad_value):
        """
        Split sequences at split_points and apply center padding.
        Returns padded sequences and the max left length.
        """
        left_parts, right_parts = [], []
        max_left, max_right = 0, 0
        for seq, split in zip(batch, split_points):
            left = seq[:split]
            right = seq[split:]
            left_parts.append(left)
            right_parts.append(right)
            max_left = max(max_left, len(left))
            max_right = max(max_right, len(right))
        
        left_parts_padded = self._right_pad_batch(left_parts, max_left, pad_value)
        right_parts_padded = self._left_pad_batch(right_parts, max_right, pad_value)
        padded_batch = [torch.cat([l, r], dim=0) for l, r in zip(left_parts_padded, right_parts_padded)]
        
        return padded_batch, max_left

    def _center_pad_batch(self, batch, split_points):
        """Apply center padding to all batch components."""
        batch["input_ids"], max_left = self._split_and_center_pad(batch["input_ids"], split_points, self.tokenizer.pad_token_id)
        batch["attention_mask"], _ = self._split_and_center_pad(batch["attention_mask"], split_points, 0)
        batch["labels"], _ = self._split_and_center_pad(batch["labels"], split_points, -100)
        return max_left

    def _discard_samples_that_are_too_long(self, batch, split_points, max_length):
        """Filter out samples that exceed max_length."""
        filtered = []
        filtered_split_points = []
        for i in range(len(batch["input_ids"])):
            seq = batch["input_ids"][i]
            if len(seq) <= max_length:
                filtered.append({k: batch[k][i] for k in batch})
                filtered_split_points.append(split_points[i])
        if not filtered:
            # If everything was filtered out, return an empty dict of lists and split_points.
            return {k: [] for k in batch}, []
        # Flatten dicts-of-lists for each key
        out = {k: [d[k] for d in filtered] for k in batch}
        return out, filtered_split_points

    def prepare_batch(self, batch, max_length=None):
        """
        Prepare a batch with center padding for dual tower processing.
        
        Returns:
            dict with keys:
                - input_ids: [B, T]
                - attention_mask: [B, T]
                - labels: [B, T]
                - img_region_mask: [B, T] boolean mask for image region
                - last_img_idx: scalar int (same for all samples after center padding)
                - images: list of processed images
        """
        # 1) Handle empty batch
        if not batch:
            return {"input_ids": [], "labels": [], "attention_mask": [], "images": [], "last_img_idx": []}
        
        # 2) Drop None rows
        batch = [s for s in batch if s is not None]
        if not batch:
            return {"input_ids": [], "labels": [], "attention_mask": [], "images": [], "last_img_idx": []}
        batch = {k: [item[k] for item in batch] for k in batch[0]}
        split_points = [int(pos.item()) + 1 if hasattr(pos, "item") else int(pos) + 1 for pos in batch["last_image_token_pos"]]

        # 3) Discard samples that are too long
        max_len = self.max_length if max_length is None else max_length
        batch, split_points = self._discard_samples_that_are_too_long(batch, split_points, max_len)

        # 4) Center pad the batch
        if not batch["input_ids"]:
            return {"input_ids": [], "labels": [], "attention_mask": [], "images": [], "last_img_idx": []}

        max_left = self._center_pad_batch(batch, split_points)

        batch_size = len(batch["input_ids"])
        seq_length = batch["input_ids"][0].shape[0]

        # Calculate last_img_idx: it's max_left - 1 (the last position before padding)
        # If max_left is 0, there's no image token, so return -1
        last_img_idx = max_left - 1 if max_left > 0 else -1

        return {
            "input_ids": torch.stack(batch["input_ids"]),
            "attention_mask": torch.stack(batch["attention_mask"]),
            "labels": torch.stack(batch["labels"]),
            "img_region_mask": torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1) < max_left,
            "last_img_idx": last_img_idx,
            "images": batch["images"],
        }

    def __call__(self, batch):
        return self.prepare_batch(batch, max_length=self.max_length)


class COCOCaptionsVanillaCollator(BaseCollator):
    """ Simple left padding basically same with VQACollator """
    def __init__(self, tokenizer, max_length):
        super().__init__(tokenizer)
        self.max_length = max_length

    def _pad_batch(self, batch, max_length):
        batch["input_ids"] = [torch.nn.functional.pad(ids, (max_length - len(ids), 0), value=self.tokenizer.pad_token_id) for ids in batch["input_ids"]]
        batch["labels"] = [torch.nn.functional.pad(labels, (max_length - len(labels), 0), value=-100) for labels in batch["labels"]]
        batch["attention_mask"] = [torch.nn.functional.pad(attention_mask, (max_length - len(attention_mask), 0), value=0) for attention_mask in batch["attention_mask"]]

    def __call__(self, batch):
        batch = self.prepare_batch(batch, max_length=self.max_length)
        return batch
