import torch

from data.advanced_datasets import ConstantLengthDataset


class _DummyTokenizer:
    pad_token_id = 0


class _DummyDataset:
    mp_image_token_length = 4
    tokenizer = _DummyTokenizer()


def test_pack_one_group_builds_document_ids():
    ds = ConstantLengthDataset(
        _DummyDataset(),
        infinite=False,
        max_sample_length=128,
        seq_length=128,
        num_of_sequences=1,
        queue_size=1,
        max_images_per_example=4,
        max_images_per_knapsack=18,
        pack_sequences=True,
    )

    sample0 = {
        "input_ids": torch.tensor([11, 12, 13]),
        "labels": torch.tensor([-100, 12, 13]),
        "attention_mask": torch.tensor([1, 1, 1]),
        "images": [],
        "document_ids": torch.zeros(3, dtype=torch.long),
    }
    sample1 = {
        "input_ids": torch.tensor([21, 22, 23, 24]),
        "labels": torch.tensor([-100, 22, 23, 24]),
        "attention_mask": torch.tensor([1, 1, 1, 1]),
        "images": [],
        "document_ids": torch.zeros(4, dtype=torch.long),
    }

    input_ids, labels, attention_mask, images, document_ids = ds._pack_one_group(
        [0, 1], [sample0, sample1], max_len=128
    )

    assert input_ids.shape == (7,)
    assert labels.shape == (7,)
    assert attention_mask.shape == (7,)
    assert isinstance(images, list)
    assert document_ids.shape == (7,)
    assert torch.equal(document_ids, torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.long))

