import random

import numpy
import torch

from utils.checkpointing import (
    capture_rng_state,
    load_model_optimizer_state,
    load_trainer_state,
    restore_rng_state,
    save_model_optimizer_state,
    save_trainer_state,
)


def test_rng_restore_roundtrip():
    random.seed(123)
    numpy.random.seed(123)
    torch.manual_seed(123)

    rng_state = capture_rng_state()

    python_next = random.random()
    numpy_next = numpy.random.rand(3)
    torch_next = torch.rand(3)

    restore_rng_state(rng_state)

    assert random.random() == python_next
    assert numpy.allclose(numpy.random.rand(3), numpy_next)
    assert torch.allclose(torch.rand(3), torch_next)


def test_save_load_model_optimizer(tmp_path):
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randn(8, 4)
    y = model(x).sum()
    y.backward()
    optimizer.step()

    ckpt_dir = tmp_path / "ckpt"
    save_model_optimizer_state(str(ckpt_dir), model, optimizer, use_dcp=False, strict=True)

    model2 = torch.nn.Linear(4, 2)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    load_model_optimizer_state(str(ckpt_dir), model2, optimizer2, use_dcp=False, strict=True)

    for key, value in model.state_dict().items():
        assert torch.allclose(value, model2.state_dict()[key])

    state1 = optimizer.state_dict()
    state2 = optimizer2.state_dict()
    assert state1.keys() == state2.keys()
    assert state1["param_groups"] == state2["param_groups"]
    assert state1["state"].keys() == state2["state"].keys()
    for param_id, param_state in state1["state"].items():
        for k, v in param_state.items():
            v2 = state2["state"][param_id][k]
            if torch.is_tensor(v):
                assert torch.allclose(v, v2)
            else:
                assert v == v2


def test_save_load_trainer_state(tmp_path):
    ckpt_dir = tmp_path / "trainer_state"
    trainer_state = {
        "global_step": 5,
        "epoch": 1,
        "micro_step_in_epoch": 8,
        "run_name": "test_run",
    }
    save_trainer_state(str(ckpt_dir), trainer_state)

    loaded = load_trainer_state(str(ckpt_dir), strict=True)
    assert loaded["global_step"] == 5
    assert loaded["epoch"] == 1
    assert loaded["micro_step_in_epoch"] == 8
    assert loaded["run_name"] == "test_run"
