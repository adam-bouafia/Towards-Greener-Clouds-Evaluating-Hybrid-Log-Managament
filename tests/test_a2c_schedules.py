import numpy as np
import importlib.util, pytest

if importlib.util.find_spec("torch") is None:
    pytest.skip("torch not installed; skipping A2C schedule tests", allow_module_level=True)

from src.train import LRScheduleCallback, EntropyAnnealCallback

class DummyModel:
    class DummyPolicy:
        def __init__(self):
            import torch
            self.optimizer = torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=1e-3)
    def __init__(self):
        self.policy = DummyModel.DummyPolicy()
        self.ent_coef = 0.1


def test_lr_schedule():
    cb = LRScheduleCallback(total_timesteps=100, initial_lr=0.01)
    cb.model = DummyModel()
    # simulate steps
    for step in range(101):
        cb.num_timesteps = step
        cb._on_step()
    assert abs(cb.current_lr - 0.0) < 1e-9


def test_entropy_anneal():
    cb = EntropyAnnealCallback(start_ent=0.1, target_ent=0.01, anneal_steps=50)
    cb.model = DummyModel()
    for step in range(51):
        cb.num_timesteps = step
        cb._on_step()
    assert abs(cb.current - 0.01) < 1e-9
