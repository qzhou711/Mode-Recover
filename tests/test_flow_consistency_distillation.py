import torch

from tests.test_flow_matching import ConstantFlow
from train_flow_consistency_distillation import (
    consistency_endpoint,
    pseudo_huber,
    teacher_pair,
)


def test_teacher_pair_follows_same_ode_trajectory():
    state = torch.zeros(2, 5, 4)
    noise = torch.zeros(2, 5, 2)
    velocity = torch.tensor([[[2.0, -1.0]]])
    teacher = ConstantFlow(velocity, "heun", 16)
    x_start, x_end = teacher_pair(teacher, noise, state, 0.25, 0.5, 16)
    assert torch.allclose(x_start, 0.25 * velocity.expand_as(noise), atol=1e-6)
    assert torch.allclose(x_end, 0.50 * velocity.expand_as(noise), atol=1e-6)


def test_consistency_endpoint_is_constant_along_constant_flow():
    state = torch.zeros(2, 5, 4)
    velocity = torch.tensor([[[2.0, -1.0]]])
    model = ConstantFlow(velocity, "heun", 16)
    for time_value in (0.0, 0.25, 0.75):
        time = torch.full((2,), time_value)
        x_t = time_value * velocity.expand(2, 5, 2)
        endpoint = consistency_endpoint(model, x_t, time, state)
        assert torch.allclose(endpoint, velocity.expand_as(endpoint), atol=1e-6)


def test_consistency_loss_backpropagates_to_online_prediction():
    prediction = torch.randn(4, 5, 2, requires_grad=True)
    target = torch.randn_like(prediction)
    loss = pseudo_huber(prediction, target)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
