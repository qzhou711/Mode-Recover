import torch
import torch.nn as nn

from agents.models.flow_matching.flow_matching import FlowMatching
from distill_flow_matching_avoiding import (
    conditional_pairwise_geometry,
    normalized_pairwise_geometry,
    repeat_conditions,
    teacher_shortcut_target,
)


class ConstantFlow(FlowMatching):
    def __init__(self, velocity, solver, steps):
        nn.Module.__init__(self)
        self.constant_velocity = velocity
        self.solver = solver
        self.solver_steps = steps
        self.action_dim = velocity.shape[-1]
        self.min_action = None
        self.max_action = None

    def velocity(self, x_t, t, state, goal=None):
        return self.constant_velocity.expand_as(x_t)


def test_linear_path_and_target_velocity():
    noise = torch.tensor([[[1.0, -2.0]]])
    action = torch.tensor([[[5.0, 4.0]]])
    t = torch.tensor([0.25]).reshape(1, 1, 1)
    x_t = (1.0 - t) * noise + t * action
    assert torch.allclose(x_t, torch.tensor([[[2.0, -0.5]]]))
    assert torch.allclose(action - noise, torch.tensor([[[4.0, 6.0]]]))


@torch.no_grad()
def test_euler_and_heun_integrate_constant_velocity_exactly():
    state = torch.zeros(2, 5, 4)
    initial = torch.zeros(2, 5, 2)
    velocity = torch.tensor([[[2.0, -3.0]]])
    for solver in ("euler", "heun"):
        flow = ConstantFlow(velocity, solver, steps=7)
        result = flow.sample(state, initial_noise=initial)
        assert torch.allclose(result, velocity.expand_as(result), atol=1e-6)
        partial = flow.integrate(initial, state, start_time=0.25, end_time=0.75, steps=5)
        assert torch.allclose(partial, 0.5 * velocity.expand_as(partial), atol=1e-6)


def test_velocity_network_receives_gradient():
    prediction = torch.randn(4, 5, 2, requires_grad=True)
    target = torch.randn_like(prediction)
    loss = torch.nn.functional.mse_loss(prediction, target)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_shortcut_target_matches_constant_teacher_flow():
    state = torch.zeros(2, 5, 4)
    noise = torch.zeros(2, 5, 2)
    velocity = torch.tensor([[[1.5, -0.5]]])
    teacher = ConstantFlow(velocity, "heun", steps=16)
    x_t, target = teacher_shortcut_target(teacher, noise, state, 0.5, 16)
    assert torch.allclose(x_t, 0.5 * velocity.expand_as(x_t), atol=1e-6)
    assert torch.allclose(target, velocity.expand_as(target), atol=1e-6)


def test_normalized_pairwise_geometry_is_shape_preserving():
    teacher = torch.randn(8, 5, 2)
    assert normalized_pairwise_geometry(teacher, teacher).item() < 1e-10
    assert normalized_pairwise_geometry(3.0 * teacher, teacher).item() < 1e-10
    collapsed = teacher.mean(dim=0, keepdim=True).expand_as(teacher)
    assert normalized_pairwise_geometry(collapsed, teacher).item() > 0.1


def test_conditional_geometry_only_compares_samples_of_same_state():
    teacher = torch.randn(12, 5, 2)
    assert conditional_pairwise_geometry(teacher, teacher, 4).item() < 1e-10
    collapsed = teacher.reshape(3, 4, 5, 2).mean(dim=1, keepdim=True).expand(-1, 4, -1, -1)
    collapsed = collapsed.reshape_as(teacher)
    assert conditional_pairwise_geometry(collapsed, teacher, 4).item() > 0.1


def test_repeat_conditions_creates_contiguous_same_state_groups():
    state = torch.arange(24).reshape(6, 1, 4)
    action = torch.arange(12).reshape(6, 1, 2)
    repeated_state, repeated_action = repeat_conditions(state, action, 3)
    assert repeated_state.shape[0] == 6
    assert torch.equal(repeated_state[:3], state[:1].expand(3, -1, -1))
    assert torch.equal(repeated_state[3:], state[1:2].expand(3, -1, -1))
    assert torch.equal(repeated_action[:3], action[:1].expand(3, -1, -1))
