from typing import Optional

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class FlowMatching(nn.Module):
    """Conditional linear flow matching for action-sequence policies."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        model: DictConfig,
        device: str = "cuda",
        solver_steps: int = 16,
        solver: str = "heun",
        time_scale: float = 100.0,
    ):
        super().__init__()
        if solver_steps < 1:
            raise ValueError("solver_steps must be positive")
        if solver not in {"euler", "heun"}:
            raise ValueError(f"unknown solver: {solver}")
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = hydra.utils.instantiate(model)
        self.solver_steps = solver_steps
        self.solver = solver
        self.time_scale = time_scale
        self.min_action = None
        self.max_action = None

    def get_params(self):
        return self.model.get_params()

    def velocity(self, x_t, t, state, goal=None):
        return self.model(x_t, t * self.time_scale, state, goal)

    def loss(self, action, state, goal=None):
        noise = torch.randn_like(action)
        t = torch.rand(action.shape[0], device=action.device, dtype=action.dtype)
        t_view = t.reshape(action.shape[0], *((1,) * (action.ndim - 1)))
        x_t = (1.0 - t_view) * noise + t_view * action
        target_velocity = action - noise
        predicted_velocity = self.velocity(x_t, t, state, goal)
        return F.mse_loss(predicted_velocity, target_velocity)

    @torch.no_grad()
    def integrate(self, x, state, goal=None, start_time: float = 0.0, end_time: float = 1.0, steps: Optional[int] = None):
        """Integrate the learned ODE over an arbitrary time interval."""
        if not 0.0 <= start_time < end_time <= 1.0:
            raise ValueError("expected 0 <= start_time < end_time <= 1")
        batch_size = x.shape[0]
        n_steps = steps or self.solver_steps
        dt = (end_time - start_time) / n_steps
        for index in range(n_steps):
            current_time = start_time + index * dt
            t = torch.full((batch_size,), current_time, device=x.device, dtype=x.dtype)
            v = self.velocity(x, t, state, goal)
            if self.solver == "euler" or index + 1 == n_steps:
                x = x + dt * v
            else:
                x_predictor = x + dt * v
                t_next = torch.full((batch_size,), current_time + dt, device=x.device, dtype=x.dtype)
                v_next = self.velocity(x_predictor, t_next, state, goal)
                x = x + 0.5 * dt * (v + v_next)
        return x

    @torch.no_grad()
    def sample(self, state, goal=None, initial_noise: Optional[torch.Tensor] = None, steps: Optional[int] = None):
        batch_size, sequence_length = state.shape[:2]
        if initial_noise is None:
            x = torch.randn(batch_size, sequence_length, self.action_dim, device=state.device, dtype=state.dtype)
        else:
            x = initial_noise.clone().to(device=state.device, dtype=state.dtype)
        x = self.integrate(x, state, goal, 0.0, 1.0, steps)
        if self.min_action is not None and self.max_action is not None:
            x = x.clamp(self.min_action, self.max_action)
        return x

    def forward(self, state, goal=None):
        return self.sample(state, goal)
