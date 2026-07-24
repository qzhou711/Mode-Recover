from typing import Optional

import torch

from agents.ddpm_agent import DiffusionAgent


class FlowMatchingAgent(DiffusionAgent):
    """Flow Matching policy using the established d3il agent lifecycle."""

    def __init__(
        self,
        *args,
        state_noise_std: float = 0.0,
        state_noise_prob: float = 0.0,
        **kwargs,
    ):
        if state_noise_std < 0:
            raise ValueError("state_noise_std must be non-negative")
        if not 0.0 <= state_noise_prob <= 1.0:
            raise ValueError("state_noise_prob must be in [0, 1]")
        super().__init__(*args, **kwargs)
        self.eval_model_name = "eval_best_flow.pth"
        self.last_model_name = "last_flow.pth"
        self.state_noise_std = state_noise_std
        self.state_noise_prob = state_noise_prob

    def augment_state(self, state: torch.Tensor) -> torch.Tensor:
        """Perturb actual robot position while preserving desired-position history."""
        if self.state_noise_std == 0.0 or self.state_noise_prob == 0.0:
            return state
        if state.shape[-1] != 4:
            raise ValueError("Avoiding state perturbation expects four state features")
        augmented = state.clone()
        batch_size = state.shape[0]
        noise_shape = (batch_size,) + (1,) * (state.ndim - 2) + (2,)
        noise = torch.randn(noise_shape, device=state.device, dtype=state.dtype)
        apply = torch.rand(
            (batch_size,) + (1,) * (state.ndim - 1),
            device=state.device,
            dtype=state.dtype,
        ) < self.state_noise_prob
        augmented[..., 2:4] += apply * self.state_noise_std * noise
        return augmented

    def train_step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> float:
        self.model.train()
        state = self.augment_state(self.scaler.scale_input(state))
        action = self.scaler.scale_output(action)
        if goal is not None:
            goal = self.scaler.scale_input(goal)
        loss = self.model.loss(action, state, goal)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps += 1
        if self.steps % self.update_ema_every_n_steps == 0:
            self.ema_helper.update(self.model.get_params())
        return loss.detach().item()
