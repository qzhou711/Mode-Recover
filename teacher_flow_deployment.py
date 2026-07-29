"""Self-contained deployed Flow teacher; never constructs a dataset or training Agent."""
from pathlib import Path
import torch
from omegaconf import OmegaConf
from agents.models.flow_matching.flow_matching import FlowMatching


def build_flow(layers, embed_dim, heads, device="cuda", steps=16):
    network = OmegaConf.create({
        "_target_": "agents.models.diffusion.diffusion_models.DiffusionTransformerNetwork",
        "state_dim": 4, "action_dim": 2, "goal_conditioned": False,
        "goal_seq_len": 10, "obs_seq_len": 5, "embed_pdrob": 0,
        "goal_drop": 0, "attn_pdrop": 0.2, "resid_pdrop": 0.1,
        "embed_dim": embed_dim, "n_layers": layers, "n_heads": heads,
        "device": device, "linear_output": True,
    })
    return FlowMatching(4, 2, network, device=device, solver_steps=steps,
                        solver="heun", time_scale=100.0).to(device)


class DeploymentScaler:
    def __init__(self, metadata, device):
        self.device = device
        for key in ("x_mean", "x_std", "y_mean", "y_std", "y_bounds_tensor"):
            setattr(self, key, metadata[key].to(device))
    def scale_input(self, value):
        value = value.to(self.device)
        return ((value - self.x_mean) / (self.x_std + 1e-12)).float()
    def scale_output(self, value):
        value = value.to(self.device)
        return ((value - self.y_mean) / (self.y_std + 1e-12)).float()
    def inverse_scale_output(self, value):
        return value * (self.y_std + 1e-12) + self.y_mean


def load_deployed_teacher(bundle_dir, device="cuda"):
    bundle_dir = Path(bundle_dir)
    metadata = torch.load(bundle_dir / "deployment_metadata.pt", map_location="cpu")
    model = build_flow(metadata["teacher_layers"], metadata["teacher_embed_dim"],
                       metadata["teacher_heads"], device, metadata["teacher_steps"])
    checkpoint = torch.load(metadata["source_checkpoint"], map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    model.min_action = metadata["y_bounds_tensor"][0].to(device)
    model.max_action = metadata["y_bounds_tensor"][1].to(device)
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return model, DeploymentScaler(metadata, device), metadata
