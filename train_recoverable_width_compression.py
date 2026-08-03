"""Recoverability-guided FM-3x72-16 -> FM-3x48-16 width compression."""
import argparse, json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from agents.models.diffusion.ema import ExponentialMovingAverage
from teacher_flow_deployment import build_flow
from train_flow_progressive_compression import initialize_student, selection_basis
from train_teacher_generated_flow_v2 import activation_matrix, differentiable_integrate, save_ema


def canonical_signs(basis):
    rows = torch.argmax(basis.abs(), dim=0)
    signs = torch.sign(basis[rows, torch.arange(basis.shape[1])])
    return basis * torch.where(signs == 0, torch.ones_like(signs), signs)


def make_basis(method, activations, teacher, seed):
    if method == "random_orthogonal":
        generator = torch.Generator().manual_seed(seed)
        q, _ = torch.linalg.qr(torch.randn(72, 48, generator=generator), mode="reduced")
        return canonical_signs(q), {"method": method}
    if method == "activation_coordinate":
        basis, meta = selection_basis(activations, 3, 3)
        return basis, {"method": method, **meta}
    centered = activations - activations.mean(0, keepdim=True)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    if method == "pca":
        metric = covariance
    elif method == "qkv_sensitivity":
        sensitivity = torch.zeros_like(covariance)
        state = teacher.state_dict()
        for layer in range(3):
            for name in ("query", "key", "value"):
                weight = state[f"model.blocks.{layer}.attn.{name}.weight"].detach().cpu()
                sensitivity += weight.T @ weight
        covariance = covariance / covariance.trace().clamp_min(1e-12)
        sensitivity = sensitivity / sensitivity.trace().clamp_min(1e-12)
        metric = covariance + sensitivity
    else:
        raise ValueError(method)
    values, vectors = torch.linalg.eigh(metric)
    order = torch.argsort(values, descending=True)[:48]
    basis = canonical_signs(vectors[:, order])
    explained = float(values[order].clamp_min(0).sum() / values.clamp_min(0).sum().clamp_min(1e-12))
    return basis, {"method": method, "metric_explained_fraction": explained}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle-dir", type=Path, required=True)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--buffer", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--method", choices=["random_orthogonal", "pca", "activation_coordinate", "qkv_sensitivity"], required=True)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-batches", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--endpoint-weight", type=float, default=0.03)
    p.add_argument("--save-epochs", default="50,100,250,500")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    meta = torch.load(a.bundle_dir / "deployment_metadata.pt", map_location="cpu")
    teacher = build_flow(3, 72, 4, "cuda", 16)
    checkpoint = a.teacher if a.teacher.is_file() else a.teacher / "eval_best_flow.pth"
    teacher.load_state_dict(torch.load(checkpoint, map_location="cuda"), strict=True)
    teacher.min_action = meta["y_bounds_tensor"][0].cuda(); teacher.max_action = meta["y_bounds_tensor"][1].cuda()
    teacher.eval()
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    data = torch.load(a.buffer, map_location="cpu")
    assert data["metadata"]["uses_original_demonstrations"] is False
    assert data["metadata"]["uses_expert_actions"] is False
    # Buffers may use arbitrary global episode ids. The episode-level tensors
    # are stored in sorted shard/episode order, so map ids to their dense rank
    # instead of assuming they start at zero.
    episode_ids = data["episode_ids"].long()
    unique_episode_ids = torch.unique(episode_ids, sorted=True)
    if len(unique_episode_ids) != len(data["successes"]):
        raise ValueError("episode-level successes do not align with unique sample episode ids")
    dense_episode_ids = torch.searchsorted(unique_episode_ids, episode_ids)
    if not torch.equal(unique_episode_ids[dense_episode_ids], episode_ids):
        raise ValueError("failed to map global episode ids to dense episode indices")
    keep = data["successes"].bool()[dense_episode_ids]
    states, noises = data["states"].float()[keep], data["noises"].float()[keep]
    endpoints = data["teacher_endpoints"].float()[keep]
    loader = DataLoader(TensorDataset(states, noises, endpoints), batch_size=a.batch_size,
                        shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(a.seed))
    activations = activation_matrix(teacher, states, noises, 8, a.batch_size)
    basis, basis_meta = make_basis(a.method, activations, teacher, a.seed)
    orthogonality_error = float((basis.T @ basis - torch.eye(48)).abs().max())
    if orthogonality_error > 1e-4: raise RuntimeError(f"non-orthonormal width basis: {orthogonality_error}")
    student = build_flow(3, 48, 3, "cuda", 16).train()
    initialize_student(teacher, student, basis, [0, 1, 2])
    torch.save(student.state_dict(), a.output_dir / "initial_flow.pth")
    optimizer = torch.optim.AdamW(student.get_params(), lr=a.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, a.epochs, eta_min=1e-6)
    ema = ExponentialMovingAverage(student.get_params(), 0.995, "cuda")
    milestones = {int(x) for x in a.save_epochs.split(",")}; history=[]; best=math.inf
    for epoch in range(1, a.epochs + 1):
        vv, ev = [], []
        for bi, (state, noise, endpoint) in enumerate(loader):
            if bi >= a.max_batches: break
            state, noise, endpoint = state.cuda(), noise.cuda(), endpoint.cuda()
            t = float(torch.rand(()).clamp(.02, .98)); tv = torch.full((len(state),), t, device="cuda")
            with torch.no_grad():
                x_t = teacher.integrate(noise, state, start_time=0., end_time=t, steps=max(1, round(16*t)))
                target = teacher.velocity(x_t, tv, state)
            velocity = F.mse_loss(student.velocity(x_t, tv, state), target)
            endpoint_loss = F.mse_loss(differentiable_integrate(student, noise, state, steps=16), endpoint)
            loss = velocity + a.endpoint_weight * endpoint_loss
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(student.get_params(), 1.0); optimizer.step(); ema.update(student.get_params())
            vv.append(float(velocity.detach())); ev.append(float(endpoint_loss.detach()))
        scheduler.step(); score=float(np.mean(vv)+a.endpoint_weight*np.mean(ev))
        row={"epoch":epoch,"selection_loss":score,"velocity_loss":float(np.mean(vv)),"endpoint_loss":float(np.mean(ev))};history.append(row)
        if score < best: best=score; save_ema(student,ema,a.output_dir/"structure_best_flow.pth")
        if epoch in milestones: save_ema(student,ema,a.output_dir/f"pretrain_epoch_{epoch:04d}.pth")
        if epoch%25==0: print(json.dumps(row),flush=True)
    (a.output_dir/"metrics.json").write_text(json.dumps({"experiment":"recoverability-guided width compression","teacher":"FM-3x72-16 keep013","student":"FM-3x48-16","method":a.method,"basis":basis_meta,"orthogonality_error":orthogonality_error,"teacher_buffer_samples":len(states),"uses_original_demonstrations":False,"uses_expert_actions":False,"history":history},indent=2))


if __name__ == "__main__": main()
