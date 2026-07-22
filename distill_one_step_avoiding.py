"""Distill a 2-step Avoiding DDIM policy into a one-step policy.

Two objectives are supported:

* progressive: analytically derive the epsilon prediction whose single DDIM
  update reaches the frozen teacher's two-step endpoint.
* ctm: local consistency distillation between adjacent points on a frozen
  teacher DDIM trajectory, using an EMA target student and a DSM anchor.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from hydra import compose, initialize
from tqdm import trange

from agents.models.diffusion.ema import ExponentialMovingAverage
from agents.models.diffusion.utils import extract


def make_teacher(weights_dir: Path, batch_size: int):
    with initialize(config_path="configs"):
        cfg = compose(
            config_name="avoiding_config",
            overrides=[
                "agents=ddpm_transformer_agent",
                "window_size=5",
                "epoch=1",
                f"train_batch_size={batch_size}",
                "simulation.render=False",
            ],
        )
    agent = hydra.utils.instantiate(cfg.agents)
    agent.load_pretrained_model(str(weights_dir), sv_name="eval_best_ddpm.pth")
    return agent


def two_step_teacher_endpoint(teacher, state, initial_noise):
    batch_size = initial_noise.shape[0]
    t_high = torch.full(
        (batch_size,), teacher.n_timesteps - 1,
        device=initial_noise.device, dtype=torch.long,
    )
    t_low = torch.zeros(batch_size, device=initial_noise.device, dtype=torch.long)
    x_low = teacher.ddim_sample(initial_noise, t_high, state, None, prev_t=0)
    x_zero = teacher.ddim_sample(x_low, t_low, state, None, prev_t=-1)
    return x_zero.clamp(teacher.min_action, teacher.max_action)


def epsilon_for_endpoint(model, x_t, t, endpoint):
    """Invert the deterministic final DDIM update for an epsilon model."""
    alpha_t = extract(model.alphas_cumprod, t, x_t.shape).to(x_t.dtype)
    endpoint = endpoint.to(x_t.dtype)
    return (x_t - torch.sqrt(alpha_t) * endpoint) / torch.sqrt(1.0 - alpha_t)


def predict_clean(model, x_t, t, state):
    epsilon = model.model(x_t, t, state, None)
    sqrt_recip = extract(model.sqrt_recip_alphas_cumprod, t, x_t.shape).to(x_t.dtype)
    sqrt_recipm1 = extract(model.sqrt_recipm1_alphas_cumprod, t, x_t.shape).to(x_t.dtype)
    return sqrt_recip * x_t - sqrt_recipm1 * epsilon


def consistency_output(model, x_t, t, state, sigma_data=0.5):
    """Boundary-conditioned clean-action prediction (f(x, 0) ~= x)."""
    alpha = extract(model.alphas_cumprod, t, x_t.shape).to(x_t.dtype)
    sigma = torch.sqrt((1.0 - alpha).clamp_min(1e-8) / alpha.clamp_min(1e-8))
    c_skip = sigma_data ** 2 / (sigma.square() + sigma_data ** 2)
    clean = predict_clean(model, x_t, t, state)
    return c_skip * x_t + (1.0 - c_skip) * clean


def pseudo_huber(prediction, target, delta=0.03, weight=None):
    error = prediction - target
    loss = torch.sqrt(error.square() + delta ** 2) - delta
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def dsm_loss(model, action, state):
    batch_size = action.shape[0]
    t = torch.randint(0, model.n_timesteps, (batch_size,), device=action.device)
    noise = torch.randn_like(action)
    x_noisy = model.q_sample(action, t, noise).to(action.dtype)
    epsilon = model.model(x_noisy, t, state, None)
    return F.mse_loss(epsilon, noise)


@torch.no_grad()
def update_target(target, online, decay):
    for target_parameter, online_parameter in zip(
        target.parameters(), online.parameters()
    ):
        target_parameter.mul_(decay).add_(online_parameter, alpha=1.0 - decay)


def save_ema_model(student, ema, output_dir: Path, name: str):
    ema.store(student.get_params())
    ema.copy_to(student.get_params())
    torch.save(student.state_dict(), output_dir / name)
    ema.restore(student.get_params())


def train_progressive(args, teacher_agent, teacher, student):
    optimizer = torch.optim.Adam(student.get_params(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    ema = ExponentialMovingAverage(
        student.get_params(), args.ema_decay, teacher_agent.device
    )
    best_loss, best_epoch, history = math.inf, -1, []

    for epoch in trange(args.epochs, desc="strict progressive DDIM 2-to-1"):
        totals, distillation_losses, diffusion_losses = [], [], []
        for batch_index, (state, action, _) in enumerate(
            teacher_agent.train_dataloader
        ):
            if args.max_batches_per_epoch and batch_index >= args.max_batches_per_epoch:
                break
            state = teacher_agent.scaler.scale_input(state).float()
            action = teacher_agent.scaler.scale_output(action).float()
            x_t = torch.randn_like(action)
            t = torch.full(
                (x_t.shape[0],), teacher.n_timesteps - 1,
                device=x_t.device, dtype=torch.long,
            )

            with torch.no_grad():
                endpoint = two_step_teacher_endpoint(teacher, state, x_t)
                epsilon_target = epsilon_for_endpoint(teacher, x_t, t, endpoint)

            epsilon_prediction = student.model(x_t, t, state, None)
            distillation_loss = F.mse_loss(
                epsilon_prediction, epsilon_target
            )
            diffusion_loss = dsm_loss(student, action, state)
            loss = distillation_loss + args.diffusion_weight * diffusion_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.get_params(), 1.0)
            optimizer.step()
            ema.update(student.get_params())

            totals.append(loss.detach().item())
            distillation_losses.append(distillation_loss.detach().item())
            diffusion_losses.append(diffusion_loss.detach().item())

        scheduler.step()
        record = {
            "epoch": epoch,
            "loss": float(np.mean(totals)),
            "progressive_loss": float(np.mean(distillation_losses)),
            "diffusion_loss": float(np.mean(diffusion_losses)),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        if record["loss"] < best_loss:
            best_loss, best_epoch = record["loss"], epoch
            save_ema_model(student, ema, args.output_dir, "eval_best_ddpm.pth")
        if epoch % 10 == 0 or epoch + 1 == args.epochs:
            print(json.dumps(record), flush=True)

    save_ema_model(student, ema, args.output_dir, "last_ddpm.pth")
    return best_epoch, best_loss, history


def train_ctm(args, teacher_agent, teacher, student):
    target = copy.deepcopy(student)
    target.eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.Adam(student.get_params(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    best_loss, best_epoch, history = math.inf, -1, []

    for epoch in trange(args.epochs, desc="CTM-style + DSM 2-to-1"):
        totals, consistency_losses, diffusion_losses = [], [], []
        for batch_index, (state, action, _) in enumerate(
            teacher_agent.train_dataloader
        ):
            if args.max_batches_per_epoch and batch_index >= args.max_batches_per_epoch:
                break
            state = teacher_agent.scaler.scale_input(state).float()
            action = teacher_agent.scaler.scale_output(action).float()
            noise = torch.randn_like(action)

            timestep = int(
                torch.randint(1, teacher.n_timesteps, (1,)).item()
            )
            t = torch.full(
                (action.shape[0],), timestep,
                device=action.device, dtype=torch.long,
            )
            s = torch.full_like(t, timestep - 1)
            x_t = teacher.q_sample(action, t, noise).to(action.dtype)

            with torch.no_grad():
                x_s = teacher.ddim_sample(
                    x_t, t, state, None, prev_t=timestep - 1
                )
                target_clean = predict_clean(target, x_s, s, state).clamp(
                    teacher.min_action, teacher.max_action
                ).to(action.dtype)

            online_clean = predict_clean(student, x_t, t, state).to(action.dtype)
            consistency_loss = F.mse_loss(online_clean, target_clean)
            diffusion_loss = dsm_loss(student, action, state)
            loss = (
                args.consistency_weight * consistency_loss
                + args.diffusion_weight * diffusion_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.get_params(), 1.0)
            optimizer.step()
            update_target(target, student, args.ema_decay)

            totals.append(loss.detach().item())
            consistency_losses.append(consistency_loss.detach().item())
            diffusion_losses.append(diffusion_loss.detach().item())

        scheduler.step()
        record = {
            "epoch": epoch,
            "loss": float(np.mean(totals)),
            "ctm_consistency_loss": float(np.mean(consistency_losses)),
            "diffusion_loss": float(np.mean(diffusion_losses)),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        if record["loss"] < best_loss:
            best_loss, best_epoch = record["loss"], epoch
            torch.save(target.state_dict(), args.output_dir / "eval_best_ddpm.pth")
        if epoch % 10 == 0 or epoch + 1 == args.epochs:
            print(json.dumps(record), flush=True)

    torch.save(target.state_dict(), args.output_dir / "last_ddpm.pth")
    return best_epoch, best_loss, history


def train_consistency_policy(args, teacher_agent, teacher, student):
    """Boundary-conditioned local consistency with SNR weighting and DSM."""
    target = copy.deepcopy(student).eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(student.get_params(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    best_loss, best_epoch, history = math.inf, -1, []
    for epoch in trange(args.epochs, desc="Consistency Policy 8-to-1"):
        totals, consistency_losses, diffusion_losses = [], [], []
        for batch_index, (state, action, _) in enumerate(teacher_agent.train_dataloader):
            if args.max_batches_per_epoch and batch_index >= args.max_batches_per_epoch:
                break
            state = teacher_agent.scaler.scale_input(state).float()
            action = teacher_agent.scaler.scale_output(action).float()
            timestep = int(torch.randint(1, teacher.n_timesteps, (1,)).item())
            max_jump = min(args.max_teacher_jump, timestep)
            jump = int(torch.randint(1, max_jump + 1, (1,)).item())
            target_timestep = timestep - jump
            t = torch.full((action.shape[0],), timestep, device=action.device, dtype=torch.long)
            s_t = torch.full_like(t, target_timestep)
            x_t = teacher.q_sample(action, t, torch.randn_like(action)).to(action.dtype)
            with torch.no_grad():
                x_s = teacher.ddim_sample(x_t, t, state, None, prev_t=target_timestep)
                target_clean = consistency_output(target, x_s, s_t, state, args.sigma_data).clamp(teacher.min_action, teacher.max_action)
            online_clean = consistency_output(student, x_t, t, state, args.sigma_data)
            alpha = extract(teacher.alphas_cumprod, t, x_t.shape).to(x_t.dtype)
            snr = alpha / (1.0 - alpha).clamp_min(1e-6)
            weight = (snr + 1.0).rsqrt().clamp(0.25, 4.0)
            consistency_loss = pseudo_huber(online_clean, target_clean, args.huber_delta, weight)
            diffusion_loss = dsm_loss(student, action, state)
            loss = args.consistency_weight * consistency_loss + args.diffusion_weight * diffusion_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.get_params(), 1.0)
            optimizer.step()
            update_target(target, student, args.ema_decay)
            totals.append(loss.detach().item()); consistency_losses.append(consistency_loss.detach().item()); diffusion_losses.append(diffusion_loss.detach().item())
        scheduler.step()
        record = {"epoch": epoch, "loss": float(np.mean(totals)), "consistency_policy_loss": float(np.mean(consistency_losses)), "diffusion_loss": float(np.mean(diffusion_losses)), "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(record)
        if record["loss"] < best_loss:
            best_loss, best_epoch = record["loss"], epoch
            torch.save(target.state_dict(), args.output_dir / "eval_best_ddpm.pth")
        if epoch % 10 == 0 or epoch + 1 == args.epochs:
            print(json.dumps(record), flush=True)
    torch.save(target.state_dict(), args.output_dir / "last_ddpm.pth")
    return best_epoch, best_loss, history


def train_distribution(args, teacher_agent, teacher, student):
    """One-step distribution matching with a learned fake-score network."""
    fake_score = copy.deepcopy(teacher).train()
    for parameter in fake_score.parameters():
        parameter.requires_grad_(True)
    generator_optimizer = torch.optim.Adam(student.get_params(), lr=args.learning_rate)
    fake_optimizer = torch.optim.Adam(fake_score.get_params(), lr=args.fake_learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(generator_optimizer, T_max=args.epochs, eta_min=1e-6)
    ema = ExponentialMovingAverage(student.get_params(), args.ema_decay, teacher_agent.device)
    best_loss, best_epoch, history = math.inf, -1, []
    for epoch in trange(args.epochs, desc="Distribution/score distillation 8-to-1"):
        totals, distribution_losses, diffusion_losses, fake_losses = [], [], [], []
        for batch_index, (state, action, _) in enumerate(teacher_agent.train_dataloader):
            if args.max_batches_per_epoch and batch_index >= args.max_batches_per_epoch:
                break
            state = teacher_agent.scaler.scale_input(state).float()
            action = teacher_agent.scaler.scale_output(action).float()
            batch_size = action.shape[0]
            high_t = torch.full((batch_size,), teacher.n_timesteps - 1, device=action.device, dtype=torch.long)
            generated = predict_clean(student, torch.randn_like(action), high_t, state).clamp(teacher.min_action, teacher.max_action)
            fake_t = torch.randint(0, teacher.n_timesteps, (batch_size,), device=action.device)
            fake_noise = torch.randn_like(action)
            fake_x = teacher.q_sample(generated.detach(), fake_t, fake_noise).to(action.dtype)
            fake_prediction = fake_score.model(fake_x, fake_t, state, None)
            fake_loss = F.mse_loss(fake_prediction, fake_noise)
            fake_optimizer.zero_grad(set_to_none=True)
            fake_loss.backward()
            torch.nn.utils.clip_grad_norm_(fake_score.get_params(), 1.0)
            fake_optimizer.step()
            match_t = torch.randint(1, teacher.n_timesteps, (batch_size,), device=action.device)
            match_x = teacher.q_sample(generated, match_t, torch.randn_like(action)).to(action.dtype)
            with torch.no_grad():
                teacher_epsilon = teacher.model(match_x.detach(), match_t, state, None)
                fake_epsilon = fake_score.model(match_x.detach(), match_t, state, None)
                score_delta = teacher_epsilon - fake_epsilon
                dims = tuple(range(1, score_delta.ndim))
                scale = score_delta.abs().mean(dim=dims, keepdim=True).clamp_min(1e-4)
                score_delta = score_delta / scale
            distribution_loss = (generated * score_delta.detach()).mean()
            diffusion_loss = dsm_loss(student, action, state)
            loss = args.distribution_weight * distribution_loss + args.diffusion_weight * diffusion_loss
            generator_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.get_params(), 1.0)
            generator_optimizer.step()
            ema.update(student.get_params())
            totals.append(loss.detach().item()); distribution_losses.append(distribution_loss.detach().item()); diffusion_losses.append(diffusion_loss.detach().item()); fake_losses.append(fake_loss.detach().item())
        scheduler.step()
        record = {"epoch": epoch, "loss": float(np.mean(totals)), "distribution_loss": float(np.mean(distribution_losses)), "diffusion_loss": float(np.mean(diffusion_losses)), "fake_score_loss": float(np.mean(fake_losses)), "learning_rate": generator_optimizer.param_groups[0]["lr"]}
        history.append(record)
        selection_loss = abs(record["distribution_loss"]) + args.diffusion_weight * record["diffusion_loss"]
        if selection_loss < best_loss:
            best_loss, best_epoch = selection_loss, epoch
            save_ema_model(student, ema, args.output_dir, "eval_best_ddpm.pth")
        if epoch % 10 == 0 or epoch + 1 == args.epochs:
            print(json.dumps(record), flush=True)
    save_ema_model(student, ema, args.output_dir, "last_ddpm.pth")
    return best_epoch, best_loss, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["progressive", "ctm", "consistency_policy", "distribution"], required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--teacher-steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--diffusion-weight", type=float, default=0.1)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--sigma-data", type=float, default=0.5)
    parser.add_argument("--huber-delta", type=float, default=0.03)
    parser.add_argument("--max-teacher-jump", type=int, default=4)
    parser.add_argument("--distribution-weight", type=float, default=0.1)
    parser.add_argument("--fake-learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-batches-per-epoch", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wandb.init(mode="disabled")

    teacher_agent = make_teacher(args.teacher_dir, args.batch_size)
    teacher = teacher_agent.model
    teacher.eval()
    teacher.sampler = "ddim"
    teacher.ddim_eta = 0.0
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    student = copy.deepcopy(teacher)
    for parameter in student.parameters():
        parameter.requires_grad_(True)
    student.train()
    student.sampling_steps = 1

    if args.method == "progressive":
        best_epoch, best_loss, history = train_progressive(
            args, teacher_agent, teacher, student
        )
    elif args.method == "ctm":
        best_epoch, best_loss, history = train_ctm(args, teacher_agent, teacher, student)
    elif args.method == "consistency_policy":
        best_epoch, best_loss, history = train_consistency_policy(args, teacher_agent, teacher, student)
    else:
        best_epoch, best_loss, history = train_distribution(args, teacher_agent, teacher, student)

    summary = {
        "method": args.method,
        "teacher_steps": args.teacher_steps,
        "student_steps": 1,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "diffusion_weight": args.diffusion_weight,
        "consistency_weight": args.consistency_weight,
        "ema_decay": args.ema_decay,
        "sigma_data": args.sigma_data,
        "huber_delta": args.huber_delta,
        "max_teacher_jump": args.max_teacher_jump,
        "distribution_weight": args.distribution_weight,
        "teacher_checkpoint": str(args.teacher_dir / "eval_best_ddpm.pth"),
        "history": history,
    }
    with open(args.output_dir / "distillation_metrics.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))
    wandb.finish()


if __name__ == "__main__":
    main()
