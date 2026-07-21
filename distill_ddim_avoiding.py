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


def ddim_rollout(model, state, initial_noise, sampling_steps):
    schedule = np.linspace(
        model.n_timesteps - 1, 0, sampling_steps, dtype=np.int64
    ).tolist()
    x = initial_noise
    for index, timestep in enumerate(schedule):
        t = torch.full(
            (x.shape[0],), timestep, device=x.device, dtype=torch.long
        )
        prev_t = schedule[index + 1] if index + 1 < len(schedule) else -1
        x = model.ddim_sample(x, t, state, None, prev_t=prev_t)
    return x.clamp(model.min_action, model.max_action)


def save_ema_model(student, ema, output_dir: Path, name: str):
    ema.store(student.get_params())
    ema.copy_to(student.get_params())
    torch.save(student.state_dict(), output_dir / name)
    ema.restore(student.get_params())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher-dir",
        type=Path,
        default=Path(
            "logs/avoiding/trained/ddpm_transformer_10000_cosine_seed42"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/avoiding/distilled/ddim_student_4step_seed42"),
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--teacher-steps", type=int, default=8)
    parser.add_argument("--student-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--diffusion-weight", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.student_steps >= args.teacher_steps:
        raise ValueError("student-steps must be smaller than teacher-steps")
    if args.teacher_steps > 8:
        raise ValueError("teacher-steps cannot exceed the underlying 8-step model")

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
    student.sampling_steps = args.student_steps

    optimizer = torch.optim.Adam(
        student.get_params(), lr=args.learning_rate
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    ema = ExponentialMovingAverage(
        student.get_params(), args.ema_decay, teacher_agent.device
    )

    best_loss = math.inf
    best_epoch = -1
    history = []

    for epoch in trange(
        args.epochs, desc=f"{args.teacher_steps}-to-{args.student_steps} distillation"
    ):
        epoch_total = []
        epoch_consistency = []
        epoch_diffusion = []

        for state, action, _ in teacher_agent.train_dataloader:
            state = teacher_agent.scaler.scale_input(state)
            action = teacher_agent.scaler.scale_output(action)
            noise = torch.randn_like(action)

            with torch.no_grad():
                teacher_target = ddim_rollout(
                    teacher, state, noise, args.teacher_steps
                )

            student_output = ddim_rollout(
                student, state, noise, args.student_steps
            )
            consistency_loss = F.mse_loss(student_output, teacher_target)
            if args.diffusion_weight > 0:
                diffusion_loss = student.loss(action, state, None)
                loss = consistency_loss + args.diffusion_weight * diffusion_loss
            else:
                diffusion_loss = torch.zeros((), device=consistency_loss.device)
                loss = consistency_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.get_params(), 1.0)
            optimizer.step()
            ema.update(student.get_params())

            epoch_total.append(loss.detach().item())
            epoch_consistency.append(consistency_loss.detach().item())
            epoch_diffusion.append(diffusion_loss.detach().item())

        scheduler.step()
        mean_total = float(np.mean(epoch_total))
        record = {
            "epoch": epoch,
            "loss": mean_total,
            "consistency_loss": float(np.mean(epoch_consistency)),
            "diffusion_loss": float(np.mean(epoch_diffusion)),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)

        if mean_total < best_loss:
            best_loss = mean_total
            best_epoch = epoch
            save_ema_model(
                student, ema, args.output_dir, "eval_best_ddpm.pth"
            )

        if epoch % 10 == 0 or epoch + 1 == args.epochs:
            print(json.dumps(record), flush=True)

    save_ema_model(student, ema, args.output_dir, "last_ddpm.pth")
    torch.save(student.state_dict(), args.output_dir / "non_ema_model_state_dict.pth")
    summary = {
        "teacher_steps": args.teacher_steps,
        "student_steps": args.student_steps,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "diffusion_weight": args.diffusion_weight,
        "teacher_checkpoint": str(args.teacher_dir / "eval_best_ddpm.pth"),
        "history": history,
    }
    with open(args.output_dir / "distillation_metrics.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))
    wandb.finish()


if __name__ == "__main__":
    main()
