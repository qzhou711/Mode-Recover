"""Collect demo-free intervention DAgger trajectories for structural distillation."""
import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from envs.gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv
from teacher_flow_deployment import DeploymentScaler, build_flow


def load_policy(path, layers, embed_dim, heads, steps, meta):
    model = build_flow(layers, embed_dim, heads, "cuda", steps)
    checkpoint = path if path.is_file() else path / "eval_best_flow.pth"
    model.load_state_dict(torch.load(checkpoint, map_location="cuda"), strict=True)
    model.min_action = meta["y_bounds_tensor"][0].cuda()
    model.max_action = meta["y_bounds_tensor"][1].cuda()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle-dir", type=Path, required=True)
    p.add_argument("--student", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--episode-start", type=int, required=True)
    p.add_argument("--n-episodes", type=int, required=True)
    p.add_argument("--takeover-horizon", type=int, required=True)
    p.add_argument("--takeover-to-end", action="store_true",
                   help="After the first trigger, let the teacher control the rest of the episode.")
    p.add_argument("--threshold", type=float, default=0.4180944411691571)
    p.add_argument("--seed", type=int, default=271828)
    p.add_argument("--progress-every", type=int, default=5)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    meta = torch.load(a.bundle_dir / "deployment_metadata.pt", map_location="cpu")
    assert (meta["teacher_layers"], meta["teacher_embed_dim"], meta["teacher_heads"], meta["teacher_steps"]) == (4, 72, 4, 16)
    teacher_path = Path(meta["source_checkpoint"])
    if not teacher_path.is_absolute():
        teacher_path = Path.cwd() / teacher_path
    teacher = load_policy(teacher_path, 4, 72, 4, 16, meta)
    student = load_policy(a.student, 3, 48, 3, 16, meta)
    scaler = DeploymentScaler(meta, "cuda")
    env = ObstacleAvoidanceEnv(render=False)
    env.start()

    states, noises, student_endpoints, teacher_corrections = [], [], [], []
    episode_ids, control_steps, disagreements = [], [], []
    teacher_control, triggers = [], []
    paths, successes, modes, intervention_counts, intervention_steps = [], [], [], [], []
    for local in range(a.n_episodes):
        eid = a.episode_start + local
        np.random.seed(a.seed + eid)
        torch.manual_seed(a.seed + eid)
        context = deque(maxlen=meta["window_size"])
        obs = env.reset()
        pred_action = env.robot_state()
        fixed_z = pred_action[2:]
        path = [env.robot.current_c_pos[:2].copy()]
        done = False
        info = (np.zeros(9), False)
        step = 0
        takeover_remaining = 0
        episode_triggers = 0
        episode_teacher_steps = 0
        while not done:
            raw = np.concatenate((pred_action[:2], obs))
            scaled = scaler.scale_input(torch.from_numpy(raw).float().view(1, 4))
            context.append(scaled)
            state = torch.stack(tuple(context), dim=1)
            generator = torch.Generator(device="cpu").manual_seed(a.seed + eid * 100003 + step * 1009)
            noise = torch.randn(1, state.shape[1], 2, generator=generator).cuda()
            with torch.no_grad():
                student_endpoint = student.sample(state, initial_noise=noise, steps=16)
                teacher_endpoint = teacher.sample(state, initial_noise=noise, steps=16)
            disagreement = float((student_endpoint - teacher_endpoint).square().mean().sqrt())
            triggered = takeover_remaining == 0 and disagreement > a.threshold
            if triggered:
                takeover_remaining = -1 if a.takeover_to_end else a.takeover_horizon
                episode_triggers += 1
            use_teacher = takeover_remaining != 0
            endpoint = teacher_endpoint if use_teacher else student_endpoint
            if use_teacher:
                if takeover_remaining > 0:
                    takeover_remaining -= 1
                episode_teacher_steps += 1
            if len(context) == meta["window_size"]:
                states.append(state[0].cpu()); noises.append(noise[0].cpu())
                student_endpoints.append(student_endpoint[0].cpu()); teacher_corrections.append(teacher_endpoint[0].cpu())
                episode_ids.append(eid); control_steps.append(step); disagreements.append(disagreement)
                teacher_control.append(use_teacher); triggers.append(triggered)
            action = scaler.inverse_scale_output(endpoint[:, -1])[0].cpu().numpy()
            pred_action = action + raw[:2]
            command = np.concatenate((pred_action, fixed_z, [0, 1, 0, 0]))
            obs, _, done, info = env.step(command)
            path.append(env.robot.current_c_pos[:2].copy())
            step += 1
        paths.append(np.asarray(path)); successes.append(bool(info[1])); modes.append(np.asarray(info[0], dtype=np.int8))
        intervention_counts.append(episode_triggers); intervention_steps.append(episode_teacher_steps)
        if (local + 1) % a.progress_every == 0 or local + 1 == a.n_episodes:
            print(json.dumps({"progress": {"completed": local + 1, "total": a.n_episodes, "successes": int(sum(successes)), "interventions": int(sum(intervention_counts))}}), flush=True)
    payload = {
        "states": torch.stack(states), "noises": torch.stack(noises),
        "student_endpoints": torch.stack(student_endpoints), "teacher_corrections": torch.stack(teacher_corrections),
        "episode_ids": torch.tensor(episode_ids), "control_steps": torch.tensor(control_steps),
        "disagreements": torch.tensor(disagreements), "teacher_control": torch.tensor(teacher_control),
        "triggers": torch.tensor(triggers), "paths": paths, "successes": torch.tensor(successes),
        "modes": torch.from_numpy(np.stack(modes)), "intervention_counts": torch.tensor(intervention_counts),
        "intervention_steps": torch.tensor(intervention_steps),
        "metadata": {"teacher_architecture": "FM-4x72-16", "student_architecture": "FM-3x48-16",
                     "takeover_horizon": a.takeover_horizon, "threshold": a.threshold,
                     "takeover_to_end": a.takeover_to_end,
                     "uses_original_demonstrations": False, "uses_expert_actions": False,
                     "teacher_queries_on_student_states": True, "seed": a.seed},
    }
    torch.save(payload, a.output_dir / "intervention_buffer.pt")


if __name__ == "__main__":
    main()
