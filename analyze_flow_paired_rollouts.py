import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
import wandb

from envs.gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv
from visualize_avoiding import make_agent


class PairedNoiseFlowRunner:
    def __init__(self, agent, steps, base_seed):
        self.agent = agent
        self.steps = steps
        self.base_seed = base_seed
        self.context = deque(maxlen=agent.window_size)
        self.agent.model.eval()

    def reset(self):
        self.context.clear()

    @torch.no_grad()
    def predict(self, state, episode_id, control_step):
        state = torch.from_numpy(state).float().to(self.agent.device).unsqueeze(0)
        state = self.agent.scaler.scale_input(state)
        self.context.append(state)
        input_state = torch.stack(tuple(self.context), dim=1)
        generator = torch.Generator(device="cpu").manual_seed(
            self.base_seed + episode_id * 100003 + control_step * 1009
        )
        full_noise = torch.randn(
            1, self.agent.window_size, 2, generator=generator
        )
        noise = full_noise[:, -input_state.shape[1] :].to(self.agent.device)
        prediction = self.agent.model.sample(
            input_state, initial_noise=noise, steps=self.steps
        )
        prediction = self.agent.scaler.inverse_scale_output(prediction)
        return prediction[:, -1].detach().cpu().numpy()


def run_episode(env, runner, episode_id, seed):
    np.random.seed(seed + episode_id)
    torch.manual_seed(seed + episode_id)
    runner.reset()
    obs = env.reset()
    pred_action = env.robot_state()
    fixed_z = pred_action[2:]
    path = [env.robot.current_c_pos[:2].copy()]
    done = False
    info = (np.zeros(9), False)
    control_step = 0
    while not done:
        policy_obs = np.concatenate((pred_action[:2], obs))
        pred_action = (
            runner.predict(policy_obs, episode_id, control_step)[0] + policy_obs[:2]
        )
        command = np.concatenate((pred_action, fixed_z, [0, 1, 0, 0]))
        obs, _, done, info = env.step(command)
        path.append(env.robot.current_c_pos[:2].copy())
        control_step += 1
    return (
        np.asarray(path),
        bool(info[1]),
        np.asarray(info[0], dtype=np.int8),
        control_step,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--teacher-layers", type=int, required=True)
    parser.add_argument("--teacher-embed-dim", type=int, required=True)
    parser.add_argument("--teacher-heads", type=int, required=True)
    parser.add_argument("--teacher-steps", type=int, default=16)
    parser.add_argument("--student-dir", type=Path, required=True)
    parser.add_argument("--student-layers", type=int, required=True)
    parser.add_argument("--student-embed-dim", type=int, required=True)
    parser.add_argument("--student-heads", type=int, required=True)
    parser.add_argument("--student-steps", type=int, default=1)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--n-episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wandb.init(mode="disabled")

    teacher_agent = make_agent(
        "flow_matching_transformer_agent",
        5,
        args.teacher_dir,
        "eval_best_flow.pth",
        [
            f"n_layer={args.teacher_layers}",
            f"n_embd={args.teacher_embed_dim}",
            f"n_head={args.teacher_heads}",
        ],
    )
    student_agent = make_agent(
        "flow_matching_transformer_agent",
        5,
        args.student_dir,
        "eval_best_flow.pth",
        [
            f"n_layer={args.student_layers}",
            f"n_embd={args.student_embed_dim}",
            f"n_head={args.student_heads}",
        ],
    )
    teacher = PairedNoiseFlowRunner(teacher_agent, args.teacher_steps, args.seed)
    student = PairedNoiseFlowRunner(student_agent, args.student_steps, args.seed)
    env = ObstacleAvoidanceEnv(render=False)
    env.start()

    records = {
        "episode_ids": [],
        "teacher_paths": [],
        "student_paths": [],
        "teacher_successes": [],
        "student_successes": [],
        "teacher_modes": [],
        "student_modes": [],
        "teacher_steps": [],
        "student_steps": [],
    }
    for local_episode in range(args.n_episodes):
        episode_id = args.episode_start + local_episode
        teacher_result = run_episode(env, teacher, episode_id, args.seed)
        student_result = run_episode(env, student, episode_id, args.seed)
        records["episode_ids"].append(episode_id)
        for prefix, result in (("teacher", teacher_result), ("student", student_result)):
            path, success, mode, steps = result
            records[f"{prefix}_paths"].append(path)
            records[f"{prefix}_successes"].append(success)
            records[f"{prefix}_modes"].append(mode)
            records[f"{prefix}_steps"].append(steps)

        completed = local_episode + 1
        if completed % args.progress_every == 0 or completed == args.n_episodes:
            progress = {
                "episode_start": args.episode_start,
                "completed": completed,
                "total": args.n_episodes,
                "teacher_successes": int(np.sum(records["teacher_successes"])),
                "student_successes": int(np.sum(records["student_successes"])),
                "finished": completed == args.n_episodes,
            }
            print(json.dumps({"paired_progress": progress}), flush=True)
            (args.output_dir / "progress.json").write_text(json.dumps(progress, indent=2))
            np.savez_compressed(
                args.output_dir / "paired_checkpoint.npz",
                **{
                    key: np.asarray(value, dtype=object)
                    if key.endswith("_paths")
                    else np.asarray(value)
                    for key, value in records.items()
                },
            )

    np.savez_compressed(
        args.output_dir / "paired_rollouts.npz",
        **{
            key: np.asarray(value, dtype=object)
            if key.endswith("_paths")
            else np.asarray(value)
            for key, value in records.items()
        },
    )
    wandb.finish()


if __name__ == "__main__":
    main()
