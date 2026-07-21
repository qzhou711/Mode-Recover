import argparse
import json
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from hydra import compose, initialize

from envs.gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv
from envs.gym_avoiding_env.gym_avoiding.envs.objects.avoiding_objects import get_obj_xy_list


def make_agent(agent_config, window_size, weights_dir, weights_name):
    with initialize(config_path="configs"):
        cfg = compose(
            config_name="avoiding_config",
            overrides=[
                f"agents={agent_config}",
                f"window_size={window_size}",
                "epoch=1",
                "simulation.render=False",
            ],
        )
    agent = hydra.utils.instantiate(cfg.agents)
    agent.load_pretrained_model(str(weights_dir), sv_name=weights_name)
    return agent


def rollout(agent, n_trajectories, seed):
    env = ObstacleAvoidanceEnv(render=False)
    env.start()
    trajectories, successes, modes = [], [], []
    for episode in range(n_trajectories):
        np.random.seed(seed + episode)
        torch.manual_seed(seed + episode)
        agent.reset()
        obs = env.reset()
        pred_action = env.robot_state()
        fixed_z = pred_action[2:]
        path = [env.robot.current_c_pos[:2].copy()]
        done = False
        info = (np.zeros(9), False)
        while not done:
            policy_obs = np.concatenate((pred_action[:2], obs))
            pred_action = agent.predict(policy_obs)[0] + policy_obs[:2]
            command = np.concatenate((pred_action, fixed_z, [0, 1, 0, 0]))
            obs, _, done, info = env.step(command)
            path.append(env.robot.current_c_pos[:2].copy())
        trajectories.append(np.asarray(path))
        modes.append(np.asarray(info[0], dtype=np.int8))
        successes.append(bool(info[1]))
    return trajectories, np.asarray(successes), np.asarray(modes)


def metrics(successes, modes):
    successful_modes = modes[successes]
    if len(successful_modes) == 0:
        entropy, unique_modes = 0.0, 0
    else:
        encoded = successful_modes.dot(1 << np.arange(successful_modes.shape[1]))
        _, counts = np.unique(encoded, return_counts=True)
        probs = counts / counts.sum()
        entropy = float(-(probs * np.log(probs) / np.log(24)).sum())
        unique_modes = int(len(counts))
    return {
        "n_trajectories": int(len(successes)),
        "success_rate": float(successes.mean()),
        "successful_trajectories": int(successes.sum()),
        "unique_successful_modes": unique_modes,
        "normalized_mode_entropy": entropy,
    }


def draw(ax, trajectories, successes, title):
    for path, success in zip(trajectories, successes):
        ax.plot(path[:, 0], path[:, 1], color="#1976d2" if success else "#d32f2f", alpha=0.45, lw=1.2)
    for x, y in get_obj_xy_list():
        ax.add_patch(plt.Circle((x, y), 0.03, color="black", alpha=0.8))
    ax.axhline(0.35, color="#2e7d32", ls="--", lw=1.5, label="goal")
    ax.scatter([0.525], [-0.28], marker="*", s=90, color="#ff9800", zorder=5, label="start")
    ax.set(title=title, xlabel="x [m]", ylabel="y [m]", xlim=(0.25, 0.75), ylim=(-0.32, 0.42))
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trajectories", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", choices=["bc", "ddpm"], default=["bc", "ddpm"])
    parser.add_argument("--ddpm-weights-dir", type=Path, default=Path("logs/avoiding/trained/ddpm_transformer_seed42"))
    parser.add_argument("--ddpm-sampler", choices=["ddpm", "ddim"], default="ddpm")
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--ddim-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("logs/avoiding/trajectory_comparison"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wandb.init(mode="disabled")

    specs = {
        "BC": ("bc_agent", 1, Path("logs/avoiding/trained/bc_seed42"), "eval_best_bc.pth"),
        "DDPM-Transformer": (
            "ddpm_transformer_agent", 5,
            args.ddpm_weights_dir, "eval_best_ddpm.pth",
        ),
    }
    selected = {"bc": "BC", "ddpm": "DDPM-Transformer"}
    specs = {selected[key]: specs[selected[key]] for key in args.models}
    results = {}
    raw = {}
    for name, spec in specs.items():
        agent = make_agent(*spec)
        if name == "DDPM-Transformer":
            agent.model.sampler = args.ddpm_sampler
            agent.model.ddim_eta = args.ddim_eta
            if args.ddim_steps is not None:
                agent.model.sampling_steps = args.ddim_steps
        trajectories, successes, modes = rollout(agent, args.n_trajectories, args.seed)
        results[name] = metrics(successes, modes)
        raw[name] = (trajectories, successes, modes)
        np.savez_compressed(
            args.output_dir / f"{name.lower().replace('-', '_')}_trajectories.npz",
            trajectories=np.asarray(trajectories, dtype=object), successes=successes, modes=modes,
        )

    fig, axes = plt.subplots(1, len(raw), figsize=(5.5 * len(raw), 5), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (name, (trajectories, successes, _)) in zip(axes, raw.items()):
        m = results[name]
        draw(ax, trajectories, successes, f"{name}: success={m['success_rate']:.1%}, modes={m['unique_successful_modes']}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.savefig(args.output_dir / "trajectory_comparison.png", dpi=220)
    plt.close(fig)

    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    wandb.finish()


if __name__ == "__main__":
    main()
