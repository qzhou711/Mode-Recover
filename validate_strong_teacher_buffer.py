"""Hard gate for the expanded demonstration-free strong-Teacher buffer."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--buffer", type=Path, required=True)
parser.add_argument("--bundle-dir", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--episodes", type=int, default=2400)
parser.add_argument("--minimum-sr", type=float, default=0.90)
parser.add_argument("--minimum-mode-count", type=int, default=20)
args = parser.parse_args()
data = torch.load(args.buffer, map_location="cpu")
metadata = torch.load(args.bundle_dir / "deployment_metadata.pt", map_location="cpu")
episode_ids = data["episode_ids"].long()
unique_episodes = torch.unique(episode_ids)
success = data["successes"].bool()
mode_codes = (data["modes"].long() * (1 << torch.arange(data["modes"].shape[1]))).sum(1)
successful_codes = mode_codes[success]
codes, counts = torch.unique(successful_codes, return_counts=True)
step_pairs = torch.stack((episode_ids, data["control_steps"].long()), 1)
checks = {
    "teacher_architecture": metadata["teacher_layers"] == 4
        and metadata["teacher_embed_dim"] == 72
        and metadata["teacher_heads"] == 4
        and metadata["teacher_steps"] == 16,
    "uses_no_original_demonstrations": not data["metadata"]["uses_original_demonstrations"],
    "uses_no_expert_actions": not data["metadata"]["uses_expert_actions"],
    "episode_count": len(success) == args.episodes,
    "unique_episode_count": len(unique_episodes) == args.episodes,
    "episode_id_range": int(unique_episodes.min()) == 0
        and int(unique_episodes.max()) == args.episodes - 1,
    "episode_ids_contiguous": torch.equal(unique_episodes, torch.arange(args.episodes)),
    "unique_episode_timestep_pairs": len(torch.unique(step_pairs, dim=0)) == len(step_pairs),
    "sample_tensors_aligned": len(data["states"]) == len(data["noises"])
        == len(data["teacher_endpoints"]) == len(episode_ids) == len(data["control_steps"]),
    "finite_states": bool(torch.isfinite(data["states"]).all()),
    "finite_noises": bool(torch.isfinite(data["noises"]).all()),
    "finite_endpoints": bool(torch.isfinite(data["teacher_endpoints"]).all()),
    "success_rate": float(success.float().mean()) >= args.minimum_sr,
    "mode_coverage": len(codes) == 24,
    "minimum_mode_count": int(counts.min()) >= args.minimum_mode_count if len(counts) else False,
}
result = {
    "passed": all(checks.values()),
    "checks": checks,
    "teacher": {
        "layers": metadata["teacher_layers"],
        "embed_dim": metadata["teacher_embed_dim"],
        "heads": metadata["teacher_heads"],
        "steps": metadata["teacher_steps"],
        "source_checkpoint": metadata["source_checkpoint"],
    },
    "episodes": len(success),
    "samples": len(data["states"]),
    "success_rate": float(success.float().mean()),
    "successful_episodes": int(success.sum()),
    "mode_coverage": len(codes),
    "mode_codes": codes.tolist(),
    "mode_counts": counts.tolist(),
    "minimum_mode_count": int(counts.min()) if len(counts) else 0,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
if not result["passed"]:
    raise SystemExit(2)
