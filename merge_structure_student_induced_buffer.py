import argparse
import json
from pathlib import Path

import numpy as np
import torch


p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--expected-episodes", type=int, default=480)
p.add_argument("--episode-start", type=int, default=3000)
a = p.parse_args()
files = sorted(a.root.glob("shards/*/student_induced_buffer.pt"))
if not files:
    raise SystemExit("no student-induced shards")
parts = [torch.load(path, map_location="cpu") for path in files]
tensor_keys = [
    "states", "noises", "student_endpoints", "teacher_corrections",
    "episode_ids", "control_steps", "student_successes", "student_modes",
]
merged = {key: torch.cat([part[key] for part in parts]) for key in tensor_keys}
merged["student_paths"] = sum([part["student_paths"] for part in parts], [])
merged["metadata"] = {
    "teacher_architecture": "FM-4x72-16",
    "student_architecture": "FM-3x48-16",
    "uses_original_demonstrations": False,
    "uses_expert_actions": False,
    "teacher_queries_on_student_states": True,
    "shards": [str(path) for path in files],
}
episodes = merged["student_successes"].numel()
unique = torch.unique(merged["episode_ids"])
expected_ids = torch.arange(a.episode_start, a.episode_start + a.expected_episodes)
checks = {
    "episode_count": episodes == a.expected_episodes,
    "unique_episode_count": unique.numel() == a.expected_episodes,
    "episode_ids_contiguous": torch.equal(unique, expected_ids),
    "sample_alignment": len({len(merged[key]) for key in ["states", "noises", "student_endpoints", "teacher_corrections", "episode_ids", "control_steps"]}) == 1,
    "finite": all(torch.isfinite(merged[key]).all().item() for key in ["states", "noises", "student_endpoints", "teacher_corrections"]),
    "no_demonstrations": not merged["metadata"]["uses_original_demonstrations"],
    "teacher_on_student_states": merged["metadata"]["teacher_queries_on_student_states"],
}
success = merged["student_successes"].numpy().astype(bool)
modes = merged["student_modes"].numpy()[success]
if len(modes):
    encoded = modes.dot(1 << np.arange(modes.shape[1]))
    _, counts = np.unique(encoded, return_counts=True)
    probability = counts / counts.sum()
    coverage = len(counts)
    entropy = float(-(probability * np.log(probability) / np.log(24)).sum())
else:
    coverage, entropy = 0, 0.0
metrics = {
    "passed": all(checks.values()),
    "checks": checks,
    "episodes": episodes,
    "samples": len(merged["states"]),
    "student_success_rate": float(success.mean()),
    "student_mode_coverage": coverage,
    "student_mode_entropy": entropy,
    "uses_original_demonstrations": False,
}
torch.save(merged, a.root / "student_induced_buffer.pt")
(a.root / "validation.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
if not metrics["passed"]:
    raise SystemExit(2)
