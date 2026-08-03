#!/usr/bin/env bash
#SBATCH --job-name=strong_teacher_p6
#SBATCH --account=ccr200024p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:4
#SBATCH --cpus-per-task=20
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=logs/avoiding/strong_teacher_p6_slurm_%j.log

set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il

# This pipeline is idempotent: every completed 15-episode shard is skipped,
# then all 2400 episodes are merged and strictly validated before discovery.
bash scripts/run_strong_teacher_expansion_and_discovery.sh
