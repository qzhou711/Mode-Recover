#!/usr/bin/env bash
#SBATCH --job-name=p6_pareto_round2
#SBATCH --account=ccr200024p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:4
#SBATCH --cpus-per-task=20
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=logs/avoiding/p6_pareto_round2_%j.log
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
bash scripts/run_p6_pareto480_and_round2_audit.sh
