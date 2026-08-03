#!/usr/bin/env bash
#SBATCH --job-name=p6_induced_collect
#SBATCH --account=ccr200024p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:4
#SBATCH --cpus-per-task=20
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=logs/avoiding/p6_student_induced_collection_%j.log

set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
bash scripts/run_p6_student_induced_full_pipeline.sh
