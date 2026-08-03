#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
bash scripts/run_p6_student_induced_collection.sh
bash scripts/run_p6_student_induced_repair_four_gpu.sh
