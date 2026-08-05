Long-term objective: demonstration-free arbitrary-Top-K architecture selection that preserves conditional robot/WAM behavior, followed separately by step distillation.

Short-term bottleneck: the L4K2 five-epoch calibration selected keep02 on four behavior/distribution metrics but keep03 on local velocity-CVaR.

Stage targeted: architecture-compression reward identifiability before Top-2 search.

Mechanism hypothesis: paired multi-seed 10/25-epoch repair will distinguish finite-sample noise from a persistent conflict between local vector-field fidelity and generated trajectory/distribution fidelity.

Causal intervention and controls: keep02 versus keep03, seeds 42-45, checkpoints 10 and 25; within each seed both masks use identical teacher mapping, data order, repair budget, held-out episodes, and proxy noise protocol.

Data-access status: teacher rollout only; no demonstrations, expert actions, rewards, success filtering, or mode labels; repair excludes the held-out episode residue.

Metrics and go/no-go: Endpoint-CVaR, short-trajectory paired error, Teacher-to-Student coverage, and Student-to-Teacher precision are behavior/distribution metrics; velocity-CVaR is separately reported as local fidelity. Start Top-2 search only if the four behavior metrics consistently prefer one mask across at least 7/8 seed-budget pairs. A persistent velocity-only reversal is recorded as target conflict, not averaged away.

Resource/time check: four idle V100-32GB GPUs and approximately 2h55 allocation remaining; two four-job waves plus search fit the allocation and save checkpoints/results.

Expected interpretations: stable keep02 behavior preference with keep03 velocity preference proves objective conflict; all metrics converging to one mask indicates the five-epoch discrepancy was noise; unstable behavior metrics blocks Top-2 search.
