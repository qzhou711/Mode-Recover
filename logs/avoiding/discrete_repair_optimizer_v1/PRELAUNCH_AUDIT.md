Long-term objective: demonstration-free, cross-architecture compression that transfers from D3IL to large robot/WAM policies while preserving conditional behavior diversity.

Short-term bottleneck: straight-through shared-supernet gating optimizes a mismatched local surrogate and selected keep012 instead of the known recoverable keep013.

Stage targeted: architecture compression structure selection.

Mechanism hypothesis: a small fixed budget of independent hard-subnet repair followed by held-out teacher-relative evaluation preserves the ordering of long-repair recoverability; discrete relative-reward updates can then optimize this quantity without soft/hard mismatch or shared-weight favoritism.

Causal intervention and controls: first calibrate repair budgets 0/5/10/25/50 on all four hard masks using identical initialization, optimizer, batches, data split, and seed. This four-mask pass is ground-truth calibration only. The subsequent optimizer will sample hard candidates and update a categorical architecture distribution from paired relative rewards.

Data-access status: teacher rollout only; all rollouts included; no original demonstrations, expert actions, success filtering, rewards, or mode labels. Episode IDs with residue 2 modulo 10 are held out from repair and used only for proxy evaluation.

Metrics and go/no-go: label-free Endpoint CVaR, paired short-trajectory error, and bidirectional set distance. Go to the discrete optimizer only if a short budget ranks keep013 above keep012 and keep123 last at two adjacent budgets; otherwise revise the short-repair estimator.

Resource/time check: four idle V100-32GB GPUs and approximately 4h15 allocation remaining; the 50-epoch calibration is resumable and well inside the available window.

Expected interpretations: stable early ordering validates short repair as a scalable architecture reward; unstable ordering means the estimator is too noisy/short; stable wrong ordering means the proxy or no-reward repair protocol does not predict the known long-repair target.
