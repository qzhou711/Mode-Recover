Long-term objective: transferable demonstration-free architecture compression and step distillation for multimodal robot/WAM policies.

Short-term bottleneck: LightDP-style mean-loss soft gating selected keep012 although exhaustive hard-subnet repair showed keep013 is substantially more recoverable.

Stage targeted: architecture compression / cross-architecture transfer-map selection only.

Mechanism hypothesis: mean fidelity and soft supernet evaluation underweight hard states and conditional sample support; exact hard Top-3 search with tail, trajectory, and bidirectional set objectives will rank recoverable subnets better.

Causal intervention and controls: four paired objectives (mean, CVaR, CVaR+trajectory, CVaR+trajectory+set); identical teacher, buffer, episode split, seed, training budget, optimizer, and hard Top-3 mechanism. Only the held-out gate objective changes.

Data-access status: teacher rollout only; no original demonstrations, expert actions, environment rewards, success filtering, or D3IL mode labels. Train and gate-validation episodes are disjoint.

Metrics and go/no-go: primary screening outcome is the selected hard layer set. Go if enhanced objectives select known-recoverable keep013 consistently enough to justify multi-seed replication. Any selected model must undergo identical repair and Standard-120 screening, then multi-seed Standard-480 confirmation before a claim.

Resource/time check: four idle V100-32GB GPUs, approximately 4h44 allocation remaining at launch, sufficient for four parallel 250-epoch searches and a recoverable checkpoint milestone.

Expected interpretations: keep013 selection supports the transferable proxy hypothesis; persistent keep012 selection implicates gate optimization/soft-hard credit assignment; unstable selections implicate estimator variance; a different selection requires identical hard-subnet repair before interpretation.
