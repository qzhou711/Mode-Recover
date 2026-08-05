Long-term objective: transferable demonstration-free cross-architecture compression for multimodal robot/WAM policies, followed separately by step distillation.

Short-term bottleneck: equal-initialized shared-supernet straight-through gates still selected keep012 because they did not optimize independent post-repair recoverability.

Stage targeted: architecture compression structure selection.

Mechanism hypothesis: a categorical distribution updated from paired, independent hard-subnet short-repair preferences will converge to the truly recoverable mask without SVD priors, shared-weight favoritism, or soft/hard gradient mismatch.

Causal intervention and controls: four search seeds start from exactly uniform deletion probabilities. Each round samples two distinct hard masks, initializes both from the same frozen teacher, gives both the same 5-epoch minibatch/time schedule, and compares them on the same held-out states/noises. Only the search seed differs across replicas.

Data-access status: teacher rollout only; all rollouts included; no demonstrations, expert actions, environment rewards, success filtering, or mode labels. Repair and outer evaluation use disjoint episodes.

Metrics and go/no-go: Endpoint-CVaR is the primary preference; short-trajectory error and bidirectional conditional-set distances provide scale-free majority confirmation. Go if at least 3/4 seeds delete layer 2 (keep013) with final probability >=0.8. Do not infer closed-loop quality from the search score alone.

Resource/time check: four idle V100-32GB GPUs and approximately 3h43 allocation remaining before smoke; each round saves resumable optimizer/history state.

Expected interpretations: convergence to delete-2 validates discrete post-repair preference learning; inconsistent seeds indicate insufficient exploration or noisy short repair; systematic keep012 indicates that online sampling protocol differs from calibration; early wrong one-hot collapse indicates an annealing/exploration bug.
