Long-term objective: a demonstration-free architecture-compression selector that supports arbitrary fixed Top-K depth in large multimodal robot/WAM policies.

Short-term bottleneck: the successful four-way deletion categorical optimizer must be generalized to one per-layer score vector and exact-K subsets without reintroducing shared-supernet or straight-through bias.

Stage targeted: architecture compression structure selection.

Mechanism hypothesis: paired one-swap comparisons after independent short repair provide low-variance marginal recoverability credit, allowing exact-K stochastic subset scores to recover the best hard subset.

Causal intervention and controls: P1 tests L=4,K=3 with four seeds and identical settings to the validated discrete optimizer. Only the parameterization changes to generic keep-layer Top-K scores and one-swap pairs. P2 starts only if P1 passes; it first enumerates all six L=4,K=2 masks solely for calibration, then tests whether the generic optimizer finds the calibrated best Top-2 subset.

Data-access status: teacher rollout only; no original demonstrations, expert actions, rewards, success filtering, or mode labels; repair and outer evaluation use disjoint episodes.

Metrics and go/no-go: P1 passes if >=3/4 seeds select keep013. P2's short-repair ground truth must be stable across the proxy metrics; search passes if >=3/4 seeds select the same best Top-2 mask. Search scores alone do not establish closed-loop performance.

Resource/time check: four idle V100-32GB GPUs and approximately 3h13 allocation remaining at launch; both phases save resumable per-round state.

Expected interpretations: P1 success establishes equivalence to the deletion special case; P2 success demonstrates arbitrary-K functionality beyond a trivial complement choice; inconsistent contextual swap preferences diagnose violation of the single-score ordering assumption.
