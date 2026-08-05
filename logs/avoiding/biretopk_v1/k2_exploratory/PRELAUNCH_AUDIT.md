Long-term objective: scalable demonstration-free exact-Top-K architecture selection for multimodal robot/WAM policies.

Short-term bottleneck: L4K2 local velocity and generated-behavior proxies disagree, and conditional set metrics show seed sensitivity.

Stage targeted: exploratory architecture-search stability, not final structure validation.

Mechanism hypothesis: if Top-2 capacity still admits a clearly recoverable subset, paired one-swap preference learning may converge consistently despite noisy individual set metrics; divergent seeds would instead expose an underidentified reward/capacity regime.

Causal intervention and controls: four independent BiReTopK seeds, identical uniform layer scores, K=2, 40 rounds, paired 5-epoch independent repair, and identical proxy-majority rule within each run.

Data-access status: teacher rollout only; no demonstrations, expert actions, rewards, success filtering, or mode labels; train and held-out episodes are disjoint.

Metrics and go/no-go: report selected masks, inclusion probabilities, pairwise preference histories, and cross-seed agreement. Consistency is the primary outcome. No selected mask is called optimal because Top-2 ground truth is unresolved; any candidate requires closed-loop evaluation.

Resource/time check: four idle V100-32GB GPUs with approximately 2h48 allocation remaining; four 40-round jobs fit and are resumable.

Expected interpretations: consistent keep02 supports endpoint/trajectory preference dominance; consistent keep03 supports local-dynamics/coverage dominance; split convergence indicates reward conflict or insufficient Top-2 capacity and blocks stronger claims.
