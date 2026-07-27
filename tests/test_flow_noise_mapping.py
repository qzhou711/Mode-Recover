import numpy as np
import torch

from analyze_flow_noise_mapping import compute_metrics, pairwise_distance_correlation


def test_identical_mapping_has_unit_geometry_metrics():
    teacher = np.random.RandomState(0).randn(2, 32, 5, 2)
    metrics = compute_metrics(teacher, teacher.copy())
    for row in metrics:
        assert abs(row["variance_retention"] - 1.0) < 1e-10
        assert row["paired_mse"] == 0.0
        assert abs(row["centered_cosine"] - 1.0) < 1e-10
        assert abs(row["pairwise_distance_correlation"] - 1.0) < 1e-10


def test_collapsed_mapping_has_zero_variance_retention():
    teacher = np.random.RandomState(1).randn(1, 32, 5, 2)
    student = np.repeat(teacher.mean(axis=1, keepdims=True), 32, axis=1)
    row = compute_metrics(teacher, student)[0]
    assert row["variance_retention"] < 1e-12
    assert row["pairwise_distance_correlation"] == 0.0


def test_distance_correlation_is_invariant_to_translation_and_scale():
    teacher = np.random.RandomState(2).randn(64, 10)
    student = 3.0 * teacher + 5.0
    assert abs(pairwise_distance_correlation(teacher, student) - 1.0) < 1e-10
