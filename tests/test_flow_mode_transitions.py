import numpy as np

from merge_flow_paired_rollouts import (
    hierarchical_classes,
    information_metrics,
    mode_classes,
    transition_matrix,
    valid_mode_codes,
)


def test_valid_mode_code_count_is_24():
    codes = valid_mode_codes()
    assert len(codes) == 24
    assert len(set(codes)) == 24


def test_failures_map_to_failure_class():
    successes = np.asarray([True, False])
    modes = np.zeros((2, 9), dtype=np.int8)
    modes[0, [0, 2, 5]] = 1
    classes = mode_classes(successes, modes)
    assert classes[0] == 0
    assert classes[1] == 24


def test_identity_transition_has_unit_nmi_and_zero_conditional_entropy():
    classes = np.arange(25)
    matrix = transition_matrix(classes, classes, 25)
    information = information_metrics(matrix)
    assert abs(information["normalized_mutual_information"] - 1.0) < 1e-12
    assert abs(information["student_given_teacher_entropy"]) < 1e-12


def test_hierarchical_classes_encode_prefixes_and_failure():
    successes = np.asarray([True, True, False])
    modes = np.zeros((3, 9), dtype=np.int8)
    modes[0, [0, 2, 5]] = 1
    modes[1, [1, 4, 8]] = 1
    assert np.array_equal(hierarchical_classes(successes, modes, 1), [0, 1, 2])
    assert np.array_equal(hierarchical_classes(successes, modes, 2), [0, 5, 6])
