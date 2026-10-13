"""Regression tests for reusable numerical helpers and their boundary cases."""

import numpy as np
import pytest

from cirq_sic.utils import (
    bloch_vector,
    dirac,
    geodesic_interpolator,
    gram_matrix,
    kron,
    merge_dicts,
    mod_d_outcome_mask,
    mod_d_probabilities,
    nonneg_projection,
    pad,
    ptrace,
    rand_dm,
    rand_ket,
    symmetrize,
)


def test_random_state_generators_are_reproducible_and_physical():
    a = rand_ket(5, rng=np.random.default_rng(8))
    b = rand_ket(5, rng=np.random.default_rng(8))
    assert np.array_equal(a, b)
    assert np.isclose(np.vdot(a, a), 1)
    rho = rand_dm(5, r=3, rng=np.random.default_rng(8))
    assert np.allclose(rho, rho.conj().T)
    assert np.isclose(np.trace(rho), 1)
    assert np.min(np.linalg.eigvalsh(rho)) > -1e-14
    assert np.linalg.matrix_rank(rho, tol=1e-12) == 3
    with pytest.raises(ValueError, match="r must not exceed"):
        rand_dm(2, r=3)


@pytest.mark.parametrize("dimension", [0, -1, 1.5])
def test_random_state_rejects_invalid_dimensions(dimension):
    with pytest.raises(ValueError):
        rand_ket(dimension)


def test_partial_trace_preserves_factor_order_and_matrix_shape():
    states = [rand_dm(d, r=d, rng=np.random.default_rng(d)) for d in (2, 3, 4)]
    rho = kron(*states)
    assert np.allclose(ptrace(rho, [1], (2, 3, 4)), kron(states[0], states[2]))
    assert np.allclose(ptrace(rho, [2, 0], [2, 3, 4]), states[1])
    assert np.allclose(ptrace(rho, [], [2, 3, 4]), rho)
    assert np.allclose(ptrace(rho, [0, 1, 2], [2, 3, 4]), [[1]])
    with pytest.raises(ValueError, match="distinct"):
        ptrace(rho, [0, 0], [2, 3, 4])
    with pytest.raises(ValueError, match="shape"):
        ptrace(rho, [0], [2, 3])


def test_partial_trace_entangled_state():
    bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
    assert np.allclose(ptrace(np.outer(bell, bell), [1], [2, 2]), np.eye(2) / 2)


def test_nonnegative_normalization_does_not_mutate_or_hide_zero_mass():
    probabilities = np.array([-0.2, 0.4, 0.8])
    original = probabilities.copy()
    assert np.allclose(nonneg_projection(probabilities), [0, 1 / 3, 2 / 3])
    assert np.array_equal(probabilities, original)
    assert np.allclose(nonneg_projection([-1, 2]), [0, 1])
    for invalid in ([0, 0], [-1, -2], [np.nan, 1], [[1, 2]]):
        with pytest.raises(ValueError):
            nonneg_projection(invalid)


def test_encoded_outcomes_keep_register_order_and_leakage_mass():
    expected = np.array([a < 3 and b < 3 for a in range(4) for b in range(4)])
    assert np.array_equal(mod_d_outcome_mask(3, 2, 2), expected)
    probabilities = np.arange(16) / 120
    selected = mod_d_probabilities(probabilities, 3, 2, 2)
    assert np.array_equal(selected, probabilities[expected])
    assert selected.sum() < 1
    with pytest.raises(ValueError, match="fit"):
        mod_d_outcome_mask(5, 2, 2)
    with pytest.raises(ValueError, match="shape"):
        mod_d_probabilities([1], 3, 2, 2)


def test_geodesic_vectorization_endpoints_and_global_phase():
    first = np.array([1, 0], dtype=complex)
    last = np.array([1j, 1j]) / np.sqrt(2)
    interpolate = geodesic_interpolator(first, last)
    times = np.linspace(0, 1, 5)
    points = interpolate(times)
    assert points.shape == (5, 2)
    assert np.allclose(np.linalg.norm(points, axis=1), 1)
    assert np.allclose(points[0], first)
    assert np.allclose(
        np.outer(points[-1], points[-1].conj()), np.outer(last, last.conj())
    )
    assert np.allclose(points[:, 0], np.cos(times * np.pi / 4))
    same_ray = geodesic_interpolator(first, -first)
    assert np.allclose(same_ray(times), np.broadcast_to(first, (5, 2)))
    assert np.allclose(same_ray(0.5), first)
    with pytest.raises(ValueError, match="nonzero"):
        geodesic_interpolator([0, 0], [1, 0])
    with pytest.raises(ValueError, match="lie"):
        interpolate(2)


def test_other_numerical_helpers(capsys):
    assert np.array_equal(pad(np.array([1j]), 3), [1j, 0, 0])
    with pytest.raises(ValueError):
        pad([1, 2], 1)
    assert np.array_equal(bloch_vector([1, 0]), [0, 0, 1])
    assert np.array_equal(bloch_vector(np.eye(2) / 2), [0, 0, 0])
    effects = np.array([np.diag([1, 0]), np.diag([0, 1])])
    assert np.array_equal(gram_matrix(effects), np.eye(2))
    assert merge_dicts([{"a": 1}, {"a": 2, "b": 3}]) == {"a": 2, "b": 3}
    assert dirac(np.array([0, 1])) == [("1", 1)]
    assert "1.000" in capsys.readouterr().out
    with pytest.raises(ValueError):
        kron()


def test_symmetrize_validates_and_leaves_input_unchanged():
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    original = matrix.copy()
    symmetric = symmetrize(matrix, T=200)
    assert np.allclose(symmetric.sum(axis=0), 1)
    assert np.allclose(symmetric, symmetric.T)
    assert np.array_equal(matrix, original)
    with pytest.raises(ValueError, match="zero column"):
        symmetrize(np.zeros((2, 2)))
