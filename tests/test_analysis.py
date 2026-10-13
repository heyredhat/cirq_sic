"""Scientific regression tests for independent array analysis."""

import numpy as np
import pytest

from cirq_sic.analysis import (
    born_rule_diagnostics,
    born_rule_prediction,
    characteristic_function,
    entropy_from_probabilities,
    linear_inversion,
    operator_basis,
    project_density_matrix,
    sic_reconstruction,
    stabilizer_distribution,
    stabilizer_renyi_entropy,
    wh_entropy_from_overlaps,
)
from cirq_sic.measurements import sic_povm, wh_measurement


def random_density(d, seed=17):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    rho = matrix @ matrix.conj().T
    return rho / np.trace(rho)


@pytest.mark.parametrize("dimension", [2, 3, 4, 5])
def test_tomography_reconstructs_complex_mixed_states_and_batches(dimension):
    measurement = sic_povm(dimension)
    rho = random_density(dimension)
    p = measurement.probabilities(rho)
    np.testing.assert_allclose(linear_inversion(p, measurement.effects), rho, atol=1e-7)
    batch = np.stack([p, measurement.probabilities(np.eye(dimension) / dimension)])
    expected = np.stack([rho, np.eye(dimension) / dimension])
    np.testing.assert_allclose(
        sic_reconstruction(batch, measurement.effects * dimension), expected, atol=1e-7
    )
    np.testing.assert_allclose(
        linear_inversion(batch, measurement.effects), expected, atol=1e-7
    )


def test_tomography_rejects_incomplete_and_unnormalized_data():
    basis = np.array([np.diag([1, 0]), np.diag([0, 1])])
    with pytest.raises(ValueError, match="informationally complete"):
        linear_inversion([0.5, 0.5], basis)
    with pytest.raises(ValueError, match="sum to one"):
        linear_inversion([0.4, 0.4], basis)
    with pytest.raises(ValueError, match="positive semidefinite"):
        linear_inversion([0.5, 0.5], [np.diag([1.1, 0]), np.diag([-0.1, 1])])


def test_projection_is_euclidean_and_does_not_change_input():
    # Clipping and renormalizing would produce [0, .2/1.2, 1/1.2].
    original = np.diag([-0.2, 0.2, 1.0])
    result = project_density_matrix(original)
    np.testing.assert_allclose(np.diag(result), [0, 0.1, 0.9], atol=1e-14)
    np.testing.assert_array_equal(np.diag(original), [-0.2, 0.2, 1.0])
    batch = project_density_matrix(np.stack([original, np.eye(3) / 3]))
    np.testing.assert_allclose(np.trace(batch, axis1=-2, axis2=-1), 1)
    assert np.linalg.eigvalsh(batch).min() >= 0


@pytest.mark.parametrize("alpha", [1, 2, 3])
def test_stabilizer_states_have_zero_pauli_entropy(alpha):
    zero = np.array([1, 0])
    plus_i = np.array([1, 1j]) / np.sqrt(2)
    bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
    for state in (zero, plus_i, bell):
        assert stabilizer_renyi_entropy(state, alpha=alpha) == pytest.approx(
            0, abs=1e-12
        )


def test_t_state_known_entropy_and_large_alpha_stability():
    state = np.array([1, np.exp(1j * np.pi / 4)]) / np.sqrt(2)
    assert stabilizer_renyi_entropy(state) == pytest.approx(np.log2(4 / 3))
    assert np.isfinite(stabilizer_renyi_entropy(state, alpha=10000))


def test_mixed_purity_correction_and_not_a_mixed_magic_monotone():
    for d in (2, 4):
        assert stabilizer_renyi_entropy(np.eye(d) / d, mixed=True) == pytest.approx(0)
    rho = np.diag([0.75, 0.25])
    # A convex mixture of stabilizer states still has positive diagnostic.
    expected = -np.log2((1 + 0.5**4) / (1 + 0.5**2))
    assert stabilizer_renyi_entropy(rho, mixed=True) == pytest.approx(expected)
    with pytest.raises(ValueError, match="mixed=True"):
        stabilizer_renyi_entropy(rho)
    xi = stabilizer_distribution(rho)
    np.testing.assert_allclose(xi, [0.8, 0, 0, 0.2])
    assert xi.sum() == pytest.approx(1)
    measurement = sic_povm(2)
    assert entropy_from_probabilities(
        measurement.probabilities(rho), measurement.effects
    ) == pytest.approx(expected)


def test_wh_and_pauli_are_explicitly_distinct_and_parseval_holds():
    rho = random_density(4)
    for basis in ("wh", "pauli"):
        operators = operator_basis(4, basis=basis)
        gram = np.einsum("aij,bij->ab", operators.conj(), operators)
        np.testing.assert_allclose(gram, 4 * np.eye(16), atol=1e-12)
        chi = characteristic_function(rho, basis=basis)
        assert np.sum(np.abs(chi) ** 2) == pytest.approx(4 * np.trace(rho @ rho).real)
    assert (
        abs(
            stabilizer_renyi_entropy(rho, basis="wh", mixed=True)
            - stabilizer_renyi_entropy(rho, basis="pauli", mixed=True)
        )
        > 1e-4
    )
    with pytest.raises(ValueError, match="power-of-two"):
        operator_basis(3)
    assert stabilizer_renyi_entropy([1, 0, 0], basis="wh") == pytest.approx(0)


def test_entropy_rejects_non_ic_wh_povm():
    measurement = wh_measurement(np.array([1, 0]))
    with pytest.raises(ValueError, match="informationally complete"):
        entropy_from_probabilities(
            measurement.probabilities([1, 0]), measurement.effects
        )


@pytest.mark.parametrize("d", [2, 3, 4])
def test_qbist_born_rule_equals_direct_ground_measurement(d):
    measurement = sic_povm(d)
    rho = random_density(d)
    p = measurement.probabilities(rho)
    projectors = measurement.effects * d
    conditional = np.real(np.diagonal(projectors, axis1=1, axis2=2)).T
    ground = np.diag(rho).real
    predicted = born_rule_prediction(p, conditional, dimension=d)
    np.testing.assert_allclose(predicted, ground, atol=1e-7)
    reference = np.einsum("aij,bji->ab", measurement.effects, projectors).real
    np.testing.assert_allclose(
        born_rule_prediction(p, conditional, reference=reference), ground, atol=1e-7
    )
    report = born_rule_diagnostics(
        p,
        conditional,
        ground,
        dimension=d,
        sky_shots=1000,
        ground_shots=2000,
        conditional_shots=np.full(d * d, 3000),
    )
    assert report["l2_error"] < 1e-7
    assert np.all(report["standard_error"] > 0)
    np.testing.assert_allclose(report["covariance"].sum(axis=0), 0, atol=1e-14)
    assert np.linalg.eigvalsh(report["covariance"]).min() >= -1e-14


def test_born_rule_preserves_negative_predictions_and_rejects_singular_calibration():
    conditional = np.array([[1, 0, 0, 0], [0, 1, 1, 1]])
    p = [0, 1 / 3, 1 / 3, 1 / 3]
    predicted = born_rule_prediction(p, conditional, dimension=2)
    np.testing.assert_allclose(predicted, [-0.5, 1.5])
    report = born_rule_diagnostics(p, conditional, [0, 1], dimension=2)
    assert report["negative_predicted_mass"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="rank deficient"):
        born_rule_prediction(p, conditional, reference=np.full((4, 4), 0.25))


def test_born_covariance_matches_scalar_binomial_propagation():
    p = np.full(4, 0.25)
    c = np.array([[0.9, 0.6, 0.2, 0.3], [0.1, 0.4, 0.8, 0.7]])
    q = np.array([0.5, 0.5])
    report = born_rule_diagnostics(
        p, c, q, dimension=2, sky_shots=1000, ground_shots=1000
    )
    derivative = c @ (3 * np.eye(4) - np.ones((4, 4)) / 2)
    expected_variance = (
        0.25 / 1000
        + (np.dot(p, derivative[0] ** 2) - np.dot(p, derivative[0]) ** 2) / 1000
    )
    assert report["covariance"][0, 0] == pytest.approx(expected_variance)


@pytest.mark.parametrize("dimension", [2, 3, 4, 5])
@pytest.mark.parametrize("pure", [True, False])
def test_direct_wh_overlap_entropy_parseval_and_purity(dimension, pure):
    rho = random_density(dimension)
    if pure:
        _, vectors = np.linalg.eigh(rho)
        rho = np.outer(vectors[:, -1], vectors[:, -1].conj())
    displacement = operator_basis(dimension, basis="wh")
    overlaps = np.array(
        [np.trace(rho @ op @ rho @ op.conj().T).real / dimension for op in displacement]
    )
    chi = characteristic_function(rho, basis="wh")
    assert overlaps.sum() == pytest.approx(1)
    assert overlaps[0] == pytest.approx(np.trace(rho @ rho).real / dimension)
    assert np.sum(overlaps**2) == pytest.approx(
        np.sum(abs(chi) ** 4) / dimension**2
    )
    expected = stabilizer_renyi_entropy(rho, basis="wh", mixed=True)
    assert wh_entropy_from_overlaps(overlaps, check_physical=True) == pytest.approx(
        expected
    )
    np.testing.assert_allclose(
        wh_entropy_from_overlaps(np.stack([overlaps, overlaps])),
        [expected, expected],
        atol=1e-13,
    )


def test_wh_overlap_entropy_validates_domain_and_preserves_sampling_noise():
    assert wh_entropy_from_overlaps(np.full(9, 1 / 9)) == pytest.approx(0)
    # Finite samples can put identity below its physical lower endpoint.
    noisy = np.array([0.24, 0.25, 0.26, 0.25])
    assert wh_entropy_from_overlaps(noisy) < 0
    with pytest.raises(ValueError, match="purity bounds"):
        wh_entropy_from_overlaps(noisy, check_physical=True)
    with pytest.raises(ValueError, match="identity overlap"):
        wh_entropy_from_overlaps([0, 0.25, 0.25, 0.5])
    with pytest.raises(ValueError, match="square length"):
        wh_entropy_from_overlaps([0.5, 0.5])
