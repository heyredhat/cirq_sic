"""Independent analytic checks for the circuit-noise geometry pipeline."""

from dataclasses import replace

import cirq
import numpy as np
import pytest

from cirq_sic.noise import (
    LocalMomentNoise,
    simulate_triple_tensor,
    simulate_sic_tomography,
    triple_tensor_diagnostics,
    compare_state_spaces,
    tomography_cloud_metrics,
)
from cirq_sic.measurements import sic_povm
from cirq_sic.triples import triple_products


@pytest.fixture(scope="module")
def reference():
    return triple_products(sic_povm(2).effects * 2).real


@pytest.fixture(scope="module")
def states():
    rng = np.random.default_rng(19)
    k = rng.normal(size=(100, 2)) + 1j * rng.normal(size=(100, 2))
    return k / np.linalg.norm(k, axis=1, keepdims=True)


def test_ideal_circuits_and_geometry(reference):
    result = simulate_triple_tensor()
    np.testing.assert_allclose(result.raw_tensor, reference, atol=2e-14)
    diagnostics = triple_tensor_diagnostics(result.raw_tensor, 2, reference=reference)
    assert diagnostics["closure_rms"] < 1e-13
    geometry = compare_state_spaces(result.tensor, reference, 2, directions=100)
    np.testing.assert_allclose(geometry["radii"], 1 / np.sqrt(12), atol=1e-13)
    assert geometry["relative_volume"] == pytest.approx(1)
    assert geometry["jaccard"] == pytest.approx(1)


@pytest.mark.parametrize(
    "channel",
    [
        cirq.depolarize(0.05),
        cirq.bit_flip(0.05),
        cirq.phase_flip(0.05),
        cirq.phase_damp(0.05),
        cirq.amplitude_damp(0.05),
        cirq.rz(0.4),
    ],
)
def test_preparation_noise_matches_independent_kraus_traces(channel, reference, states):
    m = sic_povm(2)

    def apply(rho):
        return sum(k @ rho @ k.conj().T for k in cirq.kraus(channel))

    noisy = np.array([apply(rho) for rho in m.effects * 2])
    result = simulate_triple_tensor(preparation_channel=channel, ordered=False)
    np.testing.assert_allclose(result.tensor, triple_products(noisy).real, atol=2e-14)
    cloud = simulate_sic_tomography(states[:5], preparation_channel=channel)
    expected = [m.probabilities(apply(np.outer(s, s.conj()))) for s in states[:5]]
    np.testing.assert_allclose(cloud["exact"], expected, atol=2e-14)


def test_depolarizing_geometry_has_different_volume_from_tomography(reference, states):
    rate = 0.05
    lam = 1 - 4 * rate / 3
    result = simulate_triple_tensor(
        preparation_channel=cirq.depolarize(rate), ordered=False
    )
    geometry = compare_state_spaces(result.tensor, reference, 2, directions=100)
    radius = np.sqrt(3 * lam**2 - 2) / lam**2
    assert geometry["relative_volume"] == pytest.approx(radius**3)
    cloud = simulate_sic_tomography(states, preparation_channel=cirq.depolarize(rate))
    metrics = tomography_cloud_metrics(cloud["exact"], cloud["ideal"], result.tensor, 2)
    assert metrics["hull_relative_to_ideal_cloud"] == pytest.approx(lam**3)
    assert metrics["nominal_physical_fraction"] == 1
    assert metrics["hull_relative_to_ideal_cloud"] != pytest.approx(
        geometry["relative_volume"]
    )


def test_common_unitary_invisible_to_triples_but_not_paired_tomography(
    reference, states
):
    r = simulate_triple_tensor(preparation_channel=cirq.rz(0.4), ordered=False)
    np.testing.assert_allclose(r.tensor, reference, atol=2e-14)
    c = simulate_sic_tomography(states, preparation_channel=cirq.rz(0.4))
    m = tomography_cloud_metrics(c["exact"], c["ideal"], r.tensor, 2)
    assert m["paired_tv_mean"] > 0.02
    assert m["hull_relative_to_ideal_cloud"] == pytest.approx(1)


def test_readout_attenuation_and_center_failure(reference):
    rate = 0.05
    scale = 1 - 2 * rate
    result = simulate_triple_tensor(readout_error=rate, ordered=False)
    np.testing.assert_allclose(result.tensor, scale * reference, atol=2e-14)
    g = compare_state_spaces(result.tensor, reference, 2, directions=100)
    assert g["relative_volume"] == pytest.approx(((3 * scale - 2) / scale) ** 3)
    assert (
        compare_state_spaces(0.5 * reference, reference, 2)["status"]
        == "center_not_strictly_feasible"
    )
    assert "relative_volume" not in compare_state_spaces(0.5 * reference, reference, 2)


def test_sampling_seed_bootstrap_and_symmetry(reference):
    a = simulate_triple_tensor(shots=500, seed=123)
    b = simulate_triple_tensor(shots=500, seed=123)
    np.testing.assert_array_equal(a.counts_zero, b.counts_zero)
    np.testing.assert_allclose(a.standard_errors, np.sqrt((1 - a.estimates**2) / 500))
    np.testing.assert_array_equal(a.bootstrap(5, seed=55), a.bootstrap(5, seed=55))
    for t in a.bootstrap(5, seed=55):
        np.testing.assert_allclose(t, t.transpose(2, 0, 1))
    assert triple_tensor_diagnostics(a.raw_tensor, 2)["permutation_rms"] > 0
    # Symmetry-compressed data are copied, never re-sampled per permutation.
    c = simulate_triple_tensor(shots=500, seed=1, ordered=False)
    assert len(c.estimates) == 20
    np.testing.assert_allclose(
        c.bootstrap(1, seed=0)[0], c.bootstrap(1, seed=0)[0].transpose(2, 0, 1)
    )
    with pytest.raises(ValueError, match="finite-shot"):
        replace(a, shots=None).bootstrap()


def test_local_moment_noise_and_targeted_faults():
    q = cirq.LineQubit.range(2)
    noise = LocalMomentNoise(cirq.bit_flip(1), qubits=[q[0]])
    c = cirq.Circuit(cirq.I.on_each(*q))
    result = cirq.DensityMatrixSimulator(noise=noise).simulate(c, qubit_order=q)
    assert result.final_density_matrix[2, 2] == pytest.approx(1)
    r = simulate_triple_tensor(
        noise=LocalMomentNoise(cirq.bit_flip(0.1), qubits=[cirq.LineQubit(1)])
    )
    assert triple_tensor_diagnostics(r.raw_tensor, 2)["permutation_rms"] > 1e-4


def test_readout_cloud_matches_bit_confusion(states):
    cloud = simulate_sic_tomography(states[:4], readout_error=0.1, shots=100, seed=1)
    c = np.array([[0.9, 0.1], [0.1, 0.9]])
    np.testing.assert_allclose(
        cloud["exact"], cloud["ideal"] @ np.kron(c, c).T, atol=2e-14
    )
    np.testing.assert_array_equal(cloud["counts"].sum(axis=1), 100)


def test_singular_tomography_and_invalid_cloud(reference):
    uniform = np.full((10, 4), 0.25)
    result = tomography_cloud_metrics(uniform, uniform, reference, 2)
    assert result["hull_volume"] == 0
    assert np.isnan(result["hull_relative_to_ideal_cloud"])
    with pytest.raises(ValueError, match="probability"):
        tomography_cloud_metrics(uniform * 2, uniform, reference, 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dimension": 3},
        {"shots": 0},
        {"shots": True},
        {"readout_error": -1},
        {"preparation_channel": cirq.CNOT},
    ],
)
def test_invalid_simulation_inputs(kwargs):
    with pytest.raises(ValueError):
        simulate_triple_tensor(**kwargs)


def test_transform_can_compile_and_cannot_measure(reference):
    transform = lambda c: cirq.optimize_for_target_gateset(
        c, gateset=cirq.CZTargetGateset()
    )
    r = simulate_triple_tensor(ordered=False, circuit_transform=transform)
    np.testing.assert_allclose(r.tensor, reference, atol=1e-12)
    with pytest.raises(ValueError, match="measurements"):
        simulate_triple_tensor(
            circuit_transform=lambda c: c
            + cirq.Circuit(cirq.measure(cirq.LineQubit(0)))
        )


def test_amplitude_damping_affine_image(reference, states):
    gamma = 0.1
    c = simulate_sic_tomography(states, preparation_channel=cirq.amplitude_damp(gamma))
    m = tomography_cloud_metrics(c["exact"], c["ideal"], reference, 2)
    np.testing.assert_allclose(
        m["affine_singular_values"],
        [np.sqrt(1 - gamma), np.sqrt(1 - gamma), 1 - gamma],
        atol=1e-13,
    )
    assert m["affine_center_shift"] == pytest.approx(gamma / np.sqrt(12))
    assert m["affine_ball_volume_ratio"] == pytest.approx((1 - gamma) ** 2)
    assert m["affine_fit_rmse"] < 1e-14


def test_resource_guard_and_empty_certificate(reference):
    with pytest.raises(ValueError, match="max_settings"):
        simulate_triple_tensor(8)
    assert triple_tensor_diagnostics(0.5 * reference, 2)["empty_by_trace_certificate"]
    assert not triple_tensor_diagnostics(reference, 2)["empty_by_trace_certificate"]
