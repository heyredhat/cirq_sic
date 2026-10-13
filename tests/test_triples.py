"""Triple conventions, physical cycle tests, and state-space reconstruction."""

from itertools import combinations_with_replacement

import cirq
import numpy as np
import pytest

from cirq_sic.measurements import sic_povm
from cirq_sic.triples import (
    assemble_triple_products,
    make_icosphere,
    make_sum_zero_section,
    probability_spectrahedron_section_2D,
    probability_spectrahedron_section_3D,
    section_coordinates,
    sic_state_constraints,
    sic_state_moments,
    triple_product_circuit,
    triple_product_estimate,
    triple_product_state_space,
    triple_products,
)


def control_probabilities(circuit):
    result = cirq.DensityMatrixSimulator(dtype=np.complex128).simulate(
        circuit, qubit_order=sorted(circuit.all_qubits())
    )
    diagonal = np.diag(result.final_density_matrix).real
    return diagonal.reshape(2, -1).sum(axis=1)


@pytest.mark.parametrize("component", ["real", "imag"])
@pytest.mark.parametrize("dimension", [2, 3, 5])
def test_cycle_test_complex_sign_and_non_power_of_two_embedding(component, dimension):
    rng = np.random.default_rng(111)
    states = []
    for _ in range(3):
        ket = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
        states.append(ket / np.linalg.norm(ket))
    expected = (
        np.vdot(states[0], states[1])
        * np.vdot(states[1], states[2])
        * np.vdot(states[2], states[0])
    )
    circuit = triple_product_circuit(states, component=component, measure=False)
    # State-vector simulation avoids the 2^(2*qubits) density-matrix cost.
    state = (
        cirq.Simulator(dtype=np.complex128)
        .simulate(circuit, qubit_order=sorted(circuit.all_qubits()))
        .final_state_vector
    )
    p = (np.abs(state) ** 2).reshape(2, -1).sum(axis=1)
    assert triple_product_estimate(p) == pytest.approx(
        getattr(expected, component), abs=1e-12
    )


def test_nonzero_imaginary_sign_known_example():
    states = [
        np.array([1, 0]),
        np.array([1, 1]) / np.sqrt(2),
        np.array([1, 1j]) / np.sqrt(2),
    ]
    tensor = triple_products(states)
    assert tensor[0, 1, 2] == pytest.approx((1 + 1j) / 4)
    for component in ("real", "imag"):
        probabilities = control_probabilities(
            triple_product_circuit(states, component=component, measure=False)
        )
        assert triple_product_estimate(probabilities) == pytest.approx(0.25)


def test_mixed_cycle_test_uses_purification_and_measures_correct_trace():
    states = [
        np.diag([0.75, 0.25]),
        np.array([[0.5, 0.3], [0.3, 0.5]]),
        np.array([[0.5, -0.2j], [0.2j, 0.5]]),
    ]
    expected = np.trace(states[0] @ states[1] @ states[2])
    for component in ("real", "imag"):
        circuit = triple_product_circuit(states, component=component, measure=False)
        assert len(circuit.all_qubits()) == 7
        state = cirq.Simulator(dtype=np.complex128).simulate(circuit).final_state_vector
        p = (abs(state) ** 2).reshape(2, -1).sum(axis=1)
        assert triple_product_estimate(p) == pytest.approx(
            getattr(expected, component), abs=1e-12
        )


def test_sampled_cycle_test_is_seeded_and_key_is_explicit():
    states = [
        np.array([1, 0]),
        np.array([1, 1]) / np.sqrt(2),
        np.array([1, 1j]) / np.sqrt(2),
    ]
    circuit = triple_product_circuit(states, component="imag", key="test")
    result = cirq.Simulator(seed=123).run(circuit, repetitions=10000)
    p = np.bincount(result.measurements["test"].ravel(), minlength=2) / 10000
    assert triple_product_estimate(p) == pytest.approx(0.25, abs=0.04)


def test_tensor_assembly_preserves_complex_orientation_and_rejects_missing_entries():
    projectors = sic_povm(2).effects * 2
    exact = triple_products(projectors)
    triplets = list(combinations_with_replacement(range(4), 3))
    estimates = [exact[t] for t in triplets]
    assembled = assemble_triple_products(estimates, triplets, 4)
    np.testing.assert_allclose(assembled, exact, atol=1e-14)
    assert abs(assembled[0, 1, 2].imag) > 0.01
    estimates[0] += 0.1j  # Sampling noise on theoretically real repeated indices.
    np.testing.assert_allclose(
        assemble_triple_products(estimates, triplets, 4), exact, atol=1e-14
    )
    with pytest.raises(ValueError, match="complete tensor"):
        assemble_triple_products(estimates[:-1], triplets[:-1], 4)


@pytest.mark.parametrize("d", [2, 3, 4])
def test_triples_reconstruct_purity_cubic_and_full_positivity(d):
    measurement = sic_povm(d)
    projectors = measurement.effects * d
    tensor = triple_products(projectors)
    rng = np.random.default_rng(219)
    for pure in (True, False):
        amplitudes = rng.normal(size=(d, 1 if pure else d)) + 1j * rng.normal(
            size=(d, 1 if pure else d)
        )
        rho = amplitudes @ amplitudes.conj().T
        rho /= np.trace(rho)
        p = measurement.probabilities(rho)
        moments = sic_state_moments(p, tensor, d)
        assert moments["purity"] == pytest.approx(np.trace(rho @ rho).real, abs=1e-7)
        assert moments["cubic"] == pytest.approx(
            np.trace(rho @ rho @ rho).real, abs=1e-7
        )
        constraints = sic_state_constraints(p, tensor, d, atol=1e-7)
        assert constraints["is_physical"]
        assert constraints["is_pure"] == pure
        for real in (True, False):
            matrix = np.einsum(
                "i,ijk->jk", p, triple_product_state_space(tensor, d, real=real)
            )
            assert np.linalg.eigvalsh(matrix).min() > -1e-7
    # A simplex vertex is not a quantum SIC probability vector for d>1.
    impossible = np.eye(d * d)[0]
    assert not sic_state_constraints(impossible, tensor, d)["is_physical"]


def test_positivity_pencil_agrees_with_direct_reconstruction_for_arbitrary_probabilities():
    d = 3
    measurement = sic_povm(d)
    projectors = measurement.effects * d
    matrices = triple_product_state_space(triple_products(projectors), d)
    rng = np.random.default_rng(59)
    for p in rng.dirichlet(np.ones(d * d), size=30):
        rho = (d + 1) * np.einsum("i,ijk->jk", p, projectors) - np.eye(d)
        pencil = np.einsum("i,ijk->jk", p, matrices)
        assert (np.linalg.eigvalsh(rho).min() >= 0) == (
            np.linalg.eigvalsh(pencil).min() >= 0
        )


def test_section_basis_supports_one_and_three_dimensions_and_rejects_dependence():
    for count in (1, 2, 3):
        basis = make_sum_zero_section(np.eye(4)[:count])
        assert basis.shape == (4, count)
        np.testing.assert_allclose(basis.T @ basis, np.eye(count), atol=1e-14)
        np.testing.assert_allclose(basis.sum(axis=0), 0, atol=1e-14)
        center = np.full(4, 0.25)
        y = np.arange(count) * 0.01
        actual, residual = section_coordinates(center + basis @ y, center, basis)
        np.testing.assert_allclose(actual, y, atol=1e-14)
        assert residual < 1e-14
    with pytest.raises(ValueError, match="independent"):
        make_sum_zero_section([[1, 0, 0], [2, 0, 0]])


def test_complex_hermitian_section_and_3d_constraint_classification():
    # Complex off-diagonal entries must not be discarded in ray shooting.
    matrices = np.array([[[1, 2j], [-2j, 1]], [[1, -2j], [2j, 1]], np.eye(2)])
    center = np.full(3, 1 / 3)
    basis = make_sum_zero_section(np.eye(3)[:2])
    result = probability_spectrahedron_section_2D(
        matrices, center, basis, num_angles=30
    )
    assert "PSD" in result["active_constraint"]
    for p in result["boundary_x"]:
        assert np.linalg.eigvalsh(np.einsum("i,ijk->jk", p, matrices)).min() >= -1e-8
    # PSD radius is infinite for a constant identity pencil: simplex wins.
    result3 = probability_spectrahedron_section_3D(
        np.tile(np.eye(2), (4, 1, 1)),
        np.full(4, 0.25),
        make_sum_zero_section(np.eye(4)[:3]),
        subdivisions=0,
    )
    assert set(result3["active"]) == {"simplex"}
    vertices, faces = make_icosphere(1)
    assert vertices.shape == (42, 3)
    assert faces.shape == (80, 3)
    np.testing.assert_allclose(np.linalg.norm(vertices, axis=1), 1)


def test_section_boundary_and_plotting_interfaces_without_display():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cirq_sic.geometry import plot_section_2D_matplotlib, plot_section_3D_matplotlib

    measurement = sic_povm(2)
    matrices = triple_product_state_space(triple_products(measurement.effects * 2), 2)
    center = np.full(4, 0.25)
    section2 = probability_spectrahedron_section_2D(
        matrices, center, make_sum_zero_section(np.eye(4)[:2]), num_angles=24
    )
    section2["marked_probabilities"] = [center]
    section2["labels"] = ["mixed"]
    figure2 = plot_section_2D_matplotlib([section2], ["qubit"], show=False)
    assert len(figure2.axes[0].texts) == 1
    section3 = probability_spectrahedron_section_3D(
        matrices, center, make_sum_zero_section(np.eye(4)[:3]), subdivisions=0
    )
    figure3 = plot_section_3D_matplotlib([section3], ["qubit"], show=False)
    assert len(figure3.axes[0].collections) == 1
    plt.close(figure2)
    plt.close(figure3)


def test_section_plotly_optional_interface():
    pytest.importorskip("plotly")
    from cirq_sic.geometry import plot_section_3D_plotly

    section = probability_spectrahedron_section_3D(
        np.tile(np.eye(2), (4, 1, 1)),
        np.full(4, 0.25),
        make_sum_zero_section(np.eye(4)[:3]),
        subdivisions=0,
    )
    figure = plot_section_3D_plotly([section], ["simplex"], show=False)
    assert len(figure.data) == 2
    # The icosahedron has 30 undirected edges, with three positions per edge.
    assert len(figure.data[1].x) == 90


def test_section_validation_and_projection_residual():
    center = np.full(4, 0.25)
    basis = make_sum_zero_section(np.eye(4)[:2])
    with pytest.raises(ValueError, match="do not lie"):
        section_coordinates([0, 1, 0, 0], center, basis[:, :1])
    _, residual = section_coordinates(
        [1, 0, 0, 0], center, basis[:, 1:], require_in_section=False
    )
    assert residual > 0.1
    with pytest.raises(ValueError, match="independent columns"):
        probability_spectrahedron_section_2D(
            np.tile(np.eye(2), (4, 1, 1)), center, np.zeros((4, 2))
        )
    with pytest.raises(ValueError, match="positive definite"):
        probability_spectrahedron_section_2D(np.zeros((4, 2, 2)), center, basis)
