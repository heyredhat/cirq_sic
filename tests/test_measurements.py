"""Mathematical and circuit regression tests, especially non-binary dimensions."""

import cirq
import numpy as np
import pytest

from cirq_sic.ansatz import (
    ansatz_angles_to_params,
    ansatz_params_to_angles,
    ansatz_unitary,
    grey,
    ket_to_ansatz_angles,
)
from cirq_sic.circuits import (
    CX_d,
    CZ_d,
    X_d,
    Z_d,
    ansatz_circuit,
    arthurs_kelly_d,
    embed_gate,
    qft_d,
    real_triple_product_from_probabilities,
    simple_wh_povm_d,
)
from cirq_sic.measurements import (
    Measurement,
    bell_overlap_circuit,
    build_measurement_circuit,
    embed_state,
    prepare_state,
    sic_povm,
    wh_measurement,
)
from cirq_sic.sics import (
    all_d2_fiducials,
    frame_potential,
    frame_potential_minimum,
    load_sic_fiducial,
    refine_sic_fiducial,
    minimize_wh_frame_potential,
)
from cirq_sic.wh import change_conjugate_convention, wh_frame, wh_operators, wh_povm


def ket(d, seed=729):
    """Seeded complex state with no accidental real/conjugate symmetry."""
    rng = np.random.default_rng(seed + d)
    state = rng.normal(size=d) + 1j * rng.normal(size=d)
    return state / np.linalg.norm(state)


def density(d, seed=837):
    """Full-rank mixed state, including complex off-diagonal entries."""
    rng = np.random.default_rng(seed + d)
    matrix = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    state = matrix @ matrix.conj().T
    return state / state.trace()


def measured_probabilities(spec):
    """Marginalize the explicit environment without projecting any outcomes."""
    state = (
        cirq.Simulator(dtype=np.complex128)
        .simulate(
            spec.circuit,
            qubit_order=spec.qubit_order,
        )
        .final_state_vector
    )
    return np.sum(
        np.abs(state.reshape(2 ** len(spec.measured_qubits), -1)) ** 2, axis=1
    )


@pytest.mark.parametrize("d", [1, 2, 3, 4, 5, 6, 7, 8])
def test_frame_effects_and_displacement_convention(d):
    phi = ket(d)
    operators = wh_operators(d)
    frame = wh_frame(phi)
    measurement = wh_measurement(phi)
    assert np.allclose(
        operators["X"], operators["F"].conj().T @ operators["Z"] @ operators["F"]
    )
    assert np.allclose(frame @ frame.conj().T, d * np.eye(d))
    assert np.allclose(measurement.effects, wh_povm(phi))
    assert np.allclose(np.sum(measurement.effects, axis=0), np.eye(d))
    assert np.allclose(np.linalg.norm(measurement.states, axis=0), 1)
    for a in range(d):
        for b in range(d):
            expected = (
                np.linalg.matrix_power(operators["X"], a)
                @ np.linalg.matrix_power(operators["Z"], b)
                @ phi
            )
            assert np.allclose(frame[:, a * d + b], expected)
    assert not measurement.frame.flags.writeable
    assert not measurement.effects.flags.writeable


@pytest.mark.parametrize("d", [1, 2, 3, 4, 5, 6, 7, 8])
def test_naimark_columns_are_placed_for_appended_ancillas(d):
    measurement = wh_measurement(ket(d))
    isometry = measurement.isometry()
    unitary = measurement.unitary()
    indices = np.arange(d) * 2**measurement.num_ancilla_qubits
    assert np.allclose(isometry.conj().T @ isometry, np.eye(d))
    assert np.allclose(unitary.conj().T @ unitary, np.eye(len(unitary)))
    assert np.allclose(unitary[:, indices], isometry)
    assert np.count_nonzero(isometry[d * d :]) == 0
    # Every padded-input column is retained in an orthonormal completion.
    assert np.allclose(unitary @ unitary.conj().T, np.eye(len(unitary)))


@pytest.mark.parametrize("d", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("mixed", [False, True])
def test_full_born_distribution_and_zero_ideal_padding(d, mixed):
    measurement = wh_measurement(ket(d, seed=952))
    state = density(d) if mixed else ket(d)
    spec = build_measurement_circuit(measurement, state, measure=False)
    probabilities = measured_probabilities(spec)
    expected = measurement.probabilities(state)
    assert np.allclose(probabilities[: d * d], expected, atol=1e-12)
    assert np.allclose(probabilities[d * d :], 0, atol=1e-12)
    assert np.isclose(probabilities.sum(), 1)
    assert spec.qubit_order == spec.measured_qubits + spec.environment_qubits
    assert len(spec.environment_qubits) == (
        measurement.num_system_qubits if mixed else 0
    )


@pytest.mark.parametrize("d", [2, 3, 5, 6, 7])
def test_sic_refinement_and_linear_inverse(d):
    measurement = sic_povm(d)
    assert measurement.is_sic(atol=1e-9)
    rho = density(d)
    p = measurement.probabilities(rho)
    projectors = measurement.effects * d
    reconstructed = (d + 1) * np.einsum("k,kij->ij", p, projectors) - np.eye(d)
    assert np.allclose(reconstructed, rho, atol=1e-9)
    assert np.isclose(
        frame_potential(measurement.frame, 2), frame_potential_minimum(d), atol=1e-10
    )
    # Raw loading is explicit and stable; only the measurement factory refines.
    assert np.allclose(load_sic_fiducial(d), load_sic_fiducial(d))


@pytest.mark.parametrize("d", [2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("mixed", [False, True])
def test_bell_overlap_all_outcomes_and_purity(d, mixed):
    state = density(d) if mixed else ket(d)
    rho = state if mixed else np.outer(state, state.conj())
    spec = bell_overlap_circuit(state, measure=False)
    probabilities = measured_probabilities(spec)
    operators = wh_operators(d)["D"].reshape(d * d, d, d)
    expected = np.array(
        [np.trace(rho @ op @ rho @ op.conj().T).real / d for op in operators]
    )
    assert np.allclose(probabilities[: d * d], expected, atol=1e-12)
    assert np.allclose(probabilities[d * d :], 0, atol=1e-12)
    assert np.isclose(d * probabilities[0], np.trace(rho @ rho).real, atol=1e-12)
    if not mixed:
        assert np.allclose(
            probabilities[: d * d],
            wh_measurement(state).probabilities(state),
            atol=1e-12,
        )


def test_basis_povm_and_custom_qubit_order_sampling():
    measurement = Measurement(np.eye(3), name="Basis")
    qubits = (cirq.NamedQubit("z"), cirq.NamedQubit("a"))
    spec = measurement.circuit(np.eye(3)[2], system_qubits=qubits, key="basis")
    assert spec.qubit_order == qubits
    result = cirq.Simulator(seed=14).run(spec.circuit, repetitions=30)
    assert result.histogram(key="basis") == {2: 30}
    assert spec.outcome_indices == (0, 1, 2)


def test_mixed_purification_reduced_state():
    rho = density(3)
    system = cirq.LineQubit.range(2)
    environment = cirq.LineQubit.range(2, 4)
    circuit = cirq.Circuit(prepare_state(rho, system, environment_qubits=environment))
    result = cirq.Simulator(dtype=np.complex128).simulate(
        circuit, qubit_order=system + environment
    )
    vector = result.final_state_vector.reshape(4, 4)
    assert np.allclose(vector @ vector.conj().T, embed_state(rho))


@pytest.mark.parametrize("d", [3, 4, 5, 6, 7])
def test_embedded_legacy_operators_on_entire_register(d):
    n = (d - 1).bit_length()
    qubits = cirq.LineQubit.range(n)
    operators = wh_operators(d)
    for build, name in [(X_d, "X"), (Z_d, "Z"), (qft_d, "F")]:
        circuit = cirq.Circuit(build(d, qubits))
        matrix = circuit.unitary(qubit_order=qubits)
        expected = np.eye(2**n, dtype=complex)
        expected[:d, :d] = operators[name]
        assert np.allclose(matrix, expected)
        inverse = cirq.Circuit(cirq.inverse(circuit))
        assert np.allclose(inverse.unitary(qubit_order=qubits) @ matrix, np.eye(2**n))


@pytest.mark.parametrize("d", [3, 5, 6, 7])
@pytest.mark.parametrize("inverse", [False, True])
def test_controlled_qudit_gates_fix_padded_control_and_target(d, inverse):
    n = (d - 1).bit_length()
    qubits = cirq.LineQubit.range(2 * n)
    size = 2**n
    sign = -1 if inverse else 1
    for builder in [CX_d, CZ_d]:
        matrix = cirq.Circuit(
            builder(d, qubits[:n], qubits[n:], inverse=inverse)
        ).unitary(qubit_order=qubits)
        assert np.allclose(matrix.conj().T @ matrix, np.eye(size * size))
        for control in range(size):
            for target in range(size):
                original = control * size + target
                expected = np.eye(size * size, dtype=complex)[:, original]
                if control < d and target < d:
                    if builder is CX_d:
                        expected = np.eye(size * size)[
                            :, control * size + (target + sign * control) % d
                        ]
                    else:
                        expected *= np.exp(sign * 2j * np.pi * control * target / d)
                assert np.allclose(matrix[:, original], expected)


@pytest.mark.parametrize("d", [3, 5, 6, 7])
def test_legacy_simple_povm_embedded_without_auxiliaries(d):
    n = (d - 1).bit_length()
    system = cirq.LineQubit.range(n)
    ancillas = cirq.LineQubit.range(n, 2 * n)
    state, phi = ket(d), ket(d, seed=135)
    circuit = cirq.Circuit(
        ansatz_circuit(state)(system),
        simple_wh_povm_d(
            d, system, ancillas, (), prepare_fiducial=ansatz_circuit(phi), measure=False
        ),
    )
    vector = (
        cirq.Simulator(dtype=np.complex128)
        .simulate(circuit, qubit_order=system + ancillas)
        .final_state_vector
    )
    p = np.abs(vector.reshape(2**n, 2**n)) ** 2
    assert np.allclose(
        p[:d, :d].reshape(-1), wh_measurement(phi).probabilities(state), atol=1e-12
    )
    assert np.allclose(p[d:, :], 0, atol=1e-12)
    assert np.allclose(p[:, d:], 0, atol=1e-12)


@pytest.mark.parametrize("d", [3, 5, 7])
def test_legacy_arthurs_kelly_embedded_without_auxiliaries(d):
    n = (d - 1).bit_length()
    qubits = cirq.LineQubit.range(3 * n)
    state, phi = ket(d), ket(d, seed=846)
    circuit = cirq.Circuit(
        ansatz_circuit(state)(qubits[:n]),
        arthurs_kelly_d(
            d,
            qubits[:n],
            qubits[n : 2 * n],
            qubits[2 * n :],
            (),
            prepare_fiducial=ansatz_circuit(phi),
            measure=False,
        ),
    )
    vector = (
        cirq.Simulator(dtype=np.complex128)
        .simulate(circuit, qubit_order=qubits)
        .final_state_vector
    )
    ancilla_probabilities = np.sum(np.abs(vector.reshape((2**n,) * 3)) ** 2, axis=0)
    p = change_conjugate_convention(ancilla_probabilities[:d, :d].reshape(-1))
    assert np.allclose(p, wh_measurement(phi).probabilities(state), atol=1e-12)
    assert np.allclose(ancilla_probabilities[d:, :], 0, atol=1e-12)
    assert np.allclose(ancilla_probabilities[:, d:], 0, atol=1e-12)


@pytest.mark.parametrize(
    "state", [np.array([0, 1, 0, 0]), np.array([0, 0, 0, 1j]), ket(8), np.r_[ket(3), 0]]
)
def test_gray_ansatz_preserves_zero_branches_and_global_phase(state):
    angles = ket_to_ansatz_angles(state)
    params = ansatz_angles_to_params(*angles)
    n = (len(state) - 1).bit_length()
    assert np.allclose(ansatz_unitary(n, params)[:, 0], state)
    qubits = cirq.LineQubit.range(n)
    prepared = cirq.Circuit(ansatz_circuit(state)(qubits, conjugate=True)).unitary(
        qubit_order=qubits
    )[:, 0]
    assert np.allclose(prepared, state.conj())


def test_conjugate_permutation_and_antipodal_fiducials():
    batch = np.arange(2 * 3 * 9).reshape(2, 3, 9)
    assert np.array_equal(
        change_conjugate_convention(change_conjugate_convention(batch)), batch
    )
    assert np.array_equal(
        change_conjugate_convention(np.arange(9)), [0, 2, 1, 6, 8, 7, 3, 5, 4]
    )
    fiducials = all_d2_fiducials()
    for first, opposite in zip(fiducials[:4], fiducials[4:]):
        assert abs(np.vdot(first, opposite)) < 1e-12
    assert grey(0).shape == (1, 0)


@pytest.mark.parametrize(
    "bad", [np.zeros((2, 2)), np.eye(2) * 2, np.array([[1, np.nan]]), [1, 2]]
)
def test_reject_incomplete_or_invalid_frames(bad):
    with pytest.raises(ValueError):
        Measurement(bad)


@pytest.mark.parametrize(
    "bad", [[1, 1], [np.nan, 0], np.diag([1.1, -0.1]), [[0.5, 1], [0, 0.5]], np.eye(2)]
)
def test_reject_invalid_states(bad):
    with pytest.raises(ValueError):
        wh_measurement(ket(2)).probabilities(bad)


def test_validation_errors_are_explicit():
    with pytest.raises(ValueError, match="dimension"):
        wh_operators(2.5)
    with pytest.raises(ValueError, match="unit norm"):
        wh_frame([1, 1])
    with pytest.raises(ValueError, match="d²"):
        change_conjugate_convention([0, 1, 2])
    with pytest.raises(ValueError, match="SIC"):
        sic_povm(fiducial=np.array([1, 0]))
    with pytest.raises(ValueError, match="match"):
        sic_povm(3, fiducial=ket(2))
    with pytest.raises(FileNotFoundError):
        load_sic_fiducial(123456)
    with pytest.raises(ValueError, match="at least"):
        embed_state(ket(5), 2)
    with pytest.raises(ValueError, match="unitary"):
        list(embed_gate(np.ones((2, 2)), cirq.LineQubit.range(1)))
    with pytest.raises(ValueError, match="environment"):
        prepare_state(density(2), cirq.LineQubit.range(1))
    with pytest.raises(ValueError, match="disjoint"):
        build_measurement_circuit(
            wh_measurement(ket(2)),
            system_qubits=[cirq.LineQubit(0)],
            ancilla_qubits=[cirq.LineQubit(0)],
        )
    with pytest.raises(ValueError, match="length"):
        ansatz_params_to_angles(2, [0])
    with pytest.raises(ValueError, match="power of two"):
        ket_to_ansatz_angles(ket(3))
    with pytest.raises(ValueError, match="positive"):
        refine_sic_fiducial(ket(2), tolerance=0)


@pytest.mark.parametrize("tolerance", [np.nan, np.inf, -1])
def test_invalid_tolerances_do_not_disable_mathematical_validation(tolerance):
    with pytest.raises(ValueError, match="atol"):
        Measurement(np.eye(2), atol=tolerance)
    with pytest.raises(ValueError, match="atol"):
        Measurement(np.eye(2)).is_sic(atol=tolerance)
    with pytest.raises(ValueError, match="atol"):
        sic_povm(2, atol=tolerance)
    with pytest.raises(ValueError, match="tolerance"):
        refine_sic_fiducial(ket(2), tolerance=tolerance)


def test_parameter_validation_rejects_silent_coercion():
    with pytest.raises(ValueError, match="real"):
        ansatz_params_to_angles(1, [0, 1j, 0])
    with pytest.raises(ValueError, match="real"):
        ansatz_angles_to_params([np.array([1j])], [np.array([0])], 0)
    with pytest.raises(ValueError, match="integer"):
        list(X_d(2, cirq.LineQubit.range(1), k=True))
    with pytest.raises(ValueError, match="integer"):
        grey(True)
    with pytest.raises(ValueError, match="integer"):
        embed_state(ket(2), True)
    with pytest.raises(ValueError, match="integer"):
        refine_sic_fiducial(sic_povm(2).states[:, 0], max_nfev=False)
    with pytest.raises(ValueError, match="integer"):
        minimize_wh_frame_potential(2, maxiter=-1)


@pytest.mark.parametrize(
    "probabilities", [1, [1, 1], [-0.1, 1.1], [np.nan, 0], [1j, 0]]
)
def test_legacy_triple_estimator_validates_probabilities(probabilities):
    with pytest.raises(ValueError, match="probabilities"):
        real_triple_product_from_probabilities(probabilities)
