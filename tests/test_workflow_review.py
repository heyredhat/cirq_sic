"""Independent regressions for experiment identity, readout, and transforms."""
import json

import cirq
import numpy as np
import pytest

from cirq_sic.analysis import born_rule_prediction
from cirq_sic.measurements import Measurement, sic_povm
from cirq_sic.workflow import (
    CircuitSpec,
    Experiment,
    ExperimentResult,
    OutcomeData,
    compile_circuit,
    sky_ground_experiment,
    tomography_experiment,
    wh_entropy_experiment,
)


def test_result_content_identity_rejects_other_state_and_changed_analysis_inputs():
    first = tomography_experiment([1, 0])
    other = tomography_experiment([0, 1])
    result = first.simulate()
    assert set(result.outcomes) == {spec.name for spec in other.circuits}
    with pytest.raises(ValueError, match="fingerprint"):
        other.analyze(result)
    first.arrays["effects"][0] *= 0.9
    with pytest.raises(ValueError, match="fingerprint"):
        first.analyze(result)


def test_fingerprint_survives_round_trip_and_detects_saved_content_edits(tmp_path):
    experiment = tomography_experiment([1, 0])
    experiment.arrays["integer_input"] = np.array([1, 2], dtype=complex)
    result = experiment.run(seed=np.int64(119), repetitions=100)
    experiment.save(tmp_path / "experiment.json")
    result.save(tmp_path / "result.json")
    restored = Experiment.load(tmp_path / "experiment.json")
    loaded_result = ExperimentResult.load(tmp_path / "result.json")
    assert restored.fingerprint == experiment.fingerprint
    assert np.array_equal(
        restored.analyze(loaded_result)["probabilities"], result["measurement"].valid
    )
    document = json.loads((tmp_path / "experiment.json").read_text())
    document["arrays"]["integer_input"]["real"][0] = 99
    (tmp_path / "experiment.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="fingerprint"):
        Experiment.load(tmp_path / "experiment.json")


def test_original_integer_arrays_have_canonical_fingerprints(tmp_path):
    q = cirq.LineQubit(0)
    spec = CircuitSpec("m", cirq.Circuit(cirq.measure(q, key="m")), (q,), (q,), "m", 2)
    experiment = Experiment([spec], arrays={"integers": np.array([1, 2])})
    experiment.save(tmp_path / "integer.json")
    assert (
        Experiment.load(tmp_path / "integer.json").fingerprint == experiment.fingerprint
    )


def test_imported_data_requires_explicit_unbound_opt_in():
    experiment = tomography_experiment([1, 0])
    generated = experiment.simulate()
    imported = ExperimentResult(generated.outcomes, "external")
    with pytest.raises(ValueError, match="allow_unbound=True"):
        experiment.analyze(imported)
    np.testing.assert_allclose(
        experiment.analyze(imported, allow_unbound=True)["density_matrix"],
        np.diag([1, 0]),
        atol=1e-10,
    )
    imported.metadata["experiment_fingerprint"] = "wrong"
    with pytest.raises(ValueError, match="different experiment"):
        experiment.analyze(imported, allow_unbound=True)


def test_transform_updates_identity_preserves_layout_and_does_not_mutate_source():
    original = tomography_experiment([1, 0, 0])
    original_circuit = cirq.Circuit(original.circuits[0].circuit)
    compiled = original.transform(compile_circuit)
    assert original.circuits[0].circuit == original_circuit
    assert compiled.fingerprint != original.fingerprint
    compiled_result = compiled.simulate()
    np.testing.assert_allclose(
        compiled.analyze(compiled_result)["density_matrix"],
        np.diag([1, 0, 0]),
        atol=1e-7,
    )
    with pytest.raises(ValueError, match="fingerprint"):
        original.analyze(compiled_result)
    with pytest.raises(ValueError, match="qubit_order"):
        original.transform(
            lambda circuit: circuit.transform_qubits(
                lambda q: cirq.LineQubit(q.x + 100)
            )
        )


def test_measurement_confusion_and_classical_feedforward_are_explicitly_rejected():
    a, b = cirq.LineQubit.range(2)
    readout = cirq.MeasurementGate(
        1, key="m", confusion_map={(0,): np.array([[0.0, 1.0], [1.0, 0.0]])}
    )
    with pytest.raises(ValueError, match="confusion"):
        CircuitSpec("m", cirq.Circuit(readout(a)), (a,), (a,), "m", 2)
    conditional = cirq.Circuit(
        cirq.H(a), cirq.measure(a, key="m"), cirq.X(b).with_classical_controls("m")
    )
    # Cirq's terminal check alone does not detect this classical dependency.
    assert conditional.are_all_measurements_terminal()
    with pytest.raises(ValueError, match="Classical controls"):
        CircuitSpec("m", conditional, (a, b), (a,), "m", 2)


def test_noise_reaches_recorded_idle_qubits_and_matches_analytic_distribution():
    a, b = cirq.LineQubit.range(2)
    spec = CircuitSpec(
        "m",
        cirq.Circuit(cirq.X(a), cirq.measure(a, b, key="m")),
        (a, b),
        (a, b),
        "m",
        4,
    )
    experiment = Experiment([spec])
    # After one active moment, depolarizing noise flips each classical bit
    # with probability 2*p/3, including b which appears only in measurement.
    p = 0.3
    flip = 2 * p / 3
    expected = np.array(
        [flip * (1 - flip), flip**2, (1 - flip) ** 2, flip * (1 - flip)]
    )
    np.testing.assert_allclose(
        experiment.simulate(noise=cirq.depolarize(p))["m"].probabilities,
        expected,
        atol=1e-12,
    )
    # Insert noise explicitly: Cirq's implicit sampler noise can split a
    # unitary prefix and delay terminal sampling past post-readout noise.
    noisy = experiment.with_noise(cirq.depolarize(p))
    assert noisy.fingerprint != experiment.fingerprint
    np.testing.assert_allclose(
        noisy.simulate()["m"].probabilities, expected, atol=1e-12
    )
    sample = noisy.run(cirq.DensityMatrixSimulator(seed=23), repetitions=12000)
    np.testing.assert_allclose(sample["m"].probabilities, expected, atol=0.015)


@pytest.mark.parametrize("value", [True, 1.5, np.float64(2)])
def test_outcome_layout_requires_integer_count(value):
    with pytest.raises(ValueError, match="physical outcome"):
        OutcomeData([0.5, 0.5], value)


@pytest.mark.parametrize("value", [True, 2.0, np.float64(2)])
def test_repetitions_require_integer_count(value):
    with pytest.raises(ValueError, match="positive repetitions"):
        OutcomeData([0.5, 0.5], 2, [1, 1], value)


@pytest.mark.parametrize("counts", [[np.inf, 0], [np.nan, 0], [2**63, 0]])
def test_counts_reject_nonfinite_and_int64_overflow(counts):
    with pytest.raises(ValueError, match="Counts"):
        OutcomeData([1, 0], 2, counts, 2**63)


def test_complex_probabilities_and_nonportable_seeds_fail_before_execution():
    with pytest.raises(ValueError, match="real"):
        OutcomeData([0.5 + 0.1j, 0.5 - 0.1j], 2)
    experiment = tomography_experiment([1, 0])
    with pytest.raises(ValueError, match="seed"):
        experiment.run(seed=np.random.RandomState(5))
    for base in (0.5, 1, 0, np.inf):
        with pytest.raises(ValueError, match="greater than one"):
            wh_entropy_experiment([1, 0], base=base)


def test_rectangular_ground_and_state_batch_orientations():
    # Three input states and six ground outputs make silent transposes fail.
    d = 2
    states = [np.array([1, 0]), np.array([0, 1]), np.array([1, 1j]) / np.sqrt(2)]
    bases = [
        np.eye(2),
        np.array([[1, 1], [1, -1]]) / np.sqrt(2),
        np.array([[1, 1], [1j, -1j]]) / np.sqrt(2),
    ]
    ground = Measurement(np.column_stack(bases) / np.sqrt(3))
    experiment = sky_ground_experiment(states, sic_povm(d), ground)
    report = experiment.analyze(experiment.simulate())
    assert report["sky"].shape == (3, 4)
    assert report["ground"].shape == (3, 6)
    assert report["conditional"].shape == (6, 4)
    assert report["reference"].shape == (4, 4)
    np.testing.assert_allclose(
        report["ground"], [ground.probabilities(state) for state in states], atol=1e-12
    )
    np.testing.assert_allclose(report["residual"], 0, atol=1e-9)
    np.testing.assert_allclose(
        born_rule_prediction(
            report["sky"], report["conditional"], reference=report["reference"]
        ),
        report["ground"],
        atol=1e-9,
    )


def test_noise_timeline_preserves_readout_before_noise_and_real_idle_moments():
    class ReadoutNoise(cirq.NoiseModel):
        def noisy_moment(self, moment, system_qubits):
            if any(cirq.is_measurement(op) for op in moment):
                flips = cirq.Moment(cirq.X(q) for q in system_qubits)
                return [flips, moment, flips]
            return moment

    q = cirq.LineQubit(0)
    spec = CircuitSpec("m", cirq.Circuit(cirq.measure(q, key="m")), (q,), (q,), "m", 2)
    experiment = Experiment([spec])
    np.testing.assert_allclose(
        experiment.simulate(noise=ReadoutNoise())["m"].probabilities, [0, 1]
    )
    sample = experiment.with_noise(ReadoutNoise()).run(repetitions=10, seed=1)
    np.testing.assert_array_equal(sample["m"].counts, [0, 10])

    circuit = cirq.Circuit(
        [cirq.Moment(cirq.X(q)), cirq.Moment(), cirq.Moment(cirq.measure(q, key="m"))]
    )
    idle = Experiment([CircuitSpec("m", circuit, (q,), (q,), "m", 2)])
    # Preparation X, one noise X, idle-moment noise X: the measured bit is 1.
    # A fictitious noise X after dropping the readout would turn it back to 0.
    np.testing.assert_allclose(
        idle.simulate(noise=cirq.bit_flip(1))["m"].probabilities, [0, 1]
    )
