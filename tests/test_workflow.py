"""Integration tests of the common circuit/data lifecycle."""
import json

import cirq
import numpy as np
import pytest

from cirq_sic.measurements import Measurement, sic_povm
from cirq_sic.workflow import (
    CircuitSpec,
    Experiment,
    ExperimentResult,
    OutcomeData,
    tomography_experiment,
    entropy_experiment,
    sky_ground_experiment,
    triple_product_experiment,
    compile_circuit,
    wh_entropy_experiment,
)


def random_density(d, seed=19):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    rho = a @ a.conj().T
    return rho / np.trace(rho)


@pytest.mark.parametrize("d", [1, 2, 3, 4, 5])
def test_tomography_mixed_roundtrip(d, tmp_path):
    rho = random_density(d)
    measurement = sic_povm(d)
    experiment = tomography_experiment(rho, measurement)
    result = experiment.simulate()
    assert np.allclose(
        result["measurement"].valid, measurement.probabilities(rho), atol=1e-10
    )
    assert result["measurement"].leakage < 1e-12
    assert np.allclose(experiment.analyze(result)["density_matrix"], rho, atol=1e-9)
    path = tmp_path / "experiment.json"
    experiment.save(path)
    restored = Experiment.load(path)
    assert np.allclose(
        restored.simulate()["measurement"].probabilities,
        result["measurement"].probabilities,
        atol=1e-10,
    )
    result.save(tmp_path / "result.json")
    loaded = ExperimentResult.load(tmp_path / "result.json")
    assert np.array_equal(
        loaded["measurement"].probabilities, result["measurement"].probabilities
    )


def test_shots_seed_and_persistence(tmp_path):
    experiment = tomography_experiment(np.array([1, 1j]) / np.sqrt(2))
    first = experiment.run(repetitions=6000, seed=713)
    second = experiment.run(repetitions=6000, seed=713)
    assert np.array_equal(first["measurement"].counts, second["measurement"].counts)
    assert first["measurement"].counts.sum() == 6000
    assert first["measurement"].standard_errors.shape == (4,)
    assert (
        np.linalg.norm(
            experiment.analyze(first, physical=True)["density_matrix"]
            - np.array([[1, -1j], [1j, 1]]) / 2
        )
        < 0.06
    )
    first.save(tmp_path / "sample.json")
    loaded = ExperimentResult.load(tmp_path / "sample.json")
    assert np.array_equal(first["measurement"].counts, loaded["measurement"].counts)


@pytest.mark.parametrize("d", [2, 3])
def test_sky_ground_exact_and_sampled(d):
    states = [random_density(d), np.eye(d)[-1]]
    experiment = sky_ground_experiment(states, sic_povm(d))
    exact = experiment.analyze(experiment.simulate())
    assert np.max(abs(exact["residual"])) < 1e-9
    sampled = experiment.analyze(experiment.run(repetitions=4000, seed=42))
    assert np.max(abs(sampled["residual"])) < 0.1


def test_entropy_mixed_exact():
    experiment = entropy_experiment(np.eye(2) / 2, mixed=True)
    report = experiment.analyze(experiment.simulate())
    assert abs(report["entropy"]) < 1e-9
    assert np.isclose(report["purity"], 0.5)


def test_triples_complete_and_subset():
    states = sic_povm(2).states.T
    experiment = triple_product_experiment(states)
    report = experiment.analyze(experiment.simulate())
    projectors = np.array([np.outer(v, v.conj()) for v in states])
    expected = np.einsum("iab,jbc,kca->ijk", projectors, projectors, projectors)
    assert np.allclose(report["tensor"], expected, atol=1e-9)
    subset = triple_product_experiment(states, [(0, 1, 2)])
    report = subset.analyze(subset.simulate())
    assert "tensor" not in report
    assert np.allclose(report["estimates"], [expected[0, 1, 2]])


def test_leakage_retained_and_rejected():
    experiment = tomography_experiment(np.eye(3)[0])
    spec = experiment.circuits[0]
    p = np.ones(spec.output_size) / spec.output_size
    data = OutcomeData(p, 9)
    assert np.isclose(data.leakage, 7 / 16)
    assert data.valid.sum() < 1
    with pytest.raises(ValueError, match="leakage"):
        experiment.analyze(ExperimentResult({"measurement": data}, "sampled"))


def test_exact_noise_and_compilation():
    experiment = tomography_experiment(np.array([1.0, 0.0]))
    transformed = experiment.transform(
        lambda c: cirq.optimize_for_target_gateset(c, gateset=cirq.CZTargetGateset())
    )
    assert np.allclose(
        experiment.simulate()["measurement"].probabilities,
        transformed.simulate()["measurement"].probabilities,
    )
    noisy = experiment.simulate(noise=cirq.depolarize(0.05))[
        "measurement"
    ].probabilities
    assert np.isclose(noisy.sum(), 1)
    assert not np.allclose(noisy, experiment.simulate()["measurement"].probabilities)


def test_layout_validation_and_idle_qubit_marginal():
    q = cirq.LineQubit.range(3)
    spec = CircuitSpec(
        "test",
        cirq.Circuit(cirq.X(q[2]), cirq.measure(q[2], q[0], key="m")),
        tuple(q),
        (q[2], q[0]),
        "m",
        4,
    )
    result = Experiment([spec]).simulate()["test"]
    assert np.allclose(result.probabilities, [0, 0, 1, 0])
    with pytest.raises(ValueError, match="terminal measurement"):
        CircuitSpec("bad", spec.circuit, tuple(q), (q[0], q[2]), "m", 4)
    with pytest.raises(ValueError, match="positive integer"):
        Experiment([spec]).run(repetitions=0)
    with pytest.raises(ValueError, match="seed"):
        Experiment([spec]).run(cirq.Simulator(), seed=1)


def test_artifact_version_validation(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": 99, "kind": "cirq_sic.experiment"}))
    with pytest.raises(ValueError, match="schema version"):
        Experiment.load(path)


@pytest.mark.parametrize(
    "probabilities,counts,repetitions",
    [
        ([0.5, 0.5], [1, 2], 3),
        ([0.5, 0.5], [-1, 3], 2),
        ([1, 1], None, None),
        ([np.nan, 0], None, None),
    ],
)
def test_invalid_outcomes(probabilities, counts, repetitions):
    with pytest.raises(ValueError):
        OutcomeData(probabilities, 2, counts, repetitions)


@pytest.mark.parametrize("d", [2, 3])
def test_arbitrary_dimension_compilation(d):
    experiment = tomography_experiment(np.eye(d)[0])
    compiled = experiment.transform(compile_circuit)
    assert all(
        len(op.qubits) <= 2 or cirq.is_measurement(op)
        for op in compiled.circuits[0].circuit.all_operations()
    )
    assert np.allclose(
        compiled.simulate()["measurement"].probabilities,
        experiment.simulate()["measurement"].probabilities,
        atol=1e-7,
    )


@pytest.mark.parametrize("d", [1, 2, 3, 5])
def test_direct_mixed_wh_entropy(d):
    from cirq_sic.analysis import stabilizer_renyi_entropy

    rho = random_density(d)
    experiment = wh_entropy_experiment(rho)
    report = experiment.analyze(experiment.simulate())
    assert np.isclose(report["purity"], np.trace(rho @ rho).real, atol=1e-9)
    assert np.isclose(
        report["entropy"],
        stabilizer_renyi_entropy(rho, basis="wh", mixed=True),
        atol=1e-9,
    )
