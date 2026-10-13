"""End-to-end legacy task checks with deterministic local simulation."""

from math import comb

import cirq
import numpy as np
import pytest

from cirq_sic.circuits import get_triple_product_qubits
from cirq_sic.experiments.experiment_utils import get_sampler, shots_to_freqs
from cirq_sic.experiments.sky_ground import (
    BasisMeasurementAfterWHPOVMOnBasisStatesTask,
    BasisMeasurementOnBasisStatesTask,
    BasisMeasurementOnWHStatesTask,
    CharacterizeWHReferenceDeviceTask,
    CharacterizeWHTripleProductsTask,
    WHPOVMOnBasisStatesTask,
    WHPOVMOnStatesTask,
    create_circuits,
    exactify,
    get_wh_qubits,
    load_results,
    load_sky_ground_results,
    optimize_task_circuits,
    run_sky_ground_task,
    sky_ground_tasks,
    task_from_specs,
)
from cirq_sic.experiments.sky_ground_analysis import sky_ground_metrics
from cirq_sic.sics import load_sic_fiducial
from cirq_sic.utils import rand_ket
from cirq_sic.wh import wh_frame, wh_povm


def _specs(d=2, implementation="simple", shots=4000):
    return dict(
        dataset_id="test",
        processor_id="willow_pink",
        run_type="clean",
        qubits=get_wh_qubits(d, implementation),
        n_shots=shots,
        optimizer="none",
        d=d,
        fiducial=load_sic_fiducial(d),
        fiducial_description="numerical_sic",
        wh_implementation=implementation,
    )


@pytest.mark.parametrize("implementation", ["ak", "simple", "msimple"])
def test_sky_ground_battery(tmp_path, implementation):
    """Every task agrees with operator algebra and persists isolated raw data."""
    specs = _specs(implementation=implementation)
    sampler = cirq.Simulator(seed=31, dtype=np.complex128)
    for task_type in sky_ground_tasks:
        if (
            implementation != "ak"
            and task_type is BasisMeasurementAfterWHPOVMOnBasisStatesTask
        ):
            continue
        run_sky_ground_task(
            task_from_specs(task_type, specs), sampler=sampler, base_dir=tmp_path
        )
    tasks, observed = load_sky_ground_results(specs, base_dir=tmp_path)
    effects = wh_povm(specs["fiducial"])
    basis = [np.diag(vector) for vector in np.eye(specs["d"])]
    reference = np.array(
        [[np.trace(a @ b) / np.trace(b) for b in effects] for a in effects]
    ).real
    sky = np.array([[np.trace(a @ b) for b in basis] for a in effects]).real
    conditional = np.array(
        [[np.trace(a @ b) / np.trace(b) for b in effects] for a in basis]
    ).real
    expected = {
        CharacterizeWHReferenceDeviceTask: ("P", reference),
        WHPOVMOnBasisStatesTask: ("r", sky),
        BasisMeasurementOnWHStatesTask: ("C", conditional),
        BasisMeasurementOnBasisStatesTask: ("q", np.eye(specs["d"])),
    }
    if implementation == "ak":
        expected[BasisMeasurementAfterWHPOVMOnBasisStatesTask] = (
            "p",
            conditional @ sky,
        )
    for task_type, (key, prediction) in expected.items():
        assert np.allclose(exactify(tasks[task_type])[key], prediction, atol=1e-10)
        assert np.linalg.norm(observed[key] - prediction) < 0.1
        assert "raw_results" in load_results(tasks[task_type], base_dir=tmp_path)
    assert (
        np.linalg.norm(conditional @ np.linalg.pinv(reference) @ sky - np.eye(2)) < 1e-9
    )


def _state_task(d=4, run_type="clean"):
    specs = _specs(d=d, implementation="msimple", shots=128)
    return WHPOVMOnStatesTask(
        **specs, states=[rand_ket(d)], states_description="random"
    )


def test_msimple_d4_clean_sampler_runs():
    task = _state_task()
    circuit = optimize_task_circuits(task, task.make_circuits())[0]
    sampler = get_sampler(task.processor_id, run_type="clean", circuits=[circuit])
    frequencies = shots_to_freqs(sampler.run(circuit, repetitions=task.n_shots))
    assert np.isclose(frequencies.sum(), 1)
    assert frequencies.shape == (task.d**2,)


def test_msimple_real_sampler_fails_early():
    task = _state_task()
    with pytest.raises(
        ValueError, match="Dynamic circuits with classical feed-forward"
    ):
        get_sampler(task.processor_id, run_type="real", circuits=task.make_circuits())


def test_ak_d4_sky_ground_metrics_runs(tmp_path):
    specs = _specs(d=4, implementation="ak", shots=16)
    sampler = cirq.Simulator(seed=9, dtype=np.complex128)
    for task_type in sky_ground_tasks:
        run_sky_ground_task(
            task_from_specs(task_type, specs), sampler=sampler, base_dir=tmp_path
        )
    metrics = sky_ground_metrics(specs, base_dir=tmp_path)
    assert {
        "P_err",
        "Phi_err",
        "q_err",
        "born_rule_err",
        "p_err",
        "LTP_err",
    } <= metrics.keys()
    assert all(np.isfinite(value) for value in metrics.values())


def _triple_task(d=2, optimizer="none"):
    specs = _specs(d=d)
    specs.pop("wh_implementation")
    specs["qubits"] = get_triple_product_qubits(d)
    specs["optimizer"] = optimizer
    return CharacterizeWHTripleProductsTask(**specs)


def test_wh_triple_products_task(tmp_path):
    task = _triple_task()
    run_sky_ground_task(
        task, sampler=cirq.Simulator(seed=15, dtype=np.complex128), base_dir=tmp_path
    )
    exact = exactify(task)["T"]
    sampled = np.asarray(load_results(task, base_dir=tmp_path)["processed_data"]["T"])
    orbit = wh_frame(task.fiducial).T
    projectors = [np.outer(ket, ket.conj()) for ket in orbit]
    expected = np.array(
        [
            [[np.trace(a @ b @ c).real for c in projectors] for b in projectors]
            for a in projectors
        ]
    )
    assert np.allclose(exact, expected, atol=1e-10)
    assert np.allclose(exact, exact.transpose(1, 0, 2), atol=1e-10)
    assert np.linalg.norm(sampled - expected) < 0.3


def test_triple_products_create_circuits(tmp_path):
    task = _triple_task()
    circuits = create_circuits(task, base_dir=tmp_path)
    assert len(circuits) == comb(task.d**2 + 2, 3)


def test_triple_product_circuit_generic_compilation_preserves_probability():
    task = _triple_task(d=2, optimizer="cirq_generic")
    circuit = task.make_circuits()[3]
    optimized = optimize_task_circuits(task, [circuit])[0]
    from cirq_sic.experiments.experiment_utils import exact_simulation

    assert all(
        len(op.qubits) <= 2
        for op in optimized.all_operations()
        if not cirq.is_measurement(op)
    )
    assert np.allclose(
        exact_simulation(circuit), exact_simulation(optimized), atol=1e-8
    )
