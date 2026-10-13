"""Compatibility, data integrity, and measurement-ordering regressions."""

from dataclasses import replace
import importlib
import json
import subprocess
import sys

import cirq
import numpy as np
import pytest

from cirq_sic.experiments.experiment_utils import (
    collect_circuit_files,
    collect_circuits,
    deep_match,
    exact_simulation,
    get_sampler,
    has_dynamic_circuit_features,
    measurement_key_sort_key,
    push_measurements_to_end,
    result_measurements_matrix,
    results_to_freqs,
    shots_to_freqs,
)
from cirq_sic.experiments.sky_ground import (
    BasisMeasurementOnBasisStatesTask,
    CharacterizeWHReferenceDeviceTask,
    WHPOVMOnBasisStatesTask,
    create_circuits,
    exactify,
    get_wh_qubits,
    load_circuits,
    load_results,
    load_sky_ground_results,
    optimize_task_circuits,
    run_circuits,
    run_sky_ground_task,
    task_from_specs,
)
from cirq_sic.sics import load_sic_fiducial


def _specs():
    return dict(
        dataset_id="roundtrip",
        processor_id="willow_pink",
        run_type="clean",
        qubits=get_wh_qubits(2, "simple"),
        n_shots=64,
        optimizer="none",
        d=2,
        fiducial=load_sic_fiducial(2),
        fiducial_description="sic",
        wh_implementation="simple",
    )


def test_legacy_import_and_clean_sampler_do_not_require_optional_integrations():
    script = """
import importlib.abc
import sys
class BlockExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'recirq', 'cirq_google', 'qsimcirq', 'bqskit', 'jax', 'qiskit', 'cvxpy'}:
            raise ImportError('Optional integration blocked in minimal-install test: ' + fullname)
sys.meta_path.insert(0, BlockExtras())
from cirq_sic.experiments.sky_ground import BasisMeasurementOnBasisStatesTask
from cirq_sic.experiments.sky_ground_analysis import sky_ground_metrics
from cirq_sic.experiments.experiment_utils import get_sampler
get_sampler(run_type='clean')
"""
    subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )


def test_exact_probabilities_follow_logical_keys_not_qubit_sort_order():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.X(q1), cirq.measure(q0, key="a_0"), cirq.measure(q1, key="s_0")
    )
    assert np.array_equal(exact_simulation(circuit), [0, 0, 1, 0])
    result = cirq.Simulator(seed=1).run(circuit, repetitions=5)
    assert np.array_equal(shots_to_freqs(result), [0, 0, 1, 0])
    assert sorted(["a_10", "a_2", "s_0"], key=measurement_key_sort_key) == [
        "s_0",
        "a_2",
        "a_10",
    ]


def test_exact_simulation_applies_invert_mask_once_and_keeps_idle_qubits():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q1, q0, key="result", invert_mask=(True, False))
    )
    assert np.array_equal(exact_simulation(circuit), [0, 0, 1, 0])
    moved = push_measurements_to_end(circuit)
    assert np.array_equal(exact_simulation(moved), [0, 0, 1, 0])
    task = task_from_specs(BasisMeasurementOnBasisStatesTask, _specs())
    generic = optimize_task_circuits(
        replace(task, optimizer="cirq_generic"), [circuit]
    )[0]
    assert np.array_equal(exact_simulation(generic), [0, 0, 1, 0])


def test_exact_simulation_handles_noise_and_requested_marginal_order():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.X(q0),
        cirq.amplitude_damp(0.25)(q0),
        cirq.X(q1),
        cirq.I(q2),
        cirq.measure(q2, q0, key="result"),
    )
    assert np.allclose(exact_simulation(circuit), [0.25, 0.75, 0, 0])


def test_all_result_keys_are_kept_even_when_result_key_exists():
    result = cirq.ResultDict(
        measurements={"result": np.array([[1], [0]]), "a_0": np.array([[0], [1]])}
    )
    assert np.array_equal(result_measurements_matrix(result), [[1, 0], [0, 1]])
    assert np.array_equal(shots_to_freqs(result), [0, 0.5, 0.5, 0])
    assert np.array_equal(results_to_freqs([[result]]), results_to_freqs([result]))
    with pytest.raises(ValueError, match="one parameterization"):
        results_to_freqs([[result, result]])
    empty = cirq.ResultDict(measurements={"m": np.empty((0, 1), dtype=int)})
    with pytest.raises(ValueError, match="zero shots"):
        shots_to_freqs(empty)


def test_dynamic_operations_cannot_be_moved_or_exactly_simulated():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(
        cirq.measure(q, key="first"), cirq.X(q), cirq.measure(q, key="last")
    )
    assert has_dynamic_circuit_features(circuit)
    with pytest.raises(ValueError, match="mid-circuit"):
        push_measurements_to_end(circuit)
    with pytest.raises(ValueError, match="terminal"):
        exact_simulation(circuit)
    with pytest.raises(ValueError, match="run_type"):
        get_sampler(run_type="typo")


def test_task_roundtrip_and_stale_cache_detection(tmp_path):
    task = task_from_specs(CharacterizeWHReferenceDeviceTask, _specs())
    data = run_sky_ground_task(task, sampler=cirq.Simulator(seed=17), base_dir=tmp_path)
    record = load_results(task, base_dir=tmp_path)
    assert isinstance(record["task"], CharacterizeWHReferenceDeviceTask)
    assert np.array_equal(record["task"].fiducial, task.fiducial)
    assert np.array_equal(record["processed_data"]["P"], data["P"])
    assert len(record["raw_results"]) == 4
    assert len(load_circuits(task, base_dir=tmp_path, raw=True)) == 4
    assert np.array_equal(run_sky_ground_task(task, base_dir=tmp_path)["P"], data["P"])
    with pytest.raises(ValueError, match="different task"):
        run_sky_ground_task(replace(task, fiducial=np.array([1, 0])), base_dir=tmp_path)


def test_task_json_retains_historical_namespace(tmp_path):
    task = task_from_specs(BasisMeasurementOnBasisStatesTask, _specs())
    run_sky_ground_task(task, sampler=cirq.Simulator(seed=4), base_dir=tmp_path)
    payload = json.loads((tmp_path / f"{task.fn}.json").read_text())
    assert (
        payload["task"]["cirq_type"]
        == "recirq.sky_ground.BasisMeasurementOnBasisStatesTask"
    )


def test_circuit_preparations_can_be_serialized(tmp_path):
    q = cirq.LineQubit(7)
    task = task_from_specs(WHPOVMOnBasisStatesTask, _specs())
    task = replace(task, fiducial=None, fiducial_circuit=cirq.Circuit(cirq.H(q)))
    run_sky_ground_task(task, sampler=cirq.Simulator(seed=3), base_dir=tmp_path)
    loaded = load_results(task, base_dir=tmp_path)["task"]
    assert isinstance(loaded.fiducial_circuit, cirq.Circuit)
    assert np.allclose(exactify(task)["r"], exactify(loaded)["r"])


def test_legacy_rejects_silent_dimension_truncation_and_malformed_results():
    with pytest.raises(ValueError, match="power-of-two"):
        get_wh_qubits(3, "simple")
    specs = _specs()
    specs["d"] = 3
    with pytest.raises(ValueError, match="power-of-two"):
        task_from_specs(CharacterizeWHReferenceDeviceTask, specs)
    task = task_from_specs(WHPOVMOnBasisStatesTask, _specs())
    with pytest.raises(ValueError, match="shape"):
        task.process_results(probs=np.full((1, 4), 0.25))
    with pytest.raises(ValueError, match="exactly one"):
        task.process_results()


def test_load_does_not_hide_corrupt_records(tmp_path):
    specs = _specs()
    path = tmp_path / f"{CharacterizeWHReferenceDeviceTask.filename(**specs)}.json"
    path.parent.mkdir(parents=True)
    path.write_text("not json")
    with pytest.raises(json.JSONDecodeError):
        load_sky_ground_results(specs, base_dir=tmp_path)


def test_saved_circuit_collection_accepts_path_objects_and_ignores_non_json(tmp_path):
    task = task_from_specs(BasisMeasurementOnBasisStatesTask, _specs())
    expected = create_circuits(task, base_dir=tmp_path)
    circuits_dir = (tmp_path / task.fn).parent / "circuits"
    (circuits_dir / "notes.txt").write_text("not a circuit")
    collected = collect_circuit_files(tmp_path)
    assert collected[circuits_dir] == ["none_optimized.json", "raw.json"]
    circuit_sets = collect_circuits(tmp_path)
    assert list(circuit_sets.values())[0]["raw.json"] == expected
    results = run_circuits(task, base_dir=tmp_path, sampler=cirq.Simulator(seed=2))
    assert np.array_equal(task.process_results(results=results)["q"], np.eye(2))


def test_deep_match_handles_numpy_values_and_cycles():
    record = {"task": {"fiducial": np.array([1, 2]), "d": 2}}
    record["cycle"] = record
    assert deep_match(record, {"d": 2})
    assert deep_match(record, {"task": {"fiducial": [1, 2]}})
    assert not deep_match(record, {"d": 3})
    assert not deep_match(record, {"missing": None})


def test_runner_import_has_no_filesystem_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    importlib.import_module("cirq_sic.experiments.runners.sky_ground_args")
    assert not list(tmp_path.iterdir())


def test_runner_generates_reviewable_argument_file(tmp_path):
    from cirq_sic.experiments.runners.sky_ground_args import write_arguments

    path = write_arguments(tmp_path / "args.txt", "sample", n_shots=100)
    lines = path.read_text().splitlines()
    assert len(lines) == 36
    assert all(
        "-dataset_id sample" in line and "-n_shots 100" in line for line in lines
    )


def test_runner_cli_can_execute_without_hardware_or_plotting(tmp_path):
    from cirq_sic.experiments.runners.sky_ground_run import main

    results = main(
        ["--dataset-id", "cli-test", "--base-dir", str(tmp_path), "--n-shots", "8"]
    )
    assert len(results) == 4
    assert np.array_equal(results[BasisMeasurementOnBasisStatesTask]["q"], np.eye(2))


def test_bqskit_readouts_follow_compiler_final_mapping(monkeypatch):
    """A swapped physical output must not silently relabel logical outcomes."""
    from types import SimpleNamespace
    from cirq_sic.experiments import experiment_utils

    q0, q1 = cirq.GridQubit.rect(2, 1)
    original = cirq.Circuit(
        cirq.X(q0), cirq.measure(q0, q1, key="result", invert_mask=(False, True))
    )
    physical = cirq.Circuit(cirq.X(cirq.NamedQubit("q_1")))
    fake_compiler = SimpleNamespace(
        compile=lambda circuit, **kwargs: (physical, (0, 1), (1, 0))
    )
    fake_conversion = SimpleNamespace(
        cirq_to_bqskit=lambda circuit: circuit, bqskit_to_cirq=lambda circuit: circuit
    )
    monkeypatch.setattr(
        experiment_utils,
        "_optional_import",
        lambda name, extra: fake_conversion if name == "bqskit.ext" else fake_compiler,
    )
    compiled = experiment_utils.bqskit_optimize_circuit(
        [q0, q1], original, machine_model=object()
    )
    assert np.array_equal(exact_simulation(compiled), exact_simulation(original))
    measurement = next(
        op for op in compiled.all_operations() if cirq.is_measurement(op)
    )
    assert measurement.qubits == (q1, q0)


def test_google_routing_preserves_readout_order_and_invert_mask():
    pytest.importorskip("cirq_google")
    from cirq_sic.experiments.experiment_utils import cirq_optimize_circuit

    qubits = cirq.GridQubit.rect(3, 1, top=4, left=2)
    circuit = cirq.Circuit(
        cirq.H(qubits[0]),
        cirq.X(qubits[1]),
        cirq.TOFFOLI(*qubits),
        cirq.measure(qubits[2], key="a_0", invert_mask=(True,)),
        cirq.measure(qubits[0], qubits[1], key="s_0"),
    )
    compiled = cirq_optimize_circuit(qubits, circuit, processor_id="willow_pink")
    assert np.allclose(
        exact_simulation(compiled), exact_simulation(circuit), atol=1e-10
    )
    assert all(
        len(op.qubits) <= 2
        for op in compiled.all_operations()
        if not cirq.is_measurement(op)
    )


def test_single_rectangular_comparison_plot(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from cirq_sic.experiments.sky_ground_analysis import plot_matrix_comparison

    path = plot_matrix_comparison(
        "one matrix", [np.arange(6).reshape(2, 3)], ["example"], tmp_path / "figure"
    )
    assert path.read_bytes().startswith(b"%PDF")


@pytest.mark.parametrize("implementation, expected_count", [("ak", 5), ("simple", 4)])
def test_complete_sky_ground_plotting_battery_closes_figures(
    tmp_path, implementation, expected_count
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cirq_sic.experiments.sky_ground import run_sky_ground_tasks
    from cirq_sic.experiments.sky_ground_analysis import (
        sky_ground_images,
        sky_ground_metrics,
    )

    specs = _specs()
    specs["wh_implementation"] = implementation
    specs["qubits"] = get_wh_qubits(2, implementation)
    run_sky_ground_tasks(
        specs, sampler=cirq.Simulator(seed=3), base_dir=tmp_path / "data"
    )
    previous_figures = plt.get_fignums()
    paths = sky_ground_images(
        specs, img_dir=tmp_path / "figures", base_dir=tmp_path / "data"
    )
    assert len(paths) == expected_count
    assert len(set(paths)) == expected_count
    assert all(path.read_bytes().startswith(b"%PDF") for path in paths)
    assert plt.get_fignums() == previous_figures
    metrics = sky_ground_metrics(specs, base_dir=tmp_path / "data")
    assert all(np.isfinite(value) and value >= 0 for value in metrics.values())
    assert ("LTP_err" in metrics) == (implementation == "ak")


@pytest.mark.parametrize(
    "matrices, labels",
    [
        ([], []),
        ([np.ones((2, 2))], []),
        ([np.array([1, 2])], ["bad shape"]),
        ([np.array([[np.nan]])], ["nonfinite"]),
        ([np.array([[1j]])], ["complex"]),
    ],
)
def test_invalid_comparison_plots_raise_before_opening_figures(
    tmp_path, matrices, labels
):
    import matplotlib.pyplot as plt
    from cirq_sic.experiments.sky_ground_analysis import plot_matrix_comparison

    previous_figures = plt.get_fignums()
    with pytest.raises(ValueError):
        plot_matrix_comparison("invalid", matrices, labels, tmp_path / "invalid")
    assert plt.get_fignums() == previous_figures
    assert not list(tmp_path.iterdir())
