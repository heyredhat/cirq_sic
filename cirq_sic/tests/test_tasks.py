from cirq_sic import *
import numpy as np
import pytest

def test_sg():
    base_dir = "cirq_sic/tests/data/"
    specs = {"dataset_id": "test",
         "processor_id": "willow_pink",
         "run_type": "clean",
         "qubits": get_wh_qubits(2, "ak"),
         "n_shots": 50000,
         "optimizer": "cirq",
         "d": 2,
         "fiducial": rand_ket(2),
         "fiducial_description": "rand_ket",
         "wh_implementation": "ak"}
    
    for task_type in sky_ground_tasks:
          task = task_from_specs(task_type, specs)
          run_sky_ground_task(task, base_dir=base_dir)

    tasks, sg_results = load_sky_ground_results(specs, separate=True, base_dir=base_dir)
    tol = 1e-1

    task = tasks[CharacterizeWHReferenceDeviceTask]
    E = wh_povm(np.array(task.fiducial))
    P = np.array([[(a@b).trace()/b.trace() for b in E] for a in E]).real
    assert np.allclose(P, exactify(task, base_dir=base_dir)["P"])
    assert np.linalg.norm(sg_results["P"] - P) < 1e-1

    task = tasks[WHPOVMOnBasisStatesTask]
    E = wh_povm(np.array(task.fiducial))
    Pi = [np.diag(np.eye(task.d)[i]) for i in range(task.d)]
    r = np.array([[(a@b).trace() for b in Pi] for a in E]).real
    assert np.allclose(r, exactify(task, base_dir=base_dir)["r"])
    assert np.linalg.norm(sg_results["r"] - r) < 1e-1

    task = tasks[BasisMeasurementOnWHStatesTask]
    E = wh_povm(np.array(task.fiducial))
    Pi = [np.diag(np.eye(task.d)[i]) for i in range(task.d)]
    C = np.array([[(a@b).trace()/b.trace() for b in E] for a in Pi]).real
    assert np.allclose(C, exactify(task, base_dir=base_dir)["C"])
    assert np.linalg.norm(sg_results["C"] - C) < 1e-1

    task = tasks[BasisMeasurementOnBasisStatesTask]
    q = np.eye(task.d)
    assert np.allclose(q, exactify(task, base_dir=base_dir)["q"])
    assert np.linalg.norm(sg_results["q"] - q) < 1e-1
    
    task = tasks[BasisMeasurementAfterWHPOVMOnBasisStatesTask]
    E = wh_povm(np.array(task.fiducial))
    Pi = [np.diag(np.eye(task.d)[i]) for i in range(task.d)]
    post_measurement = np.array([sum([(a@b).trace()*b/b.trace() for b in E]) for a in Pi]).real
    p = np.array([[(pi @ rho).trace() for pi in Pi] for rho in post_measurement])
    assert np.allclose(p, exactify(task, base_dir=base_dir)["p"])
    assert np.linalg.norm(sg_results["p"] - p) < tol

def test_sg_msimple():
    base_dir = "cirq_sic/tests/data/"
    specs = {"dataset_id": "test_msimple_fixed",
         "processor_id": "willow_pink",
         "run_type": "clean",
         "qubits": get_wh_qubits(2, "msimple"),
         "n_shots": 20000,
         "optimizer": "cirq",
         "d": 2,
         "fiducial": load_sic_fiducial(2),
         "fiducial_description": "numerical_sic",
         "wh_implementation": "msimple"}

    for task_type in sky_ground_tasks:
          if task_type == BasisMeasurementAfterWHPOVMOnBasisStatesTask:
              continue
          task = task_from_specs(task_type, specs)
          run_sky_ground_task(task, base_dir=base_dir)

    tasks, sg_results = load_sky_ground_results(specs, separate=True, base_dir=base_dir)
    tol = 1e-1

    task = tasks[CharacterizeWHReferenceDeviceTask]
    E = wh_povm(np.array(task.fiducial))
    P = np.array([[(a@b).trace()/b.trace() for b in E] for a in E]).real
    assert np.allclose(P, exactify(task, base_dir=base_dir)["P"])
    assert np.linalg.norm(sg_results["P"] - P) < tol

    task = tasks[WHPOVMOnBasisStatesTask]
    E = wh_povm(np.array(task.fiducial))
    Pi = [np.diag(np.eye(task.d)[i]) for i in range(task.d)]
    r = np.array([[(a@b).trace() for b in Pi] for a in E]).real
    assert np.allclose(r, exactify(task, base_dir=base_dir)["r"])
    assert np.linalg.norm(sg_results["r"] - r) < tol

    task = tasks[BasisMeasurementOnWHStatesTask]
    E = wh_povm(np.array(task.fiducial))
    Pi = [np.diag(np.eye(task.d)[i]) for i in range(task.d)]
    C = np.array([[(a@b).trace()/b.trace() for b in E] for a in Pi]).real
    assert np.allclose(C, exactify(task, base_dir=base_dir)["C"])
    assert np.linalg.norm(sg_results["C"] - C) < tol

    task = tasks[BasisMeasurementOnBasisStatesTask]
    q = np.eye(task.d)
    assert np.allclose(q, exactify(task, base_dir=base_dir)["q"])
    assert np.linalg.norm(sg_results["q"] - q) < tol

def test_msimple_d4_clean_sampler_runs():
    d = 4
    task = WHPOVMOnStatesTask(
        dataset_id="tmp",
        processor_id="willow_pink",
        run_type="clean",
        qubits=get_wh_qubits(d, "msimple"),
        n_shots=128,
        optimizer="cirq",
        d=d,
        fiducial=load_sic_fiducial(d),
        fiducial_description="numerical_sic",
        states=[rand_ket(d)],
        states_description="rand_ket",
        wh_implementation="msimple",
    )
    circuit = optimize_task_circuits(task, task.make_circuits())[0]
    sampler = get_sampler(task.processor_id, run_type=task.run_type, circuits=[circuit])
    shots = sampler.run(circuit, repetitions=task.n_shots)
    freqs = shots_to_freqs(shots)

    assert np.isclose(freqs.sum(), 1.0)
    assert freqs.shape == (d**2,)

def test_msimple_real_sampler_fails_early():
    d = 4
    task = WHPOVMOnStatesTask(
        dataset_id="tmp",
        processor_id="willow_pink",
        run_type="real",
        qubits=get_wh_qubits(d, "msimple"),
        n_shots=16,
        optimizer="cirq",
        d=d,
        fiducial=load_sic_fiducial(d),
        fiducial_description="numerical_sic",
        states=[rand_ket(d)],
        states_description="rand_ket",
        wh_implementation="msimple",
    )
    circuit = optimize_task_circuits(task, task.make_circuits())[0]

    with pytest.raises(ValueError, match="Dynamic circuits with classical feed-forward"):
        get_sampler(task.processor_id, run_type=task.run_type, circuits=[circuit])

def test_ak_d4_sky_ground_metrics_runs(tmp_path):
    d = 4
    specs = {
        "dataset_id": "tmp_ak_d4",
        "processor_id": "willow_pink",
        "run_type": "clean",
        "qubits": get_wh_qubits(d, "ak"),
        "n_shots": 16,
        "optimizer": "cirq",
        "d": d,
        "fiducial": load_sic_fiducial(d),
        "fiducial_description": "numerical_sic",
        "states": [rand_ket(d)],
        "states_description": "rand_ket",
        "wh_implementation": "ak",
    }
    base_dir = str(tmp_path)

    for task_type in sky_ground_tasks:
        task = task_from_specs(task_type, specs)
        run_sky_ground_task(task, base_dir=base_dir)

    metrics = sky_ground_metrics(specs, base_dir=base_dir)
    assert {"P_err", "Phi_err", "q_err", "born_rule_err", "p_err", "LTP_err"} <= set(metrics)
