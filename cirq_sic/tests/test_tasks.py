from cirq_sic import *
import numpy as np

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
    
    for task_type in sk_ground_tasks:
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