import datetime
import sympy as sp

import cirq
import cirq_google as cg
import recirq
import numpy as np

from .utils import *
from .circuits import *

EXPERIMENT_NAME = "sky_ground"
DEFAULT_BASE_DIR = f'data/{EXPERIMENT_NAME}'

def _abbrev_n_shots(n_shots: int) -> str:
    """Shorter n_shots component of a filename"""
    if n_shots % 1000 == 0:
        return f'{n_shots // 1000}k'
    return str(n_shots)

def _abbrev_grid_qubit(qubit: cirq.GridQubit) -> str:
    """Formatted grid_qubit component of a filename"""
    return f'{qubit.row}_{qubit.col}'

@recirq.json_serializable_dataclass(namespace='recirq.sic_on_sic', 
                                    registry=recirq.Registry,
                                    frozen=True)
class SICOnSICTask:
    dataset_id: str
    n_shots: int
    state_qubits: list
    fiducial_qubits: list
    processor_id: str
    run_type: str

    @property
    def fn(self):
        n_shots = _abbrev_n_shots(n_shots=self.n_shots)
        return (f'{self.dataset_id}/'
                f'sic_on_sic_N{n_shots}')
    
    def __str__(self):
        return self.fn

def run_sic_on_sic(task: SICOnSICTask, base_dir=None):
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR

    if recirq.exists(task, base_dir=base_dir):
        print(f"{task} already exists. Skipping.")
        return

    print(f"Starting SIC on SIC...")
    d = 4
    a = [[a1, a2] for a1 in range(d) for a2 in range(d)]
    print(f"Creating circuits...")
    circuits = [cirq.Circuit((d4_sic_fiducial(task.state_qubits),\
                              displace(task.state_qubits, a1, a2),\
                              d4_sic_fiducial(task.fiducial_qubits, conjugate=True),\
                              simple_AK(task.state_qubits, task.fiducial_qubits)))\
                                    for a1, a2 in a]
    
    print(f"Optimizing...")
    device_data = get_device_data(task.processor_id, run_type=task.run_type)
    optimized_circuits = [process_circuit(circ, device_data["connectivity_graph"], device_data["gateset"]) for circ in circuits]
    print(f"Sampling...")
    sampler = device_data["sampler"]
    results = sampler.run_batch(programs=optimized_circuits, repetitions=task.n_shots)
    P = np.array([get_freqs(r[0], n_outcomes=d**2, n_shots=task.n_shots) for r in results])
    print(f"Saving...")
    recirq.save(task=task, data={
        "a": a,\
        "P": P.tolist()
    }, base_dir=base_dir)
    print(f"Done!")