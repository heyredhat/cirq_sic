import cirq
import cirq_google
import qsimcirq
import recirq

import numpy as np

from ..wh import *
from ..sics import *
from ..circuits import *
from ..utils import *
from .experiment_utils import *

####################################################################################

@recirq.json_serializable_dataclass(namespace="recirq.sky_ground", 
                                    registry=recirq.Registry,
                                    frozen=True)
class WHSkyGroundTask:
    """Task parameters for WH sky/ground experiment."""
    dataset_id: str
    processor_id: str
    run_type: str # "clean", "noisy", or "real"
    wh_implementation: str # "simple" or "ak"
    description: str
    n_shots: int
    qubits: list
    d: int
    flag: str

    @property
    def fn(self):
        """Filename for this task."""
        n_shots = f'{self.n_shots // 1000}k' if self.n_shots % 1000 == 0 else str(self.n_shots)
        qubits_str = f"Q{"".join([str(q) for q in self.qubits])}"
        return (f'{self.dataset_id}/'f'{self.description}_{self.flag}__d{self.d}__{self.wh_implementation}__'
                f'{self.processor_id}__{self.run_type}__{n_shots}__{qubits_str}')
    
    def __str__(self):
        return self.fn
    
    def run(self, program, **kwargs):
        base_dir = kwargs["base_dir"]
        """Create circuits from the provided program, optimize for the device, process samples, and save the data."""
        if recirq.exists(self, base_dir=base_dir):
            print(f"Task already exists. Skipping.")
            return
        print(f"Starting task {self.fn}...")
        print(f"Creating circuits...")
        circuits, context = program.create_circuits(self, **kwargs)
        print(f"Optimizing...")
        device_data = get_device_data(self.processor_id, run_type=self.run_type)
        optimized_circuits = [process_circuit(circ, device_data["connectivity_graph"], device_data["gateset"], self.qubits) for circ in circuits]
        print(f"Sampling...")
        sampler = device_data["sampler"]
        results = sampler.run_batch(programs=optimized_circuits, repetitions=self.n_shots)
        data = {**context, "data": results}
        if base_dir is not None:
            print(f"Saving...")
            recirq.save(task=self, data=data, base_dir=base_dir)
        print(f"Done!")
        return {"program": program, **kwargs,\
                "circuits": circuits, "device_data": device_data,\
                "optimized_circuits": optimized_circuits, "results": results, **data}

####################################################################################

class CharacterizeWHReferenceDevice(TaskProgram):
    """Program for characterizing a WH reference device in terms of itself: measuring the WH-POVM on the WH statates."""
    @classmethod
    def create_circuits(cls, task, *args, **kwargs):
        d = task.d
        n = int(np.log2(d))
        a = [[a1, a2] for a1 in range(d) for a2 in range(d)]
        prepare_fiducial = kwargs["prepare_fiducial"]
        if task.wh_implementation == "simple":
            state_qubits = task.qubits[:n]
            fiducial_qubits = task.qubits[n:2*n]
            circuits = [cirq.Circuit((wh_state(state_qubits, prepare_fiducial, a1, a2),\
                                      simple_wh_povm(state_qubits, fiducial_qubits, prepare_fiducial=prepare_fiducial, measure=True)))\
                                        for a1, a2 in a]
        elif task.wh_implementation == "ak":
            ancilla1_qubits = task.qubits[:n]
            ancilla2_qubits = task.qubits[n:2*n]
            state_qubits = task.qubits[2*n:3*n]
            circuits = [cirq.Circuit((wh_state(state_qubits, prepare_fiducial, a1, a2),\
                                      arthurs_kelly(state_qubits, ancilla1_qubits, ancilla2_qubits, prepare_fiducial=prepare_fiducial, measure=True)))\
                                             for a1, a2 in a]
        return circuits, {"a": a}

    @classmethod
    def process_results(cls, task, results, **kwargs):
        P = np.array([get_freqs(r[0], **kwargs) for r in results])
        if task.wh_implementation == "ak":
            P = change_conjugate_convention(P)
        return {"P": P}

class WHPOVMOnBasisStates(TaskProgram):
    """Program for measuring the WH-POVM on the computational basis states."""
    @classmethod
    def create_circuits(cls, task, *args, **kwargs):
        d = task.d
        n = int(np.log2(d))
        m = np.arange(d)
        prepare_fiducial = kwargs["prepare_fiducial"]
        if task.wh_implementation == "simple":
            state_qubits = task.qubits[:n]
            fiducial_qubits = task.qubits[n:2*n]
            circuits = [cirq.Circuit((qudit_basis_state(state_qubits, i),\
                                      simple_wh_povm(state_qubits, fiducial_qubits, prepare_fiducial=prepare_fiducial, measure=True)))\
                                        for i in m]
        elif task.wh_implementation == "ak":
            ancilla1 = task.qubits[:n]
            ancilla2 = task.qubits[n:2*n]
            state_qubits = task.qubits[2*n:3*n]
            circuits = [cirq.Circuit((qudit_basis_state(state_qubits, i),\
                                      arthurs_kelly(state_qubits, ancilla1, ancilla2, prepare_fiducial=prepare_fiducial, measure=True)))\
                                        for i in m]
        return circuits, {"m": m}

    @classmethod
    def process_results(cls, task, results, **kwargs):
        p = np.array([get_freqs(r[0], **kwargs) for r in results])
        if task.wh_implementation == "ak":
            p = change_conjugate_convention(p)
        return {"p": p}

class BasisMeasurementOnWHStates(TaskProgram):
    @classmethod
    def create_circuits(cls, task, *args, **kwargs):
        d = task.d
        n = int(np.log2(d))
        a = [[a1, a2] for a1 in range(d) for a2 in range(d)]
        prepare_fiducial = kwargs["prepare_fiducial"]
        state_qubits = task.qubits[:n]
        circuits = [cirq.Circuit((wh_state(state_qubits, prepare_fiducial, a1, a2),\
                                  cirq.measure(state_qubits, key="result")))\
                                        for a1 in range(d) for a2 in range(d)]
        return circuits, {"a": a}

    @classmethod
    def process_results(cls, task, results, **kwargs):
        C = np.array([get_freqs(r[0], **kwargs) for r in results]).T
        if task.wh_implementation == "ak":
            C = change_conjugate_convention(C)
        return {"C": C}
    
class BasisMeasurementOnBasisStates(TaskProgram):
    @classmethod
    def create_circuits(cls, task, *args, **kwargs):
        d = task.d
        n = int(np.log2(d))
        m = np.arange(d)
        state_qubits = task.qubits[:n]
        circuits = [cirq.Circuit((qudit_basis_state(state_qubits, i),
                                  cirq.measure(state_qubits, key="result"))) for i in m]
        return circuits, {"m": m}

    @classmethod
    def process_results(cls, task, results, **kwargs):
        q = np.array([get_freqs(r[0], **kwargs) for r in results]).T
        return {"q": q}
    
sky_ground_programs = [(CharacterizeWHReferenceDevice, "P"),\
                       (WHPOVMOnBasisStates, "p"),\
                       (BasisMeasurementOnWHStates, "C"),\
                       (BasisMeasurementOnBasisStates, "q")]

####################################################################################

def calculate_sky_ground_metrics(P, p, C, q, verbose=False):
    d = int(np.sqrt(P.shape[0]))
    Phi = np.linalg.inv(P)
    q_ = C @ Phi @ p
    P_sic = SIC_P(d)
    Phi_sic = np.linalg.inv(P_sic)

    P_err = np.linalg.norm(P - P_sic)
    Phi_err = np.linalg.norm(Phi - Phi_sic)
    quantumness = np.linalg.norm(np.eye(d**2) - Phi)
    quantumness_sic = np.linalg.norm(np.eye(d**2) - Phi_sic)
    q_err = np.linalg.norm(np.eye(d) - q)
    sg_q_err = np.linalg.norm(q - q_)
    
    if verbose:
        print("Sky/Ground Metrics:")
        print(f"|P - P_SIC| = {P_err}")
        print(f"|Phi - Phi_SIC| = {Phi_err}")
        print(f"|I - Phi| = {quantumness}")
        print(f"|I - Phi_SIC| = {quantumness_sic}")
        print(f"|I - q| = {q_err}")
        print(f"|q - C Phi p| = {sg_q_err}")
    return locals()

