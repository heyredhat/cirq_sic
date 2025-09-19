import cirq
import cirq_google
import qsimcirq
import recirq

from ..circuits import *
from ..utils import *
from .experiment_utils import *

@recirq.json_serializable_dataclass(namespace='recirq.sky_ground', 
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

    @property
    def fn(self):
        """Filename for this task."""
        n_shots = abbrev_n_shots(n_shots=self.n_shots)
        qubits_str = f"Q{"".join([str(q) for q in self.qubits])}"
        return (f'{self.dataset_id}/'f'{self.description}__{self.wh_implementation}__'
                f'{self.processor_id}__{self.run_type}__{n_shots}__{qubits_str}')
    
    def __str__(self):
        return self.fn
    
    def run(self, program, prepare_fiducial, base_dir=None):
        """Create circuits from the provided program, optimize for the device, process samples, and save the data."""
        if recirq.exists(self, base_dir=base_dir):
            print(f"Task already exists. Skipping.")
            return
        print(f"Starting task {self.fn}...")
        print(f"Creating circuits...")
        circuits, context = program.create_circuits(self, prepare_fiducial)
        print(f"Optimizing...")
        device_data = get_device_data(self.processor_id, run_type=self.run_type)
        optimized_circuits = [process_circuit(circ, device_data["connectivity_graph"], device_data["gateset"], self.qubits) for circ in circuits]
        print(f"Sampling...")
        sampler = device_data["sampler"]
        results = sampler.run_batch(programs=optimized_circuits, repetitions=self.n_shots)
        data = {**context, **program.process_results(self, results)}
        if base_dir is not None:
            print(f"Saving...")
            recirq.save(task=self, data=data, base_dir=base_dir)
        print(f"Done!")
        return locals()

class CharacterizeWHReferenceDevice(TaskProgram):
    """Program for characterizing a WH reference device in terms of itself: measuring the WH-POVM on the WH statates."""
    description = "characterize_wh_reference_device"
    @classmethod
    def create_circuits(cls, task, prepare_fiducial):
        d = task.d
        n = int(np.log2(d))
        a = [[a1, a2] for a1 in range(d) for a2 in range(d)]
        if task.wh_implementation == "simple":
            state_qubits = task.qubits[:n]
            fiducial_qubits = task.qubits[n:2*n]
            circuits = [cirq.Circuit((wh_state(state_qubits, prepare_fiducial, a1, a2),\
                                      simple_wh_povm(state_qubits, fiducial_qubits, prepare_fiducial, measure=True)))\
                                        for a1, a2 in a]
        elif task.wh_implementation == "ak":
            ancilla1 = task.qubits[:n]
            ancilla2 = task.qubits[n:2*n]
            state_qubits = task.qubits[2*n:3*n]
            circuits = [cirq.Circuit((wh_state(state_qubits, prepare_fiducial, a1, a2),\
                                      qudit_arthurs_kelly(state_qubits, ancilla1, ancilla2, prepare_fiducial, measure=True)))\
                                             for a1, a2 in a]
        return circuits, {"a": a}

    @classmethod
    def process_results(cls, task, results):
        P = np.array([get_freqs(r[0]) for r in results])
        return {"P": P}

class WHPOVMOnBasisStates(TaskProgram):
    """Program for measuring the WH-POVM on the computational basis states."""
    description = "wh_povm_on_basis_states"
    @classmethod
    def create_circuits(cls, task, prepare_fiducial):
        d = task.d
        n = int(np.log2(d))
        m = np.arange(d)
        if task.wh_implementation == "simple":
            state_qubits = task.qubits[:n]
            fiducial_qubits = task.qubits[n:2*n]
            circuits = [cirq.Circuit((qudit_basis_state(state_qubits, i),\
                                      simple_wh_povm(state_qubits, fiducial_qubits, prepare_fiducial, measure=True)))\
                                        for i in m]
        elif task.wh_implementation == "ak":
            ancilla1 = task.qubits[:n]
            ancilla2 = task.qubits[n:2*n]
            state_qubits = task.qubits[2*n:3*n]
            circuits = [cirq.Circuit((qudit_basis_state(state_qubits, i),\
                                      qudit_arthurs_kelly(state_qubits, ancilla1, ancilla2, prepare_fiducial, measure=True)))\
                                        for i in m]
        return circuits, {"m": m}

    @classmethod
    def process_results(cls, task, results):
        p = np.array([get_freqs(r[0]) for r in results]).T
        return {"p": p}

class BasisMeasurementOnWHStates(TaskProgram):
    description = "basis_measurement_on_wh_states"
    @classmethod
    def create_circuits(cls, task, prepare_fiducial):
        d = task.d
        n = int(np.log2(d))
        a = [[a1, a2] for a1 in range(d) for a2 in range(d)]
        state_qubits = task.qubits[:n]
        circuits = [cirq.Circuit((wh_state(state_qubits, prepare_fiducial, a1, a2),\
                                  cirq.measure(state_qubits, key="result")))\
                                        for a1 in range(d) for a2 in range(d)]
        return circuits, {"a": a}

    @classmethod
    def process_results(cls, task, results):
        C = np.array([get_freqs(r[0]) for r in results]).T
        return {"C": C}
    
class BasisMeasurementOnBasisStates(TaskProgram):
    description = "basis_measurement_on_basis_states"
    @classmethod
    def create_circuits(cls, task, *args):
        d = task.d
        n = int(np.log2(d))
        state_qubits = task.qubits[:n]
        m = np.arange(d)
        circuits = [cirq.Circuit((qudit_basis_state(state_qubits, i),
                                  cirq.measure(state_qubits, key="result"))) for i in m]
        return circuits, {"m": m}

    @classmethod
    def process_results(cls, task, results):
        q = np.array([get_freqs(r[0]) for r in results]).T
        return {"q": q}
    
sky_ground_programs = [CharacterizeWHReferenceDevice, WHPOVMOnBasisStates, BasisMeasurementOnWHStates, BasisMeasurementOnBasisStates]