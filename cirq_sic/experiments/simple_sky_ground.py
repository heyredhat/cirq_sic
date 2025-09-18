import cirq
import cirq_google
import qsimcirq
import recirq

from ..circuits import *
from ..utils import *
from .experiment_utils import *

@recirq.json_serializable_dataclass(namespace='recirq.simple_ak', 
                                    registry=recirq.Registry,
                                    frozen=True)
class SimpleAKTask:
    """Task parameters for simple AK experiment."""
    dataset_id: str
    processor_id: str
    run_type: str
    n_shots: int
    state_qubits: list
    fiducial_qubits: list
    desc: str

    @property
    def fn(self):
        """Filename for this task."""
        n_shots = abbrev_n_shots(n_shots=self.n_shots)
        qubits_str = f"S{"".join([str(q) for q in self.state_qubits])}_F{"".join([str(q) for q in self.fiducial_qubits])}"
        return (f'{self.dataset_id}/'
                f'simple_{self.desc}__'
                f'{self.processor_id}__{self.run_type}__{n_shots}__{qubits_str}')
    
    def __str__(self):
        return self.fn
    
    def run(self, program, base_dir):
        if recirq.exists(self, base_dir=base_dir):
            print(f"Task already exists. Skipping.")
            return
        print(f"Starting simple_{self.desc}...")
        print(f"Creating circuits...")
        circuits, context = program.create_circuits(self)
        print(f"Optimizing...")
        device_data = get_device_data(self.processor_id, run_type=self.run_type)
        optimized_circuits = [process_circuit(circ, device_data["connectivity_graph"], device_data["gateset"]) for circ in circuits]
        print(f"Sampling...")
        sampler = device_data["sampler"]
        results = sampler.run_batch(programs=optimized_circuits, repetitions=self.n_shots)
        print(f"Saving...")
        recirq.save(task=self, data={**context, **program.process_results(self, results)}, base_dir=base_dir)
        print(f"Done!")
        return self

class SimpleSICOnSIC(TaskProgram):
    desc = "sic_on_sic"
    @classmethod
    def create_circuits(cls, task):
        d = 4
        a = [[a1, a2] for a1 in range(d) for a2 in range(d)]
        circuits = [cirq.Circuit((wh_state(task.state_qubits, a1, a2, d4_sic_fiducial),\
                                  d4_sic_fiducial(task.fiducial_qubits, conjugate=True),\
                                  simple_AK(task.state_qubits, task.fiducial_qubits)))\
                                        for a1, a2 in a]
        return circuits, {"a": a}

    @classmethod
    def process_results(cls, task, results):
        P = np.array([get_freqs(r[0]) for r in results])
        return {"P": P}

class SimpleSICOnBasisStates(TaskProgram):
    desc = "sic_on_basis_states"
    @classmethod
    def create_circuits(cls, task):
        d = 4
        m = np.arange(d)
        circuits = [cirq.Circuit((qudit_basis_state(task.state_qubits, i),\
                                  d4_sic_fiducial(task.fiducial_qubits, conjugate=True),\
                                  simple_AK(task.state_qubits, task.fiducial_qubits)))\
                                        for i in m]
        return circuits, {"m": m}

    @classmethod
    def process_results(cls, task, results):
        p = np.array([get_freqs(r[0]) for r in results]).T
        return {"p": p}

class SimpleBasisMeasurementOnSIC(TaskProgram):
    desc = "basis_measurement_on_sic"
    @classmethod
    def create_circuits(cls, task):
        d = 4
        a = [[a1, a2] for a1 in range(d) for a2 in range(d)]
        circuits = [cirq.Circuit((wh_state(task.state_qubits, a1, a2, d4_sic_fiducial),\
                                  cirq.measure(task.state_qubits, key="result")))\
                                        for a1 in range(d) for a2 in range(d)]
        return circuits, {"a": a}

    @classmethod
    def process_results(cls, task, results):
        C = np.array([get_freqs(r[0]) for r in results]).T
        return {"C": C}
    
class SimpleBasisMeasurementOnBasisStates(TaskProgram):
    desc = "basis_measurement_on_basis_states"
    @classmethod
    def create_circuits(cls, task):
        d = 4
        m = np.arange(d)
        circuits = [cirq.Circuit((qudit_basis_state(task.state_qubits, i),
                                  cirq.measure(task.state_qubits, key="result"))) for i in m]
        return circuits, {"m": m}

    @classmethod
    def process_results(cls, task, results):
        q = np.array([get_freqs(r[0]) for r in results]).T
        return {"q": q}
    
simple_sky_ground_programs = [SimpleSICOnSIC, SimpleSICOnBasisStates, SimpleBasisMeasurementOnSIC, SimpleBasisMeasurementOnBasisStates]