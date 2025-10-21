from pathlib import Path

import numpy as np

import cirq
import recirq

from ..wh import *
from ..sics import *
from ..circuits import *
from ..utils import *
from .experiment_utils import *

####################################################################################

def get_wh_qubits(d, wh_implementation):
    n = int(np.log2(d))
    cols = 2 if wh_implementation == "simple" else 3
    return cirq.GridQubit.rect(cols, n, top=4, left=2)

####################################################################################

EXPERIMENT_NAME = "sky_ground"
DEFAULT_BASE_DIR = f"data/{EXPERIMENT_NAME}"

def load_results(task, base_dir=None):
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR
    return recirq.read_json(f"{base_dir}/{task.fn}.json")

def load_circuits(task, base_dir=None, raw=False):
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR
    circuits_directory = Path(f"{base_dir}/{task.fn}").parent / "circuits"
    if raw:
        path = circuits_directory / "raw.json"
        return cirq.read_json(str(path))
    else:
        optimized_path = circuits_directory / f"{task.optimizer}_optimized.json"
        optimized_circuits, mappings = cirq.read_json(str(optimized_path))
        return optimized_circuits, mappings

####################################################################################

def run_sky_ground_task(task, base_dir=None):
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR

    experiment_dir = Path(f"{base_dir}/{task.fn}").parent
    experiment_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(task.fn, experiment_dir / "experiment.log")
    
    try:
        if recirq.exists(task, base_dir=base_dir):
            logger.info(f"Task already exists. Skipping.")
            return

        logger.info(f"Starting task...")    

        logger.info(f"Creating circuits...")
        circuits_directory = experiment_dir / "circuits"
        circuits_directory.mkdir(parents=True, exist_ok=True)
        raw_path = circuits_directory / "raw.json"
        if not raw_path.exists():
            circuits = task.make_circuits()
            cirq.to_json(circuits, str(raw_path))
        else:
            circuits = cirq.read_json(str(raw_path))
        
        logger.info(f"Optimizing circuits...")
        optimized_path = circuits_directory / f"{task.optimizer}_optimized.json"
        if not optimized_path.exists():
            if task.optimizer.startswith("cirq"):
                optimized_circuits, mappings = zip(*[cirq_optimize_circuit(task.qubits, circuit, processor_id=task.processor_id) for circuit in circuits])
            elif task.optimizer.startswith("bqskit"):
                optimization_level = int(task.optimizer[-1])
                machine_model = bqskit_machine_model(task.qubits, processor_id=task.processor_id)
                optimized_circuits, mappings = zip(*[bqskit_optimize_circuit(task.qubits, circuit, machine_model, optimization_level=optimization_level) for circuit in circuits])
            cirq.to_json([optimized_circuits, mappings], str(optimized_path))
        else:
            optimized_circuits, mappings = cirq.read_json(str(optimized_path))
        
        logger.info(f"Sampling...")
        sampler = get_sampler(task.processor_id, run_type=task.run_type)
        results = sampler.run_batch(programs=optimized_circuits, repetitions=task.n_shots)

        logger.info(f"Processing results...")
        data = task.process_results(results, mappings)

        logger.info(f"Saving...")
        recirq.save(task=task, data=data, base_dir=base_dir)
    except Exception as e:
        logger.exception("Help!") 

####################################################################################

@recirq.json_serializable_dataclass(namespace="recirq.sky_ground", 
                                    registry=recirq.Registry,
                                    frozen=True)
class CharacterizeWHReferenceDeviceTask:
    dataset_id: str
    processor_id: str
    run_type: str
    qubits: list
    n_shots: int
    optimizer: str

    d: int
    fiducial: np.array
    fiducial_description: str
    wh_implementation: str

    @property
    def fn(self):
        return (f"{self.dataset_id}/"
                f"d{self.d}/"
                f"{self.__class__.__name__}/"
                f"{self.wh_implementation}/"
                f"{self.fiducial_description}/"
                f"{self.run_type}_n{abbrev_n_shots(self.n_shots)}_{self.processor_id}_q{abbrev_grid_qubits(self.qubits)}")
    
    def make_circuits(self):
        n = int(np.log2(self.d))
        a = [[a1, a2] for a1 in range(self.d) for a2 in range(self.d)]
        prepare_fiducial = ansatz_circuit(self.fiducial)
        if self.wh_implementation == "simple":
            state_qubits = self.qubits[:n]
            fiducial_qubits = self.qubits[n:2*n]
            circuits = [cirq.Circuit((wh_state(state_qubits, prepare_fiducial, a1, a2),\
                                      simple_wh_povm(state_qubits, fiducial_qubits, prepare_fiducial=prepare_fiducial, measure=True)))\
                                        for a1, a2 in a]
        elif self.wh_implementation == "ak":
            ancilla1_qubits = self.qubits[:n]
            ancilla2_qubits = self.qubits[n:2*n]
            state_qubits = self.qubits[2*n:3*n]
            circuits = [cirq.Circuit((wh_state(state_qubits, prepare_fiducial, a1, a2),\
                                      arthurs_kelly(state_qubits, ancilla1_qubits, ancilla2_qubits, prepare_fiducial=prepare_fiducial, measure=True)))\
                                                for a1, a2 in a]
        return circuits
    
    def process_results(self, results, mappings):
        P = np.array([results_to_freqs(r[0], mapping=mappings[i]) for i, r in enumerate(results)])
        if self.wh_implementation == "ak":
            P = change_conjugate_convention(P)
        P = P.T
        return {"P": P}
    
####################################################################################

@recirq.json_serializable_dataclass(namespace="recirq.sky_ground", 
                                    registry=recirq.Registry,
                                    frozen=True)
class WHPOVMOnBasisStatesTask:
    dataset_id: str
    processor_id: str
    run_type: str
    qubits: list
    n_shots: int
    optimizer: str

    d: int
    fiducial: np.array
    fiducial_description: str
    wh_implementation: str

    @property
    def fn(self):
        return (f"{self.dataset_id}/"
                f"d{self.d}/"
                f"{self.__class__.__name__}/"
                f"{self.wh_implementation}/"
                f"{self.fiducial_description}/"
                f"{self.run_type}_n{abbrev_n_shots(self.n_shots)}_{self.processor_id}_q{abbrev_grid_qubits(self.qubits)}")
    
    def make_circuits(self):
        n = int(np.log2(self.d))
        m = np.arange(self.d)
        prepare_fiducial = ansatz_circuit(self.fiducial)
        if self.wh_implementation == "simple":
            state_qubits = self.qubits[:n]
            fiducial_qubits = self.qubits[n:2*n]
            circuits = [cirq.Circuit((qudit_basis_state(state_qubits, i),\
                                      simple_wh_povm(state_qubits, fiducial_qubits, prepare_fiducial=prepare_fiducial, measure=True)))\
                                        for i in m]
        elif self.wh_implementation == "ak":
            ancilla1 = self.qubits[:n]
            ancilla2 = self.qubits[n:2*n]
            state_qubits = self.qubits[2*n:3*n]
            circuits = [cirq.Circuit((qudit_basis_state(state_qubits, i),\
                                      arthurs_kelly(state_qubits, ancilla1, ancilla2, prepare_fiducial=prepare_fiducial, measure=True)))\
                                        for i in m]
        return circuits
    
    def process_results(self, results, mappings):
        p = np.array([results_to_freqs(r[0], mapping=mappings[i]) for i, r in enumerate(results)])
        if self.wh_implementation == "ak":
            p = change_conjugate_convention(p)
        p = p.T
        return {"p": p}
    
####################################################################################

@recirq.json_serializable_dataclass(namespace="recirq.sky_ground", 
                                    registry=recirq.Registry,
                                    frozen=True)
class BasisMeasurementOnWHStatesTask:
    dataset_id: str
    processor_id: str
    run_type: str
    qubits: list
    n_shots: int
    optimizer: str

    d: int
    fiducial: np.array
    fiducial_description: str

    @property
    def fn(self):
        return (f"{self.dataset_id}/"
                f"d{self.d}/"
                f"{self.__class__.__name__}/"
                f"{self.fiducial_description}/"
                f"{self.run_type}_n{abbrev_n_shots(self.n_shots)}_{self.processor_id}_q{abbrev_grid_qubits(self.qubits)}")
    
    def make_circuits(self):
        n = int(np.log2(self.d))
        a = [[a1, a2] for a1 in range(self.d) for a2 in range(self.d)]
        prepare_fiducial = ansatz_circuit(self.fiducial)
        state_qubits = self.qubits[:n]
        circuits = [cirq.Circuit((wh_state(state_qubits, prepare_fiducial, a1, a2),\
                                  cirq.measure(state_qubits, key="result")))\
                                        for a1, a2 in a]
        return circuits
    
    def process_results(self, results, mappings):
        C = np.array([results_to_freqs(r[0], mapping=mappings[i]) for i, r in enumerate(results)])
        C = C.T
        return {"C": C}

####################################################################################

@recirq.json_serializable_dataclass(namespace="recirq.sky_ground", 
                                    registry=recirq.Registry,
                                    frozen=True)
class BasisMeasurementOnBasisStatesTask:
    dataset_id: str
    processor_id: str
    run_type: str
    qubits: list
    n_shots: int
    optimizer: str

    d: int

    @property
    def fn(self):
        return (f"{self.dataset_id}/"
                f"d{self.d}/"
                f"{self.__class__.__name__}/"
                f"{self.run_type}_n{abbrev_n_shots(self.n_shots)}_{self.processor_id}_q{abbrev_grid_qubits(self.qubits)}")
    
    def make_circuits(self):
        n = int(np.log2(self.d))
        m = np.arange(self.d)
        state_qubits = self.qubits[:n]
        circuits = [cirq.Circuit((qudit_basis_state(state_qubits, i),
                                  cirq.measure(state_qubits, key="result"))) for i in m]
        return circuits
    
    def process_results(self, results, mappings):
        q = np.array([results_to_freqs(r[0], mapping=mappings[i]) for i, r in enumerate(results)])
        q = q.T
        return {"q": q}