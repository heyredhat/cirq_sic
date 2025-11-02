import inspect
from pathlib import Path
from typing import Optional

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
    n = int(np.ceil(np.log2(d)))
    if 2**n == d:
        cols = 2 if wh_implementation == "simple" else 3
        return cirq.GridQubit.rect(cols, n, top=4, left=2)
    else:
        n = int(np.ceil(np.log2(d)) + 1)
        cols = 3 if wh_implementation == "simple" else 4
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
        optimized_circuits = cirq.read_json(str(optimized_path))
        return optimized_circuits

def load_sky_ground_results(specs, base_dir=None, separate=True):
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR
    results = {}
    for task_type in sky_ground_tasks:
        if specs["wh_implementation"] == "simple" and task_type == BasisMeasurementAfterWHPOVMOnBasisStatesTask:
            continue
        try:
            results[task_type] = recirq.read_json(f"{base_dir}/{task_type.filename(**specs)}.json")
        except:
            continue
    if separate:
        tasks = {task_type: stuff["task"] for task_type, stuff in results.items()}
        data = [stuff["processed_data"] for task_type, stuff in results.items()]
        return tasks, {k: np.array(v) for d in data for k, v in d.items()}
    else:
        return results

####################################################################################

def exactify(task, base_dir=None):
    circuits = load_circuits(task, base_dir=base_dir)
    return task.process_results(probs=np.array([exact_simulation(circuit) for i, circuit in enumerate(circuits)]))

####################################################################################

def task_from_specs(task_type, specs):
    sig = inspect.signature(task_type).parameters
    task = task_type(**{k: specs[k] for k in sig.keys() if k in specs})
    return task

def run_sky_ground_task_from_specs(task_type, specs, base_dir=None):
    run_sky_ground_task(task_from_specs(task_type, specs), base_dir=base_dir) 

def run_sky_ground_task(task, base_dir=None):
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR

    experiment_dir = Path(f"{base_dir}/{task.fn}").parent
    experiment_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(task.fn, experiment_dir / "experiment.log")
    
    circuits_directory = experiment_dir / "circuits"
    try:
        if recirq.exists(task, base_dir=base_dir) and Path():
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
                optimized_circuits = [cirq_optimize_circuit(task.qubits, circuit, processor_id=task.processor_id) for circuit in circuits]
            elif task.optimizer.startswith("bqskit"):
                optimization_level = int(task.optimizer[-1])
                machine_model = bqskit_machine_model(task.qubits, processor_id=task.processor_id)
                optimized_circuits = [bqskit_optimize_circuit(task.qubits, circuit, machine_model, optimization_level=optimization_level) for circuit in circuits]
            cirq.to_json(optimized_circuits, str(optimized_path))
        else:
            optimized_circuits = cirq.read_json(str(optimized_path))
        
        logger.info(f"Sampling...")
        sampler = get_sampler(task.processor_id, run_type=task.run_type)
        results = sampler.run_batch(programs=optimized_circuits, repetitions=task.n_shots)

        logger.info(f"Processing results...")
        data = task.process_results(results=results)

        logger.info(f"Saving...")
        recirq.save(task=task, data={"processed_data": data}, base_dir=base_dir)
    except Exception as e:
        logger.exception("Help!") 

def run_sky_ground_tasks(specs, base_dir=None):
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR
    for task_type in sky_ground_tasks:
        if specs["wh_implementation"] == "simple" and task_type == BasisMeasurementAfterWHPOVMOnBasisStatesTask:
            continue
        run_sky_ground_task(task_from_specs(task_type, specs), base_dir=base_dir)

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
    fiducial_description: str
    wh_implementation: str

    fiducial: Optional[np.array] = None
    fiducial_circuit: Optional[cirq.Circuit] = None

    @classmethod
    def filename(cls, **specs):
        return (f"{specs['dataset_id']}/"
                f"d{specs['d']}/"
                f"{cls.__name__}/"
                f"{specs['wh_implementation']}/"
                f"{specs['fiducial_description']}/"
                f"{specs['optimizer']}_{specs['run_type']}_n{abbrev_n_shots(specs['n_shots'])}_{specs['processor_id']}_q{abbrev_grid_qubits(specs['qubits'])}")

    @property
    def fn(self):
        return CharacterizeWHReferenceDeviceTask.filename(**self.__dict__)

    def make_circuits(self):
        a = [[a1, a2] for a1 in range(self.d) for a2 in range(self.d)]
        n = int(np.log2(self.d))
        if type(self.fiducial_circuit) != type(None):
            prepare_fiducial = self.fiducial_circuit
        else:
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
    
    def process_results(self, results=None, probs=None):
        P = results_to_freqs(results) if type(probs) == type(None) else probs  
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
    fiducial_description: str
    wh_implementation: str

    fiducial: Optional[np.array] = None
    fiducial_circuit: Optional[cirq.Circuit] = None

    @classmethod
    def filename(cls, **specs):
        return (f"{specs['dataset_id']}/"
                f"d{specs['d']}/"
                f"{cls.__name__}/"
                f"{specs['wh_implementation']}/"
                f"{specs['fiducial_description']}/"
                f"{specs['optimizer']}_{specs['run_type']}_n{abbrev_n_shots(specs['n_shots'])}_{specs['processor_id']}_q{abbrev_grid_qubits(specs['qubits'])}")

    @property
    def fn(self):
        return self.__class__.filename(**self.__dict__)
    
    def make_circuits(self):
        n = int(np.log2(self.d))
        m = np.arange(self.d)
        if type(self.fiducial_circuit) != type(None):
            prepare_fiducial = self.fiducial_circuit
        else:
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
    
    def process_results(self, results=None, probs=None):
        r = results_to_freqs(results) if type(probs) == type(None) else probs  
        if self.wh_implementation == "ak":
            r = change_conjugate_convention(r)
        r = r.T
        return {"r": r}
    
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
    fiducial_description: str

    fiducial: Optional[np.array] = None
    fiducial_circuit: Optional[cirq.Circuit] = None

    @classmethod
    def filename(cls, **specs):
        return (f"{specs['dataset_id']}/"
                f"d{specs['d']}/"
                f"{cls.__name__}/"
                f"{specs['fiducial_description']}/"
                f"{specs['optimizer']}_{specs['run_type']}_n{abbrev_n_shots(specs['n_shots'])}_{specs['processor_id']}_q{abbrev_grid_qubits(specs['qubits'])}")

    @property
    def fn(self):
        return self.__class__.filename(**self.__dict__)
    
    def make_circuits(self):
        n = int(np.log2(self.d))
        a = [[a1, a2] for a1 in range(self.d) for a2 in range(self.d)]
        if type(self.fiducial_circuit) != type(None):
            prepare_fiducial = self.fiducial_circuit
        else:
            prepare_fiducial = ansatz_circuit(self.fiducial)
        state_qubits = self.qubits[:n]
        circuits = [cirq.Circuit((wh_state(state_qubits, prepare_fiducial, a1, a2),\
                                  cirq.measure(state_qubits, key="result")))\
                                        for a1, a2 in a]
        return circuits
    
    def process_results(self, results=None, probs=None):
        C = results_to_freqs(results) if type(probs) == type(None) else probs  
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

    @classmethod
    def filename(cls, **specs):
        return (f"{specs['dataset_id']}/"
                f"d{specs['d']}/"
                f"{cls.__name__}/"
                f"{specs['optimizer']}_{specs['run_type']}_n{abbrev_n_shots(specs['n_shots'])}_{specs['processor_id']}_q{abbrev_grid_qubits(specs['qubits'])}")

    @property
    def fn(self):
        return self.__class__.filename(**self.__dict__)
    
    def make_circuits(self):
        n = int(np.log2(self.d))
        m = np.arange(self.d)
        state_qubits = self.qubits[:n]
        circuits = [cirq.Circuit((qudit_basis_state(state_qubits, i),
                                  cirq.measure(state_qubits, key="result"))) for i in m]
        return circuits
    
    def process_results(self, results=None, probs=None):
        q = results_to_freqs(results) if type(probs) == type(None) else probs  
        q = q.T
        return {"q": q}

####################################################################################

@recirq.json_serializable_dataclass(namespace="recirq.sky_ground", 
                                    registry=recirq.Registry,
                                    frozen=True)
class BasisMeasurementAfterWHPOVMOnBasisStatesTask:
    dataset_id: str
    processor_id: str
    run_type: str
    qubits: list
    n_shots: int
    optimizer: str

    d: int
    fiducial_description: str

    fiducial: Optional[np.array] = None
    fiducial_circuit: Optional[cirq.Circuit] = None

    @classmethod
    def filename(cls, **specs):
        return (f"{specs['dataset_id']}/"
                f"d{specs['d']}/"
                f"{cls.__name__}/"
                f"{specs['fiducial_description']}/"
                f"{specs['optimizer']}_{specs['run_type']}_n{abbrev_n_shots(specs['n_shots'])}_{specs['processor_id']}_q{abbrev_grid_qubits(specs['qubits'])}")

    @property
    def fn(self):
        return self.__class__.filename(**self.__dict__)
    
    def make_circuits(self):
        n = int(np.log2(self.d))
        m = np.arange(self.d)
        if type(self.fiducial_circuit) != type(None):
            prepare_fiducial = self.fiducial_circuit
        else:
            prepare_fiducial = ansatz_circuit(self.fiducial)
        ancilla1 = self.qubits[:n]
        ancilla2 = self.qubits[n:2*n]
        state_qubits = self.qubits[2*n:3*n]
        circuits = [cirq.Circuit((qudit_basis_state(state_qubits, i),\
                                  arthurs_kelly(state_qubits, ancilla1, ancilla2, prepare_fiducial=prepare_fiducial, measure=False),
                                  cirq.measure(state_qubits, key="result")))\
                                    for i in m]
        return circuits
    
    def process_results(self, results=None, probs=None):
        p = results_to_freqs(results) if type(probs) == type(None) else probs  
        p = p.T
        return {"p": p}

####################################################################################

@recirq.json_serializable_dataclass(namespace="recirq.sky_ground", 
                                    registry=recirq.Registry,
                                    frozen=True)
class WHPOVMOnStatesTask:
    dataset_id: str
    processor_id: str
    run_type: str
    qubits: list
    n_shots: int
    optimizer: str

    d: int
    fiducial_description: str
    states_description: str
    wh_implementation: str

    fiducial: Optional[np.array] = None
    fiducial_circuit: Optional[cirq.Circuit] = None
    states: Optional[list] = None
    states_circuits: Optional[list] = None

    @classmethod
    def filename(cls, **specs):
        return (f"{specs['dataset_id']}/"
                f"d{specs['d']}/"
                f"{cls.__name__}/"
                f"{specs['wh_implementation']}/"
                f"{specs['fiducial_description']}/"
                f"{specs['states_description']}/"
                f"{specs['optimizer']}_{specs['run_type']}_n{abbrev_n_shots(specs['n_shots'])}_{specs['processor_id']}_q{abbrev_grid_qubits(specs['qubits'])}")

    @property
    def fn(self):
        return self.__class__.filename(**self.__dict__)
    
    def make_circuits(self):
        n = int(np.log2(self.d))
        if type(self.fiducial_circuit) != type(None):
            prepare_fiducial = self.fiducial_circuit
        else:
            prepare_fiducial = ansatz_circuit(self.fiducial)
        if type(self.states_circuits) != type(None):
            prepare_states = self.states_circuits
        else:
            prepare_states = [ansatz_circuit(state) for state in self.states]

        if self.wh_implementation == "simple":
            state_qubits = self.qubits[:n]
            fiducial_qubits = self.qubits[n:2*n]
            circuits = [cirq.Circuit((prepare_state(state_qubits),\
                                      simple_wh_povm(state_qubits, fiducial_qubits, prepare_fiducial=prepare_fiducial, measure=True)))\
                                        for prepare_state in prepare_states]
        elif self.wh_implementation == "ak":
            ancilla1 = self.qubits[:n]
            ancilla2 = self.qubits[n:2*n]
            state_qubits = self.qubits[2*n:3*n]
            circuits = [cirq.Circuit((prepare_state(state_qubits),\
                                      arthurs_kelly(state_qubits, ancilla1, ancilla2, prepare_fiducial=prepare_fiducial, measure=True)))\
                                        for prepare_state in prepare_states]
        return circuits
    
    def process_results(self, results=None, probs=None):
        r = results_to_freqs(results) if type(probs) == type(None) else probs  
        if self.wh_implementation == "ak":
            r = change_conjugate_convention(r)
        r = r.T
        return {"r": r}
    
####################################################################################

sky_ground_tasks = [CharacterizeWHReferenceDeviceTask,
                    WHPOVMOnBasisStatesTask,
                    BasisMeasurementOnWHStatesTask,
                    BasisMeasurementOnBasisStatesTask,
                    BasisMeasurementAfterWHPOVMOnBasisStatesTask]