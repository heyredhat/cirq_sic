import inspect
from itertools import combinations_with_replacement, permutations
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
    """Depending on the dimension and WH implementation, returns suitable qubits.
    At the moment these are fixed to be a arbitrary but convienient choice."""
    n = int(np.ceil(np.log2(d)))
    if 2**n == d:
        cols = 2 if wh_implementation in {"simple", "msimple"} else 3
        return cirq.GridQubit.rect(cols, n, top=4, left=2)
    else:
        n = int(np.ceil(np.log2(d)) + 1)
        cols = 3 if wh_implementation in {"simple", "msimple"} else 4
        return cirq.GridQubit.rect(cols, n, top=4, left=2)

####################################################################################

EXPERIMENT_NAME = "sky_ground"
DEFAULT_BASE_DIR = f"data/{EXPERIMENT_NAME}"
SIMPLE_WH_IMPLEMENTATIONS = {"simple", "msimple"}

def has_post_measurement_task(wh_implementation):
    return wh_implementation == "ak"

def wh_povm_circuit(wh_implementation, state_qubits, prepare_fiducial, *, ancilla_qubits=None,\
                    ancilla1_qubits=None, ancilla2_qubits=None, exact=False, measure=True):
    """Return the appropriate WH-POVM circuit generator for the chosen implementation."""
    if wh_implementation == "ak":
        return arthurs_kelly(
            state_qubits,
            ancilla1_qubits,
            ancilla2_qubits,
            prepare_fiducial=prepare_fiducial,
            measure=measure,
        )
    if wh_implementation == "simple" or (wh_implementation == "msimple" and exact):
        return simple_wh_povm(
            state_qubits,
            ancilla_qubits,
            prepare_fiducial=prepare_fiducial,
            measure=measure,
        )
    if wh_implementation == "msimple":
        return msimple_wh_povm(
            state_qubits,
            ancilla_qubits,
            prepare_fiducial=prepare_fiducial,
            measure=measure,
        )
    raise ValueError(f"Unsupported WH implementation: {wh_implementation}")

def optimize_task_circuits(task, circuits, logger=None):
    """Optimize circuits, skipping dynamic circuits to preserve feed-forward order."""
    has_dynamic = any(has_dynamic_circuit_features(circuit) for circuit in circuits)
    if has_dynamic:
        if logger is not None:
            logger.info("Dynamic circuit detected; skipping optimization to preserve measurement/feed-forward order.")
        return circuits

    if task.optimizer.startswith("cirq"):
        return [cirq_optimize_circuit(task.qubits, circuit, processor_id=task.processor_id) for circuit in circuits]

    if task.optimizer.startswith("bqskit"):
        tokens = task.optimizer.split("_")
        optimization_level = int(tokens[-1])
        machine_model = bqskit_machine_model(task.qubits, processor_id=task.processor_id)
        return [
            bqskit_optimize_circuit(
                task.qubits,
                circuit,
                machine_model,
                optimization_level=optimization_level,
                server=tokens[1],
            )
            for circuit in circuits
        ]

    raise ValueError(f"Unsupported optimizer: {task.optimizer}")

def load_results(task, base_dir=None):
    """Loads the results associated with a task from a directory structure rooted in base_dir."""
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR
    return recirq.read_json(f"{base_dir}/{task.fn}.json")

def load_circuits(task, base_dir=None, raw=False):
    """Loads the circuits associated with a task from a directory structure rooted in base_dir. If raw=True, returns the unoptimized circuits."""
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
    """Given a specification dictionary, returns the results from a full battery of sky/ground tasks. 
    If `separate=True`, returns the a tuple: a task list and a dictionary of results."""
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR
    results = {}
    for task_type in sky_ground_tasks:
        if not has_post_measurement_task(specs["wh_implementation"]) and task_type == BasisMeasurementAfterWHPOVMOnBasisStatesTask:
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
    """Given a task, loads its circuits, and performs an exact numerical calculation of the resulting probabilities, which are processed in accordance with the task."""
    circuits = task.make_exact_circuits() if hasattr(task, "make_exact_circuits") else task.make_circuits()
    return task.process_results(probs=np.array([exact_simulation(circuit) for i, circuit in enumerate(circuits)]))

####################################################################################

def task_from_specs(task_type, specs):
    """Builds a task of class `task_type` given specification dictionary `specs`. """
    sig = inspect.signature(task_type).parameters
    task = task_type(**{k: specs[k] for k in sig.keys() if k in specs})
    return task

def run_sky_ground_task_from_specs(task_type, specs, base_dir=None):
    """Runs a sky ground task of class `task_type` and specification dictionary `specs`."""
    run_sky_ground_task(task_from_specs(task_type, specs), base_dir=base_dir) 

def run_sky_ground_tasks(specs, sampler=None, base_dir=None):
    """Given a specification dictionary, runs all the sky/ground tasks."""
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR
    for task_type in sky_ground_tasks:
        if not has_post_measurement_task(specs["wh_implementation"]) and task_type == BasisMeasurementAfterWHPOVMOnBasisStatesTask:
            continue
        run_sky_ground_task(task_from_specs(task_type, specs), sampler=sampler, base_dir=base_dir)

def run_sky_ground_task(task, sampler=None, base_dir=None):
    """Runs the sky/ground task. Builds the directory structure, the circuits, optimizes them, samples them, and processes them."""
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
            optimized_circuits = optimize_task_circuits(task, circuits, logger=logger)
            cirq.to_json(optimized_circuits, str(optimized_path))
        else:
            optimized_circuits = cirq.read_json(str(optimized_path))
        
        logger.info(f"Sampling...")
        if type(sampler) == type(None):
            sampler = get_sampler(task.processor_id, run_type=task.run_type, circuits=optimized_circuits)
        results = sampler.run_batch(programs=optimized_circuits, repetitions=task.n_shots)

        logger.info(f"Processing results...")
        data = task.process_results(results=results)

        logger.info(f"Saving...")
        recirq.save(task=task, data={"processed_data": data}, base_dir=base_dir)
    except Exception as e:
        logger.exception("Help!") 

####################################################################################

def create_circuits(task, base_dir=None):
    """Creates (and optimizes) circuits for a given task."""
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR

    experiment_dir = Path(f"{base_dir}/{task.fn}").parent
    experiment_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(task.fn, experiment_dir / "experiment.log")
    try:
        if recirq.exists(task, base_dir=base_dir) and Path():
            logger.info(f"Task already exists. Skipping.")
            return

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
            optimized_circuits = optimize_task_circuits(task, circuits, logger=logger)
            cirq.to_json(optimized_circuits, str(optimized_path))
        else:
            optimized_circuits = cirq.read_json(str(optimized_path))

        return optimized_circuits
    except Exception as e:
        logger.exception("Help!") 

def run_circuits(task, base_dir=None):
    """Loads and runs circuits for a task, returning Results."""
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR

    experiment_dir = Path(f"{base_dir}/{task.fn}").parent
    logger = setup_logger(task.fn, experiment_dir / "experiment.log")
    try:
        if recirq.exists(task, base_dir=base_dir) and Path():
            logger.info(f"Task already exists. Skipping.")
            return

        circuits_directory = experiment_dir / "circuits"
        optimized_path = circuits_directory / f"{task.optimizer}_optimized.json"
        optimized_circuits = cirq.read_json(str(optimized_path))
        
        logger.info(f"Sampling...")
        sampler = get_sampler(task.processor_id, run_type=task.run_type, circuits=optimized_circuits)
        results = sampler.run_batch(programs=optimized_circuits, repetitions=task.n_shots)

        return results

    except Exception as e:
        logger.exception("Help!") 

def process_task_results(task, results, base_dir=None):
    """Given task and results, processes the sky/ground results, saving them to a json file with recirq."""
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR

    experiment_dir = Path(f"{base_dir}/{task.fn}").parent
    experiment_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(task.fn, experiment_dir / "experiment.log")
    try:
        if recirq.exists(task, base_dir=base_dir) and Path():
            logger.info(f"Task already exists. Skipping.")
            return
        
        logger.info(f"Processing results...")
        data = task.process_results(results=results)

        logger.info(f"Saving...")
        recirq.save(task=task, data={"processed_data": data}, base_dir=base_dir)

        return data
    except Exception as e:
        logger.exception("Help!") 

####################################################################################

@recirq.json_serializable_dataclass(namespace="recirq.sky_ground", 
                                    registry=recirq.Registry,
                                    frozen=True)
class CharacterizeWHReferenceDeviceTask:
    """For obtaining the probabilities of a WH POVM outcome given a WH covariant state."""
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
        state_qubits = self.qubits[:n]
        if self.wh_implementation in SIMPLE_WH_IMPLEMENTATIONS:
            ancilla_qubits = self.qubits[n:2*n]
            return [
                cirq.Circuit(
                    wh_state(state_qubits, prepare_fiducial, a1, a2),
                    wh_povm_circuit(
                        self.wh_implementation,
                        state_qubits,
                        prepare_fiducial,
                        ancilla_qubits=ancilla_qubits,
                        measure=True,
                    ),
                )
                for a1, a2 in a
            ]
        if self.wh_implementation == "ak":
            ancilla1_qubits = self.qubits[n:2*n]
            ancilla2_qubits = self.qubits[2*n:3*n]
            return [
                cirq.Circuit(
                    wh_state(state_qubits, prepare_fiducial, a1, a2),
                    wh_povm_circuit(
                        self.wh_implementation,
                        state_qubits,
                        prepare_fiducial,
                        ancilla1_qubits=ancilla1_qubits,
                        ancilla2_qubits=ancilla2_qubits,
                        measure=True,
                    ),
                )
                for a1, a2 in a
            ]
        raise ValueError(f"Unsupported WH implementation: {self.wh_implementation}")

    def make_exact_circuits(self):
        if self.wh_implementation != "msimple":
            return self.make_circuits()

        a = [[a1, a2] for a1 in range(self.d) for a2 in range(self.d)]
        n = int(np.log2(self.d))
        prepare_fiducial = self.fiducial_circuit if type(self.fiducial_circuit) != type(None) else ansatz_circuit(self.fiducial)
        state_qubits = self.qubits[:n]
        ancilla_qubits = self.qubits[n:2*n]
        return [
            cirq.Circuit(
                wh_state(state_qubits, prepare_fiducial, a1, a2),
                wh_povm_circuit(
                    "msimple",
                    state_qubits,
                    prepare_fiducial,
                    ancilla_qubits=ancilla_qubits,
                    exact=True,
                    measure=True,
                ),
            )
            for a1, a2 in a
        ]
    
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
    """For obtaining the probabilities of a WH POVM outcome given computational basis states."""
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
        state_qubits = self.qubits[:n]
        if self.wh_implementation in SIMPLE_WH_IMPLEMENTATIONS:
            ancilla_qubits = self.qubits[n:2*n]
            return [
                cirq.Circuit(
                    qudit_basis_state(state_qubits, i),
                    wh_povm_circuit(
                        self.wh_implementation,
                        state_qubits,
                        prepare_fiducial,
                        ancilla_qubits=ancilla_qubits,
                        measure=True,
                    ),
                )
                for i in m
            ]
        if self.wh_implementation == "ak":
            ancilla1_qubits = self.qubits[n:2*n]
            ancilla2_qubits = self.qubits[2*n:3*n]
            return [
                cirq.Circuit(
                    qudit_basis_state(state_qubits, i),
                    wh_povm_circuit(
                        self.wh_implementation,
                        state_qubits,
                        prepare_fiducial,
                        ancilla1_qubits=ancilla1_qubits,
                        ancilla2_qubits=ancilla2_qubits,
                        measure=True,
                    ),
                )
                for i in m
            ]
        raise ValueError(f"Unsupported WH implementation: {self.wh_implementation}")

    def make_exact_circuits(self):
        if self.wh_implementation != "msimple":
            return self.make_circuits()

        n = int(np.log2(self.d))
        m = np.arange(self.d)
        prepare_fiducial = self.fiducial_circuit if type(self.fiducial_circuit) != type(None) else ansatz_circuit(self.fiducial)
        state_qubits = self.qubits[:n]
        ancilla_qubits = self.qubits[n:2*n]
        return [
            cirq.Circuit(
                qudit_basis_state(state_qubits, i),
                wh_povm_circuit(
                    "msimple",
                    state_qubits,
                    prepare_fiducial,
                    ancilla_qubits=ancilla_qubits,
                    exact=True,
                    measure=True,
                ),
            )
            for i in m
        ]
    
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
    """For obtaining the probabilities of a computational basis outcome given WH covariant states."""
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
        circuits = [
            cirq.Circuit(
                wh_state(state_qubits, prepare_fiducial, a1, a2),
                measure_register(state_qubits, "s"),
            )
            for a1, a2 in a
        ]
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
    """For obtaining the probabilities of a computational basis outcome given computational basis states."""
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
        circuits = [
            cirq.Circuit(
                qudit_basis_state(state_qubits, i),
                measure_register(state_qubits, "s"),
            )
            for i in m
        ]
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
    """For obtaining the probabilities of a computational basis outcome after a WH-POVM has been performed on computational basis states.
    Only works in Arthurs-Kelly mode: after the basis state is prepared, the AK interaction is performed (but not the measurement itself), 
    and then the computational basis measurement."""
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
        state_qubits = self.qubits[:n]
        ancilla1_qubits = self.qubits[n:2*n]
        ancilla2_qubits = self.qubits[2*n:3*n]
        circuits = [
            cirq.Circuit(
                qudit_basis_state(state_qubits, i),
                arthurs_kelly(
                    state_qubits,
                    ancilla1_qubits,
                    ancilla2_qubits,
                    prepare_fiducial=prepare_fiducial,
                    measure=False,
                ),
                measure_register(state_qubits, "s"),
            )
            for i in m
        ]
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
    """WH-POVM measurement on an arbitrary set of states."""
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

        state_qubits = self.qubits[:n]
        if self.wh_implementation in SIMPLE_WH_IMPLEMENTATIONS:
            ancilla_qubits = self.qubits[n:2*n]
            return [
                cirq.Circuit(
                    prepare_state(state_qubits),
                    wh_povm_circuit(
                        self.wh_implementation,
                        state_qubits,
                        prepare_fiducial,
                        ancilla_qubits=ancilla_qubits,
                        measure=True,
                    ),
                )
                for prepare_state in prepare_states
            ]
        if self.wh_implementation == "ak":
            ancilla1_qubits = self.qubits[n:2*n]
            ancilla2_qubits = self.qubits[2*n:3*n]
            return [
                cirq.Circuit(
                    prepare_state(state_qubits),
                    wh_povm_circuit(
                        self.wh_implementation,
                        state_qubits,
                        prepare_fiducial,
                        ancilla1_qubits=ancilla1_qubits,
                        ancilla2_qubits=ancilla2_qubits,
                        measure=True,
                    ),
                )
                for prepare_state in prepare_states
            ]
        raise ValueError(f"Unsupported WH implementation: {self.wh_implementation}")

    def make_exact_circuits(self):
        if self.wh_implementation != "msimple":
            return self.make_circuits()

        n = int(np.log2(self.d))
        prepare_fiducial = self.fiducial_circuit if type(self.fiducial_circuit) != type(None) else ansatz_circuit(self.fiducial)
        prepare_states = self.states_circuits if type(self.states_circuits) != type(None) else [ansatz_circuit(state) for state in self.states]
        state_qubits = self.qubits[:n]
        ancilla_qubits = self.qubits[n:2*n]
        return [
            cirq.Circuit(
                prepare_state(state_qubits),
                wh_povm_circuit(
                    "msimple",
                    state_qubits,
                    prepare_fiducial,
                    ancilla_qubits=ancilla_qubits,
                    exact=True,
                    measure=True,
                ),
            )
            for prepare_state in prepare_states
        ]
    
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
class CharacterizeWHTripleProductsTask:
    """Real parts of WH-orbit triple products via the three-state cycle test."""
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

    def symmetric_triples(self):
        n_states = self.d**2
        return list(combinations_with_replacement(range(n_states), 3))

    def make_circuits(self):
        n = int(np.log2(self.d))
        if 2**n != self.d:
            raise ValueError("CharacterizeWHTripleProductsTask currently requires d = 2**n.")
        if len(self.qubits) < 1 + 3 * n:
            raise ValueError(f"Need at least {1 + 3*n} qubits for the triple-product cycle test.")

        prepare_fiducial = self.fiducial_circuit if type(self.fiducial_circuit) != type(None) else ansatz_circuit(self.fiducial)
        orbit_indices = [(a1, a2) for a1 in range(self.d) for a2 in range(self.d)]
        prepare_states = [
            (lambda qubits, a1=a1, a2=a2: wh_state(qubits, prepare_fiducial, a1, a2))
            for a1, a2 in orbit_indices
        ]

        control_qubit = self.qubits[0]
        register1 = self.qubits[1 : 1 + n]
        register2 = self.qubits[1 + n : 1 + 2 * n]
        register3 = self.qubits[1 + 2 * n : 1 + 3 * n]

        return [
            triple_product_measurement_circuit(
                control_qubit,
                register1,
                register2,
                register3,
                prepare_states[i],
                prepare_states[j],
                prepare_states[k],
            )
            for i, j, k in self.symmetric_triples()
        ]

    def process_results(self, results=None, probs=None):
        probabilities = results_to_freqs(results) if type(probs) == type(None) else np.asarray(probs)
        values = real_triple_product_from_probabilities(probabilities)

        n_states = self.d**2
        tensor = np.zeros((n_states, n_states, n_states))
        for value, triple in zip(values, self.symmetric_triples()):
            for perm in set(permutations(triple)):
                tensor[perm] = value

        return {"T": tensor}
    
####################################################################################

sky_ground_tasks = [CharacterizeWHReferenceDeviceTask,
                    WHPOVMOnBasisStatesTask,
                    BasisMeasurementOnWHStatesTask,
                    BasisMeasurementOnBasisStatesTask,
                    BasisMeasurementAfterWHPOVMOnBasisStatesTask]
