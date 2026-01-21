import os
import sys
import logging 
from pathlib import Path

import numpy as np
import collections

import cirq 
import cirq_google
import qsimcirq
import recirq
from cirq.value import big_endian_bits_to_int

import bqskit as bq
from bqskit.ext import bqskit_to_cirq, cirq_to_bqskit

default_project_id = "sic-povms-sandbox-444403"

####################################################################################

def setup_logger(name, path):
    """Initialize logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger

####################################################################################

def abbrev_n_shots(n_shots: int) -> str:
    """Shorter n_shots component of a filename"""
    if n_shots % 1000 == 0:
        return f'{n_shots // 1000}k'
    return str(n_shots)

def abbrev_grid_qubits(qubits):
    """Formatted grid_qubits component of a filename"""
    return "-".join([f'{qubit.row}_{qubit.col}' for qubit in qubits])

####################################################################################

def get_sampler(processor_id, run_type="noisy", project_id=default_project_id):
    """Returns the device, gateset, connectivity graph, and sampler as a dictionary.
        run_type='clean' gives an exact simulator.
        runtype='noisy' gives a simulator with a noise model.
        runtype='real' gives the real thing.
    """
    if run_type == "real":
        engine = cirq_google.Engine(project_id=project_id)
        device = engine.get_processor(processor_id).get_device()
        sampler = engine.get_sampler(processor_id=processor_id) # REWRITE!!!
    else:
        device = cirq_google.engine.create_device_from_processor_id(processor_id)
        if run_type == "noisy":
            noise_props = cirq_google.engine.load_device_noise_properties(processor_id)
            noise_model = cirq_google.NoiseModelFromGoogleNoiseProperties(noise_props)
            sim = qsimcirq.QSimSimulator(noise=noise_model)
        elif run_type == "clean":
            sim = qsimcirq.QSimSimulator()
        cal = cirq_google.engine.load_median_device_calibration(processor_id)
        sim_processor = cirq_google.engine.SimulatedLocalProcessor(
            processor_id=processor_id, sampler=sim, device=device, calibrations={cal.timestamp // 1000: cal})
        sim_engine = cirq_google.engine.SimulatedLocalEngine([sim_processor])
        sampler = sim_engine.get_sampler(processor_id)
    return sampler

####################################################################################

def exact_simulation(circuit):
    """Returns the exact outcome probabilities for a cirq circuit."""
    measurements = [op for op in circuit.all_operations() if isinstance(op.gate, cirq.MeasurementGate)]
    circuit_sans_measurements = cirq.drop_terminal_measurements(circuit)
    qubits = list(circuit_sans_measurements.all_qubits())
    result = cirq.Simulator().simulate(circuit_sans_measurements)
    return np.diag(result.density_matrix_of(measurements[0].qubits)).real

####################################################################################

bqskit_willow_gateset = [bq.ir.gates.PhasedXZGate(),
                         bq.ir.gates.CZGate(), 
                         bq.ir.gates.RXGate(),
                         bq.ir.gates.RYGate(),
                         bq.ir.gates.RZGate()]

def bqskit_machine_model(qubits, processor_id="willow_pink"):
    """Construct bqskit machine model for a cirq processor."""
    device = cirq_google.engine.create_device_from_processor_id(processor_id)
    gateset = device.metadata.compilation_target_gatesets[0]
    connectivity_graph = device.metadata.nx_graph
    qubit_index = dict([(q, i) for i, q in enumerate(device.metadata.qubit_set)])
    qubit_pairs = list(device.metadata.nx_graph.edges)
    full_coupling_graph = [(qubit_index[a], qubit_index[b]) for a, b in qubit_pairs]
    used_indices = [qubit_index[qubit] for qubit in qubits]
    restricted_coupling_graph = [pair for pair in full_coupling_graph if pair[0] in used_indices and pair[1] in used_indices]
    coupling_graph = [(used_indices.index(pair[0]), used_indices.index(pair[1])) for pair in restricted_coupling_graph]
    model = bq.MachineModel(len(qubits), gate_set=bqskit_willow_gateset, coupling_graph=coupling_graph)
    return model

def bqskit_optimize_circuit(qubits, circuit, machine_model, optimization_level=1, server="local"):
    """Optimize cirq circuit using bqskit, given a machine model. Server can be local or localhost (use the latter for runs in parallel)."""
    bq_circuit = cirq_to_bqskit(circuit)
    compiled_bq_circuit, initial_mapping, final_mapping = bq.compile(bq_circuit,\
                                                                     model=machine_model,\
                                                                     optimization_level=optimization_level,\
                                                                     with_mapping=True,\
                                                                     ip = None if server == "local" else "localhost")
    compiled_circuit = bqskit_to_cirq(compiled_bq_circuit)
    qubit_map = {cirq.NamedQubit("q_%d" % i): qubit for i, qubit in enumerate(qubits)}
    compiled_circuit = compiled_circuit.transform_qubits(qubit_map)

    measurements = [op for op in compiled_circuit.all_operations() if isinstance(op.gate, cirq.MeasurementGate)]
    final_circuit = cirq.drop_terminal_measurements(compiled_circuit)
    to_measure = {int(measurement.gate.key.split("_")[-1]): measurement.qubits[0] for measurement in measurements}
    final_circuit.append(cirq.measure(*[to_measure[i] for i in range(len(to_measure))], key="result"))
    return final_circuit

def to_one_and_two_qubit_ops(circuit: cirq.Circuit) -> cirq.Circuit:
    """Decompose circuit into one and two qubit operators. (Defunct?)"""
    def keep(op: cirq.Operation) -> bool:
        if cirq.is_measurement(op):
            return True
        return len(op.qubits) <= 2  # accept only 1- and 2-qubit non-measurement ops

    decomposed_ops = []
    for op in circuit.all_operations():
        decomposed_ops.extend(cirq.decompose(op, keep=keep, on_stuck_raise=True))
    return cirq.Circuit(decomposed_ops)

def push_measurements_to_end(circuit: cirq.Circuit) -> cirq.Circuit:
    """Push all measurements to end of circuit."""
    trailing = []
    kept = []
    for op in circuit.all_operations():
        if cirq.is_measurement(op):
            trailing.append(op)
        else:
            kept.append(op)
    return cirq.Circuit(kept + trailing)

def cirq_optimize_circuit(qubits, circuit, processor_id="willow_pink"):
    """Conform the circuit to device topology and gateset."""
    device = cirq_google.engine.create_device_from_processor_id(processor_id)
    gateset = device.metadata.compilation_target_gatesets[0]
    connectivity_graph = device.metadata.nx_graph

    mapping = dict([(q,q) for q in qubits])
    router = cirq.RouteCQC(connectivity_graph)
    routed_circuit, initial_map, final_map = router.route_circuit(circuit, initial_mapper=cirq.HardCodedInitialMapper(mapping))
    finished_circuit = cirq.optimize_for_target_gateset(routed_circuit,\
                            context=cirq.TransformerContext(deep=True), gateset=gateset)
    return finished_circuit

""" WORKING: Need a fix for arbitrary d. 
def cirq_optimize_circuit(qubits, circuit, processor_id="willow_pink"):
    device = cirq_google.engine.create_device_from_processor_id(processor_id)
    gateset = device.metadata.compilation_target_gatesets[0]
    connectivity_graph = device.metadata.nx_graph

    mapping = dict([(q,q) for q in qubits])
    router = cirq.RouteCQC(connectivity_graph)

    prepped_circuit = push_measurements_to_end(to_one_and_two_qubit_ops(circuit))
    measurement_ops = [op for op in prepped_circuit.all_operations() if cirq.is_measurement(op)]
    circuit_without_measurements = cirq.drop_terminal_measurements(prepped_circuit)

    routed_circuit, initial_map, final_map = router.route_circuit(circuit_without_measurements,\
                                                                  initial_mapper=cirq.HardCodedInitialMapper(mapping))
    finished_circuit = cirq.optimize_for_target_gateset(routed_circuit,\
                            context=cirq.TransformerContext(deep=True), gateset=gateset)

    if measurement_ops:
        finished_circuit += cirq.Circuit(measurement_ops)

    return finished_circuit
"""
####################################################################################

def shots_to_freqs(shots):
    """Get frequencies from shots data."""
    measurements = shots.measurements["result"]
    n_outcomes = 2**measurements.shape[1]
    int_outcomes = [big_endian_bits_to_int(bits) for bits in measurements]
    counts = collections.Counter(int_outcomes)
    for i in range(n_outcomes):
        if i not in counts:
            counts[i] = 0
    freqs = np.array([v for k, v in sorted(counts.items())])/counts.total()
    return freqs

def results_to_freqs(results):
    """Given cirq results, calculate frequencies."""
    return np.array([shots_to_freqs(r[0]) for i, r in enumerate(results)])

def avg_negativity(M):
    """Average negativity of the columns of M."""
    return np.sum((abs(M)-M)/2)/M.shape[1]

####################################################################################

def deep_match(obj, criteria):
    """Recursively checks if obj matches all key-value pairs in criteria at any depth."""
    if isinstance(obj, dict):
        if all((deep_match(obj.get(k, None), v) if isinstance(v, dict) else obj.get(k, None) == v) for k, v in criteria.items()):
            return True
        for v in obj.values():
            if deep_match(v, criteria):
                return True
    elif hasattr(obj, '__dict__'):
        attrs = vars(obj)
        if all((deep_match(getattr(obj, k, None), v) if isinstance(v, dict) else getattr(obj, k, None) == v) for k, v in criteria.items()):
            return True
        for v in attrs.values():
            if deep_match(v, criteria):
                return True
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            if deep_match(v, criteria):
                return True
    return False

def query_records(records, query):
    """Yields records that satisfy the query function."""
    return [record for record in records if deep_match(record, query)]

####################################################################################

def collect_circuit_files(base_dir):
    """Return {circuits_dir_path: [file names]} for every 'circuits' folder under base_dir."""
    base_path = Path(base_dir).expanduser().resolve()
    circuit_dirs: dict[Path, list[str]] = {}
    for dirpath, dirnames, filenames in os.walk(base_path):
        if Path(dirpath).name == "circuits":
            circuit_dirs[Path(dirpath)] = sorted(filenames)
    return circuit_dirs

def collect_circuits(base_dir):
    circuit_files = collect_circuit_files(base_dir)
    circuits = {}
    for k, v in circuit_files.items():
        before, during, after = str(k).partition(base_dir.rsplit("/", 1)[-1]+"/")
        circuits[during+after.removesuffix("/circuits")] = {f: cirq.read_json(k / f) for f in v}
    return circuits

def get_gate_counts(circuit, return_str=False):
    """Get gate counts for a cirq circuit."""
    counts = collections.Counter([type(op.gate).__name__ for op in circuit.all_operations() if not cirq.is_measurement(op)])
    if return_str:
        s = ", ".join([f"{gate_type}: {count}" for gate_type, count in counts.items()])
        return s + f". Total: {counts.total()}"
    else:
        return counts
    
def print_all_gate_counts(base_dir):
    circuits = collect_circuits(base_dir)
    for task_spec, circs in circuits.items():
        print(task_spec)
        for circ_type, circuits in circs.items():
            for i, circuit in enumerate(circuits):
                print(f"\t{circ_type} ({i}): [{get_gate_counts(circuit, return_str=True)}]")