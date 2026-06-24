import os
import re
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

def get_sampler(processor_id, run_type="noisy", project_id=default_project_id, circuits=None):
    """Return a sampler appropriate for the requested execution mode.

    Dynamic circuits with mid-circuit measurement and classical feed-forward are
    routed to Cirq's built-in simulators because qsim and the cirq_google
    validating sampler do not currently support those operations.
    """
    if circuits is None:
        circuit_list = []
    elif isinstance(circuits, cirq.Circuit):
        circuit_list = [circuits]
    else:
        circuit_list = list(circuits)

    has_dynamic = any(has_dynamic_circuit_features(circuit) for circuit in circuit_list)

    if run_type == "real" and has_dynamic:
        raise ValueError(
            "Dynamic circuits with classical feed-forward are not supported by the current "
            "cirq_google device sampler. Use run_type='clean' or 'noisy' for wh_implementation='msimple'."
        )

    if run_type == "real":
        engine = cirq_google.Engine(project_id=project_id)
        sampler = engine.get_sampler(processor_id=processor_id) # REWRITE!!!
    else:
        if run_type == "noisy":
            noise_props = cirq_google.engine.load_device_noise_properties(processor_id)
            noise_model = cirq_google.NoiseModelFromGoogleNoiseProperties(noise_props)
            sampler = cirq.Simulator(noise=noise_model) if has_dynamic else qsimcirq.QSimSimulator(noise=noise_model)
        elif run_type == "clean":
            sampler = cirq.Simulator() if has_dynamic else qsimcirq.QSimSimulator()
    return sampler

####################################################################################

def exact_simulation(circuit):
    """Returns the exact outcome probabilities for a cirq circuit."""
    if not are_all_measurements_terminal(circuit):
        raise ValueError("exact_simulation only supports circuits with terminal measurements.")
    measurements = [op for op in circuit.all_operations() if isinstance(op.gate, cirq.MeasurementGate)]
    circuit_sans_measurements = cirq.drop_terminal_measurements(circuit)
    result = cirq.Simulator().simulate(circuit_sans_measurements)
    if len(measurements) == 1 and measurements[0].gate.key == "result":
        measured_qubits = list(measurements[0].qubits)
    else:
        ordered_measurements = sorted(measurements, key=lambda op: measurement_key_sort_key(op.gate.key))
        measured_qubits = [qubit for op in ordered_measurements for qubit in op.qubits]
    return np.diag(result.density_matrix_of(measured_qubits)).real

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
    if has_dynamic_circuit_features(circuit):
        raise ValueError("bqskit optimization does not support dynamic circuits with mid-circuit measurement.")
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
    final_circuit.append(sorted(measurements, key=lambda op: measurement_key_sort_key(op.gate.key)))
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

def terminal_measurement_ops(circuit: cirq.Circuit):
    """Collect terminal measurement operations in register order."""
    return sorted(
        [op for op in circuit.all_operations() if cirq.is_measurement(op)],
        key=lambda op: measurement_key_sort_key(op.gate.key),
    )

def cirq_optimize_circuit(qubits, circuit, processor_id="willow_pink"):
    """Conform the circuit to device topology and gateset."""
    device = cirq_google.engine.create_device_from_processor_id(processor_id)
    gateset = device.metadata.compilation_target_gatesets[0]
    connectivity_graph = device.metadata.nx_graph

    mapping = dict([(q,q) for q in qubits])
    router = cirq.RouteCQC(connectivity_graph)

    if are_all_measurements_terminal(circuit):
        measurement_ops = terminal_measurement_ops(circuit)
        circuit = cirq.drop_terminal_measurements(circuit)
    else:
        measurement_ops = []

    routed_circuit, _, final_map = router.route_circuit(circuit, initial_mapper=cirq.HardCodedInitialMapper(mapping))
    finished_circuit = cirq.optimize_for_target_gateset(routed_circuit,\
                            context=cirq.TransformerContext(deep=True), gateset=gateset)
    if measurement_ops:
        measurement_ops = [op.transform_qubits(final_map) for op in measurement_ops]
        finished_circuit.append(cirq.Moment(measurement_ops))
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

MEASUREMENT_PREFIX_ORDER = {"s": 0, "a": 1, "a1": 1, "a2": 2}

def measurement_key_sort_key(key):
    """Sort measurement keys like s_0, a_0, a1_0, a2_0 into register order."""
    if key == "result":
        return (-1, "result", -1)
    match = re.fullmatch(r"([A-Za-z][A-Za-z0-9]*)_(\d+)", key)
    if match is None:
        return (99, key, -1)
    prefix, index = match.groups()
    return (MEASUREMENT_PREFIX_ORDER.get(prefix, 98), prefix, int(index))

def ordered_measurement_keys(measurements):
    """Return measurement keys in the register order used throughout the codebase."""
    return sorted(measurements.keys(), key=measurement_key_sort_key)

def are_all_measurements_terminal(circuit):
    """Compatibility helper for Cirq versions without are_all_measurements_terminal."""
    try:
        stripped = cirq.drop_terminal_measurements(circuit)
    except ValueError:
        return False
    return not any(cirq.is_measurement(op) for op in stripped.all_operations())

def result_measurements_matrix(shots):
    """Stack a ResultDict's measurements into one big-endian bit matrix."""
    if "result" in shots.measurements:
        measurements = shots.measurements["result"]
        return measurements if measurements.ndim == 2 else measurements.reshape(measurements.shape[0], -1)

    ordered_keys = ordered_measurement_keys(shots.measurements)
    if not ordered_keys:
        raise ValueError("No measurements found in shots data.")

    columns = []
    for key in ordered_keys:
        values = shots.measurements[key]
        columns.append(values if values.ndim == 2 else values.reshape(values.shape[0], -1))
    return np.hstack(columns)

def has_dynamic_circuit_features(circuit):
    """Whether a circuit uses mid-circuit measurement or classical feed-forward."""
    return any(getattr(op, "classical_controls", ()) for op in circuit.all_operations()) or not are_all_measurements_terminal(circuit)

def shots_to_freqs(shots):
    """Get frequencies from shots data."""
    measurements = result_measurements_matrix(shots)
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
