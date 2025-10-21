import sys
import logging 

import numpy as np
import collections

import cirq 
import cirq_google
import qsimcirq
import recirq
from cirq.value import big_endian_bits_to_int

import bqskit as bq
from bqskit.ext import bqskit_to_cirq, cirq_to_bqskit

####################################################################################

def setup_logger(name, path):
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

def get_gate_counts(circuit):
    """Get gate counts for a cirq circuit."""
    all_gate_types = [type(op.gate) for op in circuit.all_operations()]
    type_counts = collections.Counter(all_gate_types)
    print("--- Gate Counts (by type) ---")
    for gate_type, count in type_counts.items():
        print(f"{gate_type.__name__}: {count}")

def abbrev_n_shots(n_shots: int) -> str:
    """Shorter n_shots component of a filename"""
    if n_shots % 1000 == 0:
        return f'{n_shots // 1000}k'
    return str(n_shots)

def abbrev_grid_qubits(qubits):
    """Formatted grid_qubits component of a filename"""
    return "-".join([f'{qubit.row}_{qubit.col}' for qubit in qubits])

####################################################################################

def get_sampler(processor_id, run_type="noisy", PROJECT_ID="cirq_sic"):
    """Returns the device, gateset, connectivity graph, and sampler as a dictionary.
        run_type='clean' gives an exact simulator.
        runtype='noisy' gives a simulator with a noise model.
        runtype='real' gives the real thing.
    """
    if run_type == "real":
        engine = cirq_google.Engine(project_id=PROJECT_ID)
        device = engine.get_processor(processor_id).get_device()
        sampler = engine.get_sampler(processor_id=processor_id)
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

bqskit_willow_gateset = [bq.ir.gates.PhasedXZGate(),
                         bq.ir.gates.CZGate(), 
                         bq.ir.gates.RXGate(),
                         bq.ir.gates.RYGate(),
                         bq.ir.gates.RZGate()]

def bqskit_machine_model(qubits, processor_id="willow_pink"):
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

def bqskit_optimize_circuit(qubits, circuit, machine_model, optimization_level=1):
    bq_circuit = cirq_to_bqskit(circuit)
    compiled_bq_circuit, initial_mapping, final_mapping = bq.compile(bq_circuit,\
                                                                     model=machine_model,\
                                                                     optimization_level=optimization_level,\
                                                                     with_mapping=True)
    compiled_circuit = bqskit_to_cirq(compiled_bq_circuit)
    named_qubits = [cirq.NamedQubit("q_%d" % i) for i in range(len(qubits))]
    qubit_map = dict([(nq, qubits[i]) for i, nq in enumerate(named_qubits)])
    finished_circuit = cirq.drop_negligible_operations(cirq.drop_terminal_measurements(compiled_circuit.transform_qubits(qubit_map)))
    final_measurement = None
    for op in circuit.all_operations():
        if isinstance(op.gate, cirq.MeasurementGate):
            final_measurement = op
            break 
    finished_circuit.append(final_measurement)
    return finished_circuit, final_mapping

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
    return finished_circuit, None

####################################################################################

def results_to_freqs(results, mapping=None):
    if mapping is not None:
        measurements = results.measurements["result"][:, mapping]
    else:
        measurements = results.measurements["result"]
    n_outcomes = 2**measurements.shape[1]
    int_outcomes = [big_endian_bits_to_int(bits) for bits in measurements]
    counts = collections.Counter(int_outcomes)
    for i in range(n_outcomes):
        if i not in counts:
            counts[i] = 0
    freqs = np.array([v for k, v in sorted(counts.items())])/counts.total()
    return freqs

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

