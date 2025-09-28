import numpy as np

import cirq
import cirq_google
import qsimcirq
import recirq
from cirq.value import big_endian_bits_to_int

from ..utils import *
from ..circuits import *

####################################################################################

def get_device_data(processor_id, run_type="noisy", PROJECT_ID=""):
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
    gateset = device.metadata.compilation_target_gatesets[0]
    connectivity_graph = device.metadata.nx_graph
    return locals()

def process_circuit(circuit, connectivity_graph, gateset, qubits):
    """Conform the circuit to device topology and gateset."""
    mapping = dict([(q,q) for q in qubits])
    router = cirq.RouteCQC(connectivity_graph)
    routed_circuit, initial_map, final_map = router.route_circuit(circuit, initial_mapper=cirq.HardCodedInitialMapper(mapping))
    optimized_circuit = cirq.optimize_for_target_gateset(routed_circuit,\
                                context=cirq.TransformerContext(deep=True), gateset=gateset)
    return optimized_circuit

####################################################################################

def get_freqs(samples, n_shots=None):
    """From a Result, calculate the frequencies of the outcomes."""
    rng = np.random.default_rng()
    n_shots = samples.repetitions if n_shots is None else n_shots
    n_outcomes = 2**samples.measurements["result"][0].shape[0]
    measurements = samples.measurements["result"].copy()
    shuffled_measurements = rng.permutation(measurements)
    shuffled_measurements = shuffled_measurements[:n_shots,:]
    int_outcomes = [big_endian_bits_to_int(bits) for bits in shuffled_measurements]
    counts = collections.Counter(int_outcomes)
    for i in range(n_outcomes):
        if i not in counts:
            counts[i] = 0
    freqs = np.array([v for k, v in sorted(counts.items())])/counts.total()
    return freqs

####################################################################################

def abbrev_n_shots(n_shots):
    """Shorter n_shots component of a filename"""
    if n_shots % 1000 == 0:
        return f'{n_shots // 1000}k'
    return str(n_shots)

class TaskProgram:
    description: str

    @classmethod
    def create_circuits(cls, task, *args, **kwargs):
        """Takes a task, and returns circuits and context dictionary.""" 
        pass

    @classmethod
    def process_results(cls, task, *args, **kwargs):
        """Takes a task and a reslt, and returns processed data dictionary."""
        pass

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