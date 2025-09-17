import collections
from functools import reduce
import numpy as np

import cirq 
import cirq_google
import qsimcirq

def kron(*A):
    """Tensor lots of things together."""
    return reduce(np.kron, A)

def rand_ket(d):
    """Random d-dimensional normalized complex vector."""
    ket = np.random.randn(d) + 1j*np.random.randn(d)
    return ket/np.linalg.norm(ket)

def get_gate_counts(circuit):
    """Get gate counts for a cirq circuit."""
    all_gate_types = [type(op.gate) for op in circuit.all_operations()]
    type_counts = collections.Counter(all_gate_types)
    print("--- Gate Counts (by type) ---")
    for gate_type, count in type_counts.items():
        print(f"{gate_type.__name__}: {count}")

def process_circuit(circuit, connectivity_graph, gateset):
    """Conform the circuit to device topology and gateset."""
    router = cirq.RouteCQC(connectivity_graph)
    routed_circuit, initial_map, final_map = router.route_circuit(circuit)
    optimized_circuit = cirq.optimize_for_target_gateset(routed_circuit,\
                                context=cirq.TransformerContext(deep=True), gateset=gateset)
    return optimized_circuit

def get_device_data(processor_id, run_type="noisy"):
    device = cirq_google.engine.create_device_from_processor_id(processor_id)
    gateset = device.metadata.compilation_target_gatesets[0]
    connectivity_graph = device.metadata.nx_graph

    if run_type == "noisy":
        noise_props = cirq_google.engine.load_device_noise_properties(processor_id)
        noise_model = cirq_google.NoiseModelFromGoogleNoiseProperties(noise_props)
        sim = qsimcirq.QSimSimulator(noise=noise_model)
    else:
        sim = qsimcirq.QSimSimulator()
        
    cal = cirq_google.engine.load_median_device_calibration(processor_id)
    sim_processor = cirq_google.engine.SimulatedLocalProcessor(
        processor_id=processor_id, sampler=sim, device=device, calibrations={cal.timestamp // 1000: cal})
    sim_engine = cirq_google.engine.SimulatedLocalEngine([sim_processor])
    sampler = sim_engine.get_sampler(processor_id)
    return locals()

def get_freqs(samples, n_outcomes, n_shots):
    counts = samples.histogram(key="result")
    for i in range(n_outcomes):
        if i not in counts:
                counts[i] = 0
    noisy_freqs = np.array([v for k, v in sorted(counts.items())])/n_shots
    return noisy_freqs

