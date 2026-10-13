"""Execution, measurement decoding, and optional hardware adapters.

Importing this module requires only NumPy and cirq-core. Google hardware and
BQSKit are imported only by the adapters that use them.
"""

import collections
import importlib
import logging
from pathlib import Path
import re
import sys

import cirq
import numpy as np

__all__ = [
    "default_project_id",
    "setup_logger",
    "abbrev_n_shots",
    "abbrev_grid_qubits",
    "get_sampler",
    "exact_simulation",
    "bqskit_machine_model",
    "bqskit_optimize_circuit",
    "to_one_and_two_qubit_ops",
    "push_measurements_to_end",
    "terminal_measurement_ops",
    "cirq_optimize_circuit",
    "measurement_key_sort_key",
    "ordered_measurement_keys",
    "are_all_measurements_terminal",
    "result_measurements_matrix",
    "has_dynamic_circuit_features",
    "shots_to_freqs",
    "results_to_freqs",
    "avg_negativity",
    "deep_match",
    "query_records",
    "collect_circuit_files",
    "collect_circuits",
    "get_gate_counts",
    "print_all_gate_counts",
]

default_project_id = "sic-povms-sandbox-444403"
MEASUREMENT_PREFIX_ORDER = {"s": 0, "a": 1, "a1": 1, "a2": 2}


def _optional_import(name, extra):
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise ImportError(
            f"{name} is required for this operation; install cirq-sic[{extra}]."
        ) from error


def setup_logger(name, path):
    """Return a file-and-console logger, reusing handlers for the same path.

    Repeated calls do not leak open file handles or duplicate log messages.
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"cirq_sic.{name}.{path}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        for handler in (logging.FileHandler(path), logging.StreamHandler(sys.stdout)):
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    return logger


def abbrev_n_shots(n_shots: int) -> str:
    """Format a shot count, abbreviating positive multiples of 1000 with ``k``."""
    return f"{n_shots // 1000}k" if n_shots and n_shots % 1000 == 0 else str(n_shots)


def abbrev_grid_qubits(qubits):
    """Format grid qubits as ``row_col`` identifiers for legacy filenames."""
    return "-".join(f"{qubit.row}_{qubit.col}" for qubit in qubits)


def get_sampler(
    processor_id=None, run_type="noisy", project_id=default_project_id, circuits=None
):
    """Return a Cirq sampler for ``clean``, Google ``noisy``, or ``real`` execution.

    Clean simulation uses cirq-core and supports feed-forward. Google modes
    require the hardware extra; real hardware rejects dynamic circuits before
    constructing an engine. This function never submits a job itself.
    """
    if run_type not in {"clean", "noisy", "real"}:
        raise ValueError("run_type must be 'clean', 'noisy', or 'real'.")
    if circuits is None:
        circuit_list = []
    elif isinstance(circuits, cirq.AbstractCircuit):
        circuit_list = [circuits]
    else:
        circuit_list = list(circuits)
    has_dynamic = any(has_dynamic_circuit_features(circuit) for circuit in circuit_list)
    if run_type == "real" and has_dynamic:
        raise ValueError(
            "Dynamic circuits with classical feed-forward are not supported by the current "
            "cirq_google device sampler. Use run_type='clean' or 'noisy'."
        )
    if run_type == "clean":
        return cirq.Simulator(dtype=np.complex128)
    if processor_id is None:
        raise ValueError("processor_id is required for Google noisy or real execution.")
    cirq_google = _optional_import("cirq_google", "hardware")
    if run_type == "real":
        engine = cirq_google.Engine(project_id=project_id)
        return engine.get_sampler(processor_id=processor_id)
    properties = cirq_google.engine.load_device_noise_properties(processor_id)
    noise = cirq_google.NoiseModelFromGoogleNoiseProperties(properties)
    return cirq.Simulator(noise=noise, dtype=np.complex128)


def exact_simulation(circuit):
    """Return exact computational outcome probabilities for terminal measurements.

    Measurement keys follow :func:`measurement_key_sort_key`, and bits inside a
    key follow the measurement's qubit order. Invert masks are honored. With no
    measurements, all qubits are read in Cirq's default sorted order. Unitary
    circuits use a state-vector simulator; noisy channels use a density matrix.
    Mid-circuit measurements and measurement confusion maps raise ``ValueError``.
    """
    if has_dynamic_circuit_features(circuit):
        raise ValueError(
            "exact_simulation only supports circuits with terminal measurements."
        )
    measurements = terminal_measurement_ops(circuit)
    if any(op.gate.confusion_map for op in measurements):
        raise ValueError(
            "exact_simulation does not support measurement confusion maps."
        )
    measured_qubits = [q for op in measurements for q in op.qubits]
    if len(set(measured_qubits)) != len(measured_qubits):
        raise ValueError("Each qubit may be measured only once in exact_simulation.")
    if not measurements:
        measured_qubits = sorted(circuit.all_qubits())
    if any(q.dimension != 2 for q in measured_qubits):
        raise ValueError("exact_simulation expects qubit measurements.")
    stripped = cirq.drop_terminal_measurements(circuit)
    # Dropping a measurement may remove a qubit that was otherwise idle.
    stripped.append(cirq.I(q) for q in measured_qubits)
    if cirq.has_unitary(stripped):
        result = cirq.Simulator(dtype=np.complex128).simulate(stripped)
        probabilities = np.diag(result.density_matrix_of(measured_qubits)).real.copy()
    else:
        qubit_order = sorted(stripped.all_qubits())
        result = cirq.DensityMatrixSimulator(dtype=np.complex128).simulate(
            stripped, qubit_order=qubit_order
        )
        indices = [qubit_order.index(q) for q in measured_qubits]
        tensor = result.final_density_matrix.reshape((2,) * (2 * len(qubit_order)))
        reduced = cirq.partial_trace(tensor, keep_indices=indices)
        probabilities = np.diag(
            reduced.reshape(2 ** len(indices), 2 ** len(indices))
        ).real.copy()
    # drop_terminal_measurements already replaces inverted readouts with X gates.
    return probabilities


def _bqskit_willow_gateset():
    bq = _optional_import("bqskit", "optimize")
    return [
        bq.ir.gates.PhasedXZGate(),
        bq.ir.gates.CZGate(),
        bq.ir.gates.RXGate(),
        bq.ir.gates.RYGate(),
        bq.ir.gates.RZGate(),
    ]


def __getattr__(name):
    # Preserve the old optional constant without importing BQSKit at module load.
    if name == "bqskit_willow_gateset":
        return _bqskit_willow_gateset()
    raise AttributeError(name)


def bqskit_machine_model(qubits, processor_id="willow_pink"):
    """Build a BQSKit model restricted to the requested Google grid qubits."""
    bq = _optional_import("bqskit", "optimize")
    cirq_google = _optional_import("cirq_google", "hardware")
    device = cirq_google.engine.create_device_from_processor_id(processor_id)
    qubit_index = {q: i for i, q in enumerate(qubits)}
    if not set(qubits) <= set(device.metadata.qubit_set):
        raise ValueError("Some requested qubits are not on the device.")
    coupling_graph = [
        (qubit_index[a], qubit_index[b])
        for a, b in device.metadata.nx_graph.edges
        if a in qubit_index and b in qubit_index
    ]
    return bq.MachineModel(
        len(qubits), gate_set=_bqskit_willow_gateset(), coupling_graph=coupling_graph
    )


def bqskit_optimize_circuit(
    qubits, circuit, machine_model, optimization_level=1, server="local"
):
    """Compile a preparation-and-measurement experiment with BQSKit.

    Experiments must start from all-zero qubits. Terminal readouts are restored
    using BQSKit's final logical-to-physical mapping. The returned circuit uses
    ``qubits`` as physical device locations. Initial mapping is immaterial for
    an all-zero input; arbitrary externally supplied input states are unsupported.
    """
    if has_dynamic_circuit_features(circuit):
        raise ValueError(
            "bqskit optimization does not support dynamic circuits with mid-circuit measurement."
        )
    if server not in {"local", "localhost", "server"}:
        raise ValueError("server must be 'local', 'localhost', or 'server'.")
    bq = _optional_import("bqskit", "optimize")
    converters = _optional_import("bqskit.ext", "optimize")
    qubits = list(qubits)
    if not set(circuit.all_qubits()) <= set(qubits):
        raise ValueError("qubits must include every circuit qubit.")
    measurements = terminal_measurement_ops(circuit)
    logical = _remove_readouts(circuit).transform_qubits(
        {q: cirq.LineQubit(i) for i, q in enumerate(qubits)}
    )
    logical.append(cirq.I(q) for q in cirq.LineQubit.range(len(qubits)))
    compiled, _, final_mapping = bq.compile(
        converters.cirq_to_bqskit(logical),
        model=machine_model,
        optimization_level=optimization_level,
        with_mapping=True,
        ip=None if server == "local" else "localhost",
    )
    final_circuit = converters.bqskit_to_cirq(compiled).transform_qubits(
        {cirq.NamedQubit(f"q_{i}"): q for i, q in enumerate(qubits)}
    )
    output_map = {q: qubits[final_mapping[i]] for i, q in enumerate(qubits)}
    final_circuit.append(op.transform_qubits(output_map) for op in measurements)
    return final_circuit


def to_one_and_two_qubit_ops(circuit: cirq.Circuit) -> cirq.Circuit:
    """Decompose unitary gates to at most two qubits, retaining measurements."""

    def keep(op):
        return cirq.is_measurement(op) or len(op.qubits) <= 2

    return cirq.Circuit(
        cirq.decompose(op, keep=keep) for op in circuit.all_operations()
    )


def push_measurements_to_end(circuit: cirq.Circuit) -> cirq.Circuit:
    """Move terminal measurements to the end without changing circuit semantics.

    Dynamic circuits are rejected: moving their readouts can change the state or
    invalidate classical controls.
    """
    if has_dynamic_circuit_features(circuit):
        raise ValueError("Cannot move mid-circuit measurements or classical controls.")
    result = _remove_readouts(circuit)
    result.append(cirq.Moment(terminal_measurement_ops(circuit)))
    return result


def _remove_readouts(circuit):
    # Unlike drop_terminal_measurements, do not apply classical invert masks;
    # the original readouts will be appended again after optimization.
    return cirq.Circuit(
        op for op in circuit.all_operations() if not cirq.is_measurement(op)
    )


def terminal_measurement_ops(circuit):
    """Collect measurements in register order, requiring terminal readouts."""
    if not are_all_measurements_terminal(circuit):
        raise ValueError("Circuit contains nonterminal measurements.")
    return sorted(
        (op for op in circuit.all_operations() if cirq.is_measurement(op)),
        key=lambda op: measurement_key_sort_key(cirq.measurement_key_name(op)),
    )


def cirq_optimize_circuit(qubits, circuit, processor_id="willow_pink"):
    """Compile to CZ gates, optionally routing onto a Google device.

    ``processor_id=None`` uses cirq-core only and imposes no device topology.
    An explicit processor requires cirq-google; dynamic circuits are rejected
    because preserving feed-forward through hardware routing is not supported.
    """
    if has_dynamic_circuit_features(circuit):
        raise ValueError("Hardware routing does not support dynamic circuits.")
    if processor_id is None:
        return cirq.optimize_for_target_gateset(circuit, gateset=cirq.CZTargetGateset())
    cirq_google = _optional_import("cirq_google", "hardware")
    device = cirq_google.engine.create_device_from_processor_id(processor_id)
    measurements = terminal_measurement_ops(circuit)
    logical = _remove_readouts(circuit)
    # Include otherwise idle measured wires so final_map is complete.
    logical.append(cirq.I(q) for op in measurements for q in op.qubits)
    logical = to_one_and_two_qubit_ops(logical)
    router = cirq.RouteCQC(device.metadata.nx_graph)
    routed, _, final_map = router.route_circuit(
        logical, initial_mapper=cirq.HardCodedInitialMapper({q: q for q in qubits})
    )
    result = cirq.optimize_for_target_gateset(
        routed,
        context=cirq.TransformerContext(deep=True),
        gateset=device.metadata.compilation_target_gatesets[0],
    )
    result.append(cirq.Moment(op.transform_qubits(final_map) for op in measurements))
    return result


def measurement_key_sort_key(key):
    """Order ``result``, then ``s``, ``a``/``a1``, ``a2``, then unknown keys."""
    key = str(key)
    if key == "result":
        return (-1, "result", -1)
    match = re.fullmatch(r"([A-Za-z][A-Za-z0-9]*)_(\d+)", key)
    if match is None:
        return (99, key, -1)
    prefix, number = match.groups()
    return (MEASUREMENT_PREFIX_ORDER.get(prefix, 98), prefix, int(number))


def ordered_measurement_keys(measurements):
    """Return keys in the logical register order used by legacy task circuits."""
    return sorted(measurements, key=measurement_key_sort_key)


def are_all_measurements_terminal(circuit):
    """Return whether dropping terminal measurements removes every readout."""
    try:
        stripped = cirq.drop_terminal_measurements(circuit)
    except ValueError:
        return False
    return not any(cirq.is_measurement(op) for op in stripped.all_operations())


def result_measurements_matrix(shots):
    """Stack all measurement keys into a big-endian bit matrix.

    Each key must occur once per repetition. Repeated-key records cannot be
    silently flattened into a different outcome alphabet.
    """
    measurements = shots.measurements
    keys = ordered_measurement_keys(measurements)
    if not keys:
        raise ValueError("No measurements found in shots data.")
    columns = [np.asarray(measurements[key]) for key in keys]
    if any(column.ndim != 2 for column in columns):
        raise ValueError("Each measurement must be a (shots, bits) matrix.")
    if len({len(column) for column in columns}) != 1:
        raise ValueError("Measurement keys have inconsistent shot counts.")
    matrix = np.hstack(columns)
    if not np.all((matrix == 0) | (matrix == 1)):
        raise ValueError("Expected binary qubit outcomes.")
    return matrix


def has_dynamic_circuit_features(circuit):
    """Return whether a circuit uses feed-forward or nonterminal measurement."""
    return any(
        getattr(op, "classical_controls", ()) for op in circuit.all_operations()
    ) or not are_all_measurements_terminal(circuit)


def shots_to_freqs(shots):
    """Convert a Cirq result to normalized frequencies including zero-count bins."""
    measurements = result_measurements_matrix(shots)
    if len(measurements) == 0:
        raise ValueError("Cannot estimate frequencies from zero shots.")
    width = measurements.shape[1]
    integers = measurements.astype(np.int64) @ (1 << np.arange(width - 1, -1, -1))
    return np.bincount(integers, minlength=2**width) / len(measurements)


def results_to_freqs(results):
    """Decode a batch result with one parameterization per circuit.

    Both ``Sampler.run_batch``'s nested lists and a flat sequence of results are
    accepted. Parameter sweeps require an explicit selection by the caller.
    """
    frequencies = []
    for result in results:
        if isinstance(result, (list, tuple)):
            if len(result) != 1:
                raise ValueError("Expected one parameterization per circuit.")
            result = result[0]
        frequencies.append(shots_to_freqs(result))
    return np.asarray(frequencies)


def avg_negativity(matrix):
    """Return the mean total negative mass per column of a real matrix."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or not matrix.shape[1] or not np.all(np.isfinite(matrix)):
        raise ValueError("matrix must be a finite matrix with at least one column.")
    return np.maximum(-matrix, 0).sum() / matrix.shape[1]


def deep_match(obj, criteria):
    """Test whether an object contains a nested match for a criteria dictionary.

    Dictionaries, object attributes, and lists are traversed. Cycles are ignored
    after their first visit; NumPy values are compared by array equality.
    """
    if not isinstance(criteria, dict):
        raise TypeError("criteria must be a dictionary.")
    seen = set()

    def visit(value, requested):
        identity = (id(value), id(requested))
        if identity in seen:
            return False
        seen.add(identity)
        attrs = (
            value
            if isinstance(value, dict)
            else vars(value)
            if hasattr(value, "__dict__")
            else None
        )
        if attrs is not None:
            if all(
                key in attrs
                and (
                    visit(attrs[key], expected)
                    if isinstance(expected, dict)
                    else np.array_equal(attrs[key], expected)
                )
                for key, expected in requested.items()
            ):
                return True
            return any(visit(item, requested) for item in attrs.values())
        if isinstance(value, (list, tuple)):
            return any(visit(item, requested) for item in value)
        return False

    return visit(obj, criteria)


def query_records(records, query):
    """Return records matching the nested criteria dictionary ``query``."""
    return [record for record in records if deep_match(record, query)]


def collect_circuit_files(base_dir):
    """Find JSON files in every directory named ``circuits`` below ``base_dir``."""
    base = Path(base_dir).expanduser().resolve()
    directories = [base] if base.name == "circuits" else []
    directories += sorted(path for path in base.rglob("circuits") if path.is_dir())
    return {
        path: sorted(file.name for file in path.glob("*.json")) for path in directories
    }


def collect_circuits(base_dir):
    """Read saved circuit JSON, keyed by paths relative to the base directory."""
    base = Path(base_dir).expanduser().resolve()
    return {
        str(path.parent.relative_to(base))
        if path != base
        else ".": {name: cirq.read_json(path / name) for name in names}
        for path, names in collect_circuit_files(base).items()
    }


def get_gate_counts(circuit, return_str=False):
    """Count nonmeasurement operations by gate class, optionally formatting text."""
    counts = collections.Counter(
        type(op.gate).__name__
        for op in circuit.all_operations()
        if not cirq.is_measurement(op)
    )
    if return_str:
        return (
            ", ".join(f"{gate}: {count}" for gate, count in counts.items())
            + f". Total: {sum(counts.values())}"
        )
    return counts


def print_all_gate_counts(base_dir):
    """Print counts for each circuit stored beneath a legacy task directory."""
    for task_spec, files in collect_circuits(base_dir).items():
        print(task_spec)
        for circuit_type, circuits in files.items():
            if isinstance(circuits, cirq.AbstractCircuit):
                circuits = [circuits]
            for i, circuit in enumerate(circuits):
                print(
                    f"\t{circuit_type} ({i}): [{get_gate_counts(circuit, return_str=True)}]"
                )
