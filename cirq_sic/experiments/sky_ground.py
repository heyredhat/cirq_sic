"""Compatible sky/ground task API with local Cirq JSON persistence.

New applications should use :mod:`cirq_sic.workflow`. These task classes preserve
historical filenames and the ``recirq.sky_ground`` JSON namespace without
requiring ReCirq. They describe power-of-two logical dimensions; arbitrary
embedded dimensions are supported by the unified workflow.
"""

from dataclasses import dataclass, fields
import inspect
from itertools import combinations_with_replacement, permutations
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

import cirq
import numpy as np

from ..circuits import (
    ansatz_circuit,
    arthurs_kelly,
    d4_sic_fiducial,
    measure_register,
    msimple_wh_povm,
    qudit_basis_state,
    real_triple_product_from_probabilities,
    simple_wh_povm,
    triple_product_measurement_circuit,
    wh_state,
)
from ..wh import change_conjugate_convention
from .experiment_utils import (
    abbrev_grid_qubits,
    abbrev_n_shots,
    bqskit_machine_model,
    bqskit_optimize_circuit,
    cirq_optimize_circuit,
    exact_simulation,
    get_sampler,
    has_dynamic_circuit_features,
    results_to_freqs,
    setup_logger,
)

EXPERIMENT_NAME = "sky_ground"
DEFAULT_BASE_DIR = f"data/{EXPERIMENT_NAME}"
SIMPLE_WH_IMPLEMENTATIONS = {"simple", "msimple"}
_TASK_REGISTRY = {}


def _task_dataclass(cls):
    cls = dataclass(cls, frozen=True)
    _TASK_REGISTRY[f"recirq.sky_ground.{cls.__name__}"] = cls
    return cls


def _read_task_json(path):
    return cirq.read_json(path, resolvers=[_TASK_REGISTRY.get, *cirq.DEFAULT_RESOLVERS])


def _result_path(task, base_dir):
    return Path(DEFAULT_BASE_DIR if base_dir is None else base_dir) / f"{task.fn}.json"


def _probabilities(results, probs, expected_shape=None):
    if (results is None) == (probs is None):
        raise ValueError("Supply exactly one of results or probs.")
    probabilities = (
        results_to_freqs(results) if probs is None else np.asarray(probs, dtype=float)
    )
    if expected_shape is not None and probabilities.shape != expected_shape:
        raise ValueError(
            f"Expected probabilities of shape {expected_shape}, got {probabilities.shape}."
        )
    if probabilities.ndim != 2 or not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities must be a finite (circuits, outcomes) matrix.")
    if np.any(probabilities < -1e-10) or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError(
            "Each circuit must have a normalized probability distribution."
        )
    return probabilities


class _TaskMixin:
    """Shared validation and JSON support for the legacy dataclasses."""

    @classmethod
    def filename(cls, **specs):
        """Return the historical filename stem for this task specification."""
        segments = [specs["dataset_id"], f"d{specs['d']}", cls.__name__]
        segments.extend(specs[name] for name in cls._filename_fields)
        segments.append(
            f"{specs['optimizer']}_{specs['run_type']}_n{abbrev_n_shots(specs['n_shots'])}_"
            f"{specs['processor_id']}_q{abbrev_grid_qubits(specs['qubits'])}"
        )
        return "/".join(segments)

    @property
    def fn(self):
        """The task filename stem relative to its base directory."""
        return self.filename(**self.__dict__)

    def __post_init__(self):
        if (
            not isinstance(self.d, (int, np.integer))
            or self.d < 2
            or self.d & (self.d - 1)
        ):
            raise ValueError(
                "Legacy tasks require a power-of-two dimension d >= 2. "
                "Use cirq_sic.workflow for arbitrary embedded dimensions."
            )
        if not isinstance(self.n_shots, (int, np.integer)) or self.n_shots < 1:
            raise ValueError("n_shots must be a positive integer.")
        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError("qubits must be distinct.")
        n = self.d.bit_length() - 1 if isinstance(self.d, int) else int(np.log2(self.d))
        if isinstance(self, CharacterizeWHTripleProductsTask):
            needed = 1 + 3 * n
        elif isinstance(self, BasisMeasurementAfterWHPOVMOnBasisStatesTask):
            needed = 3 * n
        elif hasattr(self, "wh_implementation"):
            if self.wh_implementation not in {"ak", "simple", "msimple"}:
                raise ValueError(
                    "wh_implementation must be 'ak', 'simple', or 'msimple'."
                )
            needed = (3 if self.wh_implementation == "ak" else 2) * n
        else:
            needed = n
        if len(self.qubits) < needed:
            raise ValueError(f"This task needs at least {needed} qubits.")

    @classmethod
    def _json_namespace_(cls):
        return "recirq.sky_ground"

    def _json_dict_(self):
        values = {field.name: getattr(self, field.name) for field in fields(self)}
        for name in ("fiducial_circuit", "states_circuits"):
            value = values.get(name)
            if callable(value):
                if value is not d4_sic_fiducial:
                    raise TypeError(
                        "Custom callable preparations cannot be saved; supply a Cirq Circuit or state vector."
                    )
                values[name] = {"preparation_name": "d4_sic_fiducial"}
            if isinstance(value, list) and any(callable(item) for item in value):
                raise TypeError(
                    "Callable state preparations cannot be saved; supply Cirq Circuits or state vectors."
                )
        return values

    @classmethod
    def _from_json_dict_(cls, **kwargs):
        kwargs.pop("cirq_type", None)
        if kwargs.get("fiducial_circuit") == {"preparation_name": "d4_sic_fiducial"}:
            kwargs["fiducial_circuit"] = d4_sic_fiducial
        return cls(**kwargs)


def get_wh_qubits(d, wh_implementation):
    """Return grid registers for a legacy power-of-two WH implementation.

    Use the unified workflow's automatic qubit allocation for embedded dimensions.
    """
    if not isinstance(d, (int, np.integer)) or d < 2 or d & (d - 1):
        raise ValueError(
            "Legacy tasks require a power-of-two dimension; use cirq_sic.workflow for arbitrary d."
        )
    if wh_implementation not in {"ak", "simple", "msimple"}:
        raise ValueError("wh_implementation must be 'ak', 'simple', or 'msimple'.")
    n = int(np.log2(d))
    registers = 2 if wh_implementation in SIMPLE_WH_IMPLEMENTATIONS else 3
    return cirq.GridQubit.rect(registers, n, top=4, left=2)


def has_post_measurement_task(wh_implementation):
    """Return whether the implementation preserves the post-POVM system state."""
    return wh_implementation == "ak"


def _as_preparation(value):
    if callable(value):
        return value
    if not isinstance(value, cirq.AbstractCircuit):
        raise TypeError("A preparation must be a callable or Cirq Circuit.")
    source_qubits = sorted(value.all_qubits())

    def prepare(qubits, conjugate=False):
        if len(qubits) != len(source_qubits):
            raise ValueError("Preparation circuit has the wrong number of qubits.")
        mapped = cirq.Circuit(value).transform_qubits(dict(zip(source_qubits, qubits)))
        if conjugate:
            return cirq.Circuit(
                cirq.MatrixGate(cirq.unitary(mapped).conj()).on(*qubits)
            )
        return mapped

    return prepare


def _fiducial_preparation(task):
    if task.fiducial_circuit is not None:
        return _as_preparation(task.fiducial_circuit)
    if task.fiducial is None or np.asarray(task.fiducial).shape != (task.d,):
        raise ValueError(
            "Supply a fiducial vector of length d or a preparation circuit."
        )
    return ansatz_circuit(task.fiducial)


def _state_preparations(task):
    if task.states_circuits is not None:
        return [_as_preparation(preparation) for preparation in task.states_circuits]
    if task.states is None:
        raise ValueError("Supply states or states_circuits.")
    if any(np.asarray(state).shape != (task.d,) for state in task.states):
        raise ValueError("Each state must be a ket of length d.")
    return [ansatz_circuit(state) for state in task.states]


def _basis_preparations(task):
    return [(lambda qubits, i=i: qudit_basis_state(qubits, i)) for i in range(task.d)]


def _orbit_preparations(task):
    prepare_fiducial = _fiducial_preparation(task)
    return [
        (lambda qubits, a=a, b=b: wh_state(qubits, prepare_fiducial, a, b))
        for a in range(task.d)
        for b in range(task.d)
    ]


def _wh_readout_circuits(task, preparations, *, exact=False):
    n = int(np.log2(task.d))
    system = task.qubits[:n]
    fiducial = _fiducial_preparation(task)
    if task.wh_implementation in SIMPLE_WH_IMPLEMENTATIONS:
        registers = {"ancilla_qubits": task.qubits[n : 2 * n]}
    else:
        registers = {
            "ancilla1_qubits": task.qubits[n : 2 * n],
            "ancilla2_qubits": task.qubits[2 * n : 3 * n],
        }
    return [
        cirq.Circuit(
            prepare(system),
            wh_povm_circuit(
                task.wh_implementation, system, fiducial, exact=exact, **registers
            ),
        )
        for prepare in preparations
    ]


def _basis_readout_circuits(task, preparations):
    system = task.qubits[: int(np.log2(task.d))]
    return [
        cirq.Circuit(prepare(system), measure_register(system, "s"))
        for prepare in preparations
    ]


def wh_povm_circuit(
    wh_implementation,
    state_qubits,
    prepare_fiducial,
    *,
    ancilla_qubits=None,
    ancilla1_qubits=None,
    ancilla2_qubits=None,
    exact=False,
    measure=True,
):
    """Build the selected legacy WH interaction, optionally its coherent equivalent."""
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
    """Compile task circuits, retaining dynamic circuits unchanged.

    ``none`` preserves the circuits; ``cirq_generic`` compiles with cirq-core;
    ``cirq`` routes to the named Google processor; ``bqskit_local_N`` and
    ``bqskit_server_N`` use BQSKit optimization level N.
    """
    circuits = list(circuits)
    if task.optimizer == "none":
        return circuits
    if any(has_dynamic_circuit_features(circuit) for circuit in circuits):
        if logger is not None:
            logger.info(
                "Keeping dynamic circuits unchanged to preserve feed-forward order."
            )
        return circuits
    if task.optimizer in {"cirq", "cirq_generic"}:
        processor = None if task.optimizer == "cirq_generic" else task.processor_id
        return [
            cirq_optimize_circuit(task.qubits, circuit, processor_id=processor)
            for circuit in circuits
        ]
    if task.optimizer.startswith("bqskit_"):
        tokens = task.optimizer.split("_")
        if (
            len(tokens) != 3
            or tokens[1] not in {"local", "server", "localhost"}
            or tokens[2] not in {"1", "2", "3", "4"}
        ):
            raise ValueError(
                "Use bqskit_local_N or bqskit_server_N with N from 1 to 4."
            )
        model = bqskit_machine_model(task.qubits, processor_id=task.processor_id)
        return [
            bqskit_optimize_circuit(
                task.qubits,
                circuit,
                model,
                optimization_level=int(tokens[2]),
                server=tokens[1],
            )
            for circuit in circuits
        ]
    raise ValueError(f"Unsupported optimizer: {task.optimizer}")


def load_results(task, base_dir=None):
    """Read a task, its processed arrays, and saved raw results from JSON."""
    return _read_task_json(_result_path(task, base_dir))


def load_circuits(task, base_dir=None, raw=False):
    """Read previously built raw or optimized circuits for a task."""
    name = "raw.json" if raw else f"{task.optimizer}_optimized.json"
    return cirq.read_json(_result_path(task, base_dir).parent / "circuits" / name)


def load_sky_ground_results(specs, base_dir=None, separate=True):
    """Load existing task records; missing files are skipped, corrupt files raise.

    With ``separate=True``, return the task-type mapping and merged processed
    data dictionary. Otherwise return records indexed by their task types.
    """
    results = {}
    for task_type in sky_ground_tasks:
        if (
            not has_post_measurement_task(specs["wh_implementation"])
            and task_type is BasisMeasurementAfterWHPOVMOnBasisStatesTask
        ):
            continue
        path = (
            Path(DEFAULT_BASE_DIR if base_dir is None else base_dir)
            / f"{task_type.filename(**specs)}.json"
        )
        if path.exists():
            results[task_type] = _read_task_json(path)
    if separate:
        tasks = {task_type: record["task"] for task_type, record in results.items()}
        data = {
            name: np.asarray(value)
            for record in results.values()
            for name, value in record["processed_data"].items()
        }
        return tasks, data
    return results


def exactify(task, base_dir=None):
    """Build ideal coherent circuits and return the task's exact processed arrays.

    ``base_dir`` is retained for source compatibility and is not used.
    """
    circuits = (
        task.make_exact_circuits()
        if hasattr(task, "make_exact_circuits")
        else task.make_circuits()
    )
    return task.process_results(
        probs=np.array([exact_simulation(circuit) for circuit in circuits])
    )


def task_from_specs(task_type, specs):
    """Construct a task from the subset of specifications accepted by its class."""
    parameters = inspect.signature(task_type).parameters
    return task_type(**{name: specs[name] for name in parameters if name in specs})


def run_sky_ground_task_from_specs(task_type, specs, base_dir=None):
    """Construct and execute one task, returning its processed data."""
    return run_sky_ground_task(task_from_specs(task_type, specs), base_dir=base_dir)


def run_sky_ground_tasks(specs, sampler=None, base_dir=None):
    """Run the complete applicable task battery and return processed records."""
    results = {}
    for task_type in sky_ground_tasks:
        if (
            not has_post_measurement_task(specs["wh_implementation"])
            and task_type is BasisMeasurementAfterWHPOVMOnBasisStatesTask
        ):
            continue
        results[task_type] = run_sky_ground_task(
            task_from_specs(task_type, specs), sampler=sampler, base_dir=base_dir
        )
    return results


def run_sky_ground_task(task, sampler=None, base_dir=None):
    """Build, compile, sample, analyze, and save one task locally.

    Identical saved tasks are reused. A filename collision with a different
    specification raises an error, preventing stale random states from being
    silently reused. Exceptions propagate to the caller.
    """
    path = _result_path(task, base_dir)
    if path.exists():
        record = _read_task_json(path)
        if cirq.to_json(record["task"]) != cirq.to_json(task):
            raise ValueError(
                "A different task already uses this filename; change dataset_id or the state/fiducial description."
            )
        return {
            name: np.asarray(value) for name, value in record["processed_data"].items()
        }
    circuits = create_circuits(task, base_dir=base_dir)
    if sampler is None:
        sampler = get_sampler(
            task.processor_id, run_type=task.run_type, circuits=circuits
        )
    results = sampler.run_batch(programs=circuits, repetitions=task.n_shots)
    return process_task_results(task, results, base_dir=base_dir)


def create_circuits(task, base_dir=None):
    """Build and save fresh raw and optimized circuits; return optimized circuits.

    Rebuilding rather than trusting a description-only cache makes iteration on
    supplied state vectors reliable.
    """
    directory = _result_path(task, base_dir).parent / "circuits"
    directory.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(task.fn, directory.parent / "experiment.log")
    try:
        circuits = task.make_circuits()
        cirq.to_json(circuits, directory / "raw.json")
        optimized = optimize_task_circuits(task, circuits, logger=logger)
        cirq.to_json(optimized, directory / f"{task.optimizer}_optimized.json")
        return optimized
    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()


def run_circuits(task, base_dir=None, sampler=None):
    """Sample saved optimized circuits and return Cirq's nested batch results."""
    circuits = load_circuits(task, base_dir=base_dir)
    if sampler is None:
        sampler = get_sampler(
            task.processor_id, run_type=task.run_type, circuits=circuits
        )
    return sampler.run_batch(programs=circuits, repetitions=task.n_shots)


def process_task_results(task, results, base_dir=None):
    """Analyze results and atomically save the task, raw results, and processed data."""
    data = task.process_results(results=results)
    path = _result_path(task, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Serialize first: a failing custom callable cannot truncate an existing file.
    payload = cirq.to_json(
        {"task": task, "processed_data": data, "raw_results": results}
    )
    with NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        encoding="utf-8",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return data


@_task_dataclass
class CharacterizeWHReferenceDeviceTask(_TaskMixin):
    """Measure the WH reference matrix ``P[i,j] = Tr(E_i sigma_j)``.

    ``E_i`` is a POVM effect and ``sigma_j = d E_j`` its normalized orbit
    projector. There are ``d**2`` circuits, one per orbit state in displacement
    order. ``fiducial`` is a length-d ket; alternatively ``fiducial_circuit`` is
    a callable preparation or a unitary Cirq circuit.
    """

    _filename_fields = ("wh_implementation", "fiducial_description")

    dataset_id: str
    processor_id: str
    run_type: str
    qubits: list
    n_shots: int
    optimizer: str

    d: int
    fiducial_description: str
    wh_implementation: str

    fiducial: Optional[np.ndarray] = None
    fiducial_circuit: Optional[cirq.Circuit] = None

    def make_circuits(self):
        """Build WH readout circuits in the documented preparation order."""
        return _wh_readout_circuits(self, _orbit_preparations(self), exact=False)

    def make_exact_circuits(self):
        """Build coherent equivalents for terminal exact simulation."""
        return _wh_readout_circuits(self, _orbit_preparations(self), exact=True)

    def process_results(self, results=None, probs=None):
        """Decode outcomes into the task matrix (outcomes by prepared states)."""
        P = _probabilities(results, probs, (self.d**2, self.d**2))
        if self.wh_implementation == "ak":
            P = change_conjugate_convention(P)
        P = P.T
        return {"P": P}


@_task_dataclass
class WHPOVMOnBasisStatesTask(_TaskMixin):
    """Measure ``r[i,k] = Tr(E_i |k><k|)`` for computational basis inputs.

    There are ``d`` circuits. Rows of the resulting ``r`` matrix index the
    ``d**2`` WH outcomes; columns index the prepared basis states.
    """

    _filename_fields = ("wh_implementation", "fiducial_description")

    dataset_id: str
    processor_id: str
    run_type: str
    qubits: list
    n_shots: int
    optimizer: str

    d: int
    fiducial_description: str
    wh_implementation: str

    fiducial: Optional[np.ndarray] = None
    fiducial_circuit: Optional[cirq.Circuit] = None

    def make_circuits(self):
        """Build WH readout circuits in the documented preparation order."""
        return _wh_readout_circuits(self, _basis_preparations(self), exact=False)

    def make_exact_circuits(self):
        """Build coherent equivalents for terminal exact simulation."""
        return _wh_readout_circuits(self, _basis_preparations(self), exact=True)

    def process_results(self, results=None, probs=None):
        """Decode outcomes into the task matrix (outcomes by prepared states)."""
        r = _probabilities(results, probs, (self.d, self.d**2))
        if self.wh_implementation == "ak":
            r = change_conjugate_convention(r)
        r = r.T
        return {"r": r}


@_task_dataclass
class BasisMeasurementOnWHStatesTask(_TaskMixin):
    """Measure ``C[k,j] = Tr(|k><k| sigma_j)`` on WH orbit states.

    There are ``d**2`` circuits. Rows index basis outcomes and columns index
    orbit-state preparations. This conditional matrix enters ``q = C pinv(P) r``.
    """

    _filename_fields = ("fiducial_description",)

    dataset_id: str
    processor_id: str
    run_type: str
    qubits: list
    n_shots: int
    optimizer: str

    d: int
    fiducial_description: str

    fiducial: Optional[np.ndarray] = None
    fiducial_circuit: Optional[cirq.Circuit] = None

    def make_circuits(self):
        """Read the computational basis in the documented preparation order."""
        return _basis_readout_circuits(self, _orbit_preparations(self))

    def process_results(self, results=None, probs=None):
        """Decode outcomes into the task matrix (outcomes by prepared states)."""
        C = _probabilities(results, probs, (self.d**2, self.d))
        C = C.T
        return {"C": C}


@_task_dataclass
class BasisMeasurementOnBasisStatesTask(_TaskMixin):
    """Measure the basis readout matrix ``q[k,l] = Pr(k | prepare l)``.

    The ideal result is the d-dimensional identity. Its deviation from the
    reconstructed ``C pinv(P) r`` is the empirical sky/ground residual.
    """

    _filename_fields = ()

    dataset_id: str
    processor_id: str
    run_type: str
    qubits: list
    n_shots: int
    optimizer: str

    d: int

    def make_circuits(self):
        """Read the computational basis in the documented preparation order."""
        return _basis_readout_circuits(self, _basis_preparations(self))

    def process_results(self, results=None, probs=None):
        """Decode outcomes into the task matrix (outcomes by prepared states)."""
        q = _probabilities(results, probs, (self.d, self.d))
        q = q.T
        return {"q": q}


@_task_dataclass
class BasisMeasurementAfterWHPOVMOnBasisStatesTask(_TaskMixin):
    """Read the system after an Arthurs--Kelly WH measurement interaction.

    Ancillas are ignored. The ideal post-measurement channel is the WH
    measure-and-prepare channel, so the basis probabilities satisfy ``p = C r``.
    Rows are final basis outcomes and columns are initial basis preparations.
    """

    _filename_fields = ("fiducial_description",)

    dataset_id: str
    processor_id: str
    run_type: str
    qubits: list
    n_shots: int
    optimizer: str

    d: int
    fiducial_description: str

    fiducial: Optional[np.ndarray] = None
    fiducial_circuit: Optional[cirq.Circuit] = None

    def make_circuits(self):
        """Build circuits in the input-state order described by this task."""
        n = int(np.log2(self.d))
        m = np.arange(self.d)
        prepare_fiducial = _fiducial_preparation(self)
        state_qubits = self.qubits[:n]
        ancilla1_qubits = self.qubits[n : 2 * n]
        ancilla2_qubits = self.qubits[2 * n : 3 * n]
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
        """Decode outcomes into the task matrix (outcomes by prepared states)."""
        p = _probabilities(results, probs, (self.d, self.d))
        p = p.T
        return {"p": p}


@_task_dataclass
class WHPOVMOnStatesTask(_TaskMixin):
    """Measure a WH POVM on an ordered collection of pure input states.

    Supply length-d kets in ``states`` or preparations in ``states_circuits``.
    The returned ``r`` matrix has WH outcomes as rows and inputs as columns.
    Use the unified workflow to prepare mixed states.
    """

    _filename_fields = (
        "wh_implementation",
        "fiducial_description",
        "states_description",
    )

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

    fiducial: Optional[np.ndarray] = None
    fiducial_circuit: Optional[cirq.Circuit] = None
    states: Optional[list] = None
    states_circuits: Optional[list] = None

    def make_circuits(self):
        """Build WH readout circuits in the documented preparation order."""
        return _wh_readout_circuits(self, _state_preparations(self), exact=False)

    def make_exact_circuits(self):
        """Build coherent equivalents for terminal exact simulation."""
        return _wh_readout_circuits(self, _state_preparations(self), exact=True)

    def process_results(self, results=None, probs=None):
        """Decode outcomes into the task matrix (outcomes by prepared states)."""
        r = _probabilities(
            results,
            probs,
            (
                len(
                    self.states_circuits
                    if self.states_circuits is not None
                    else self.states
                ),
                self.d**2,
            ),
        )
        if self.wh_implementation == "ak":
            r = change_conjugate_convention(r)
        r = r.T
        return {"r": r}


@_task_dataclass
class CharacterizeWHTripleProductsTask(_TaskMixin):
    """Measure ``T[i,j,k] = Re Tr(sigma_i sigma_j sigma_k)`` by cycle tests.

    This legacy task measures only real triple products, which are fully
    symmetric under index permutations. It samples each unordered triple once
    and fills the symmetric tensor; the workflow also supports imaginary parts.
    """

    _filename_fields = ("fiducial_description",)

    dataset_id: str
    processor_id: str
    run_type: str
    qubits: list
    n_shots: int
    optimizer: str

    d: int
    fiducial_description: str

    fiducial: Optional[np.ndarray] = None
    fiducial_circuit: Optional[cirq.Circuit] = None

    def symmetric_triples(self):
        """Return all unordered triples, with repetition, of WH orbit indices."""
        n_states = self.d**2
        return list(combinations_with_replacement(range(n_states), 3))

    def make_circuits(self):
        """Build circuits in the input-state order described by this task."""
        n = int(np.log2(self.d))
        if 2**n != self.d:
            raise ValueError(
                "CharacterizeWHTripleProductsTask currently requires d = 2**n."
            )
        if len(self.qubits) < 1 + 3 * n:
            raise ValueError(
                f"Need at least {1 + 3*n} qubits for the triple-product cycle test."
            )

        prepare_states = _orbit_preparations(self)

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
        """Decode outcomes into the task matrix (outcomes by prepared states)."""
        probabilities = _probabilities(results, probs)
        if probabilities.shape != (len(self.symmetric_triples()), 2):
            raise ValueError(
                "Expected one binary outcome distribution per symmetric triple."
            )
        values = real_triple_product_from_probabilities(probabilities)

        n_states = self.d**2
        tensor = np.zeros((n_states, n_states, n_states))
        for value, triple in zip(values, self.symmetric_triples()):
            for perm in set(permutations(triple)):
                tensor[perm] = value

        return {"T": tensor}


sky_ground_tasks = [
    CharacterizeWHReferenceDeviceTask,
    WHPOVMOnBasisStatesTask,
    BasisMeasurementOnWHStatesTask,
    BasisMeasurementOnBasisStatesTask,
    BasisMeasurementAfterWHPOVMOnBasisStatesTask,
]


__all__ = [
    "EXPERIMENT_NAME",
    "DEFAULT_BASE_DIR",
    "SIMPLE_WH_IMPLEMENTATIONS",
    "get_wh_qubits",
    "has_post_measurement_task",
    "wh_povm_circuit",
    "optimize_task_circuits",
    "load_results",
    "load_circuits",
    "load_sky_ground_results",
    "exactify",
    "task_from_specs",
    "run_sky_ground_task_from_specs",
    "run_sky_ground_tasks",
    "run_sky_ground_task",
    "create_circuits",
    "run_circuits",
    "process_task_results",
    "CharacterizeWHReferenceDeviceTask",
    "WHPOVMOnBasisStatesTask",
    "BasisMeasurementOnWHStatesTask",
    "BasisMeasurementOnBasisStatesTask",
    "BasisMeasurementAfterWHPOVMOnBasisStatesTask",
    "WHPOVMOnStatesTask",
    "CharacterizeWHTripleProductsTask",
    "sky_ground_tasks",
]
