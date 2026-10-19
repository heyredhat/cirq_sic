"""Build, simulate, sample, and persist reproducible quantum experiments.

All builders return :class:`Experiment`. Exact simulation and hardware sampling
produce the same :class:`ExperimentResult` structure. Counts include every
binary output, including padded outcomes: leakage is never postselected away.
Cirq's big-endian register convention is used throughout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cirq
import numpy as np

from .measurements import Measurement, build_measurement_circuit, sic_povm

_SCHEMA_VERSION = 1


def _json_data(value):
    """Copy metadata into portable JSON types, rejecting NaN and arbitrary objects."""
    return json.loads(json.dumps(value, allow_nan=False))


def _portable_seed(seed):
    """Return a JSON-portable NumPy/Cirq seed or reject nonportable RNG objects."""
    if seed is None:
        return None
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or not 0 <= seed < 2**32
    ):
        raise ValueError("seed must be an integer in [0, 2**32) or None.")
    return int(seed)


def _readout_prefix(spec, noise):
    """Apply noise on the original timeline, then stop at the recorded readout.

    Removing readout first leaves an empty moment in Cirq, which can acquire
    spurious noise. Keeping noise after readout would instead change a state
    after its classical outcome was already recorded. The prefix retains
    actual idle moments and any noise inserted before the measurement.
    """
    circuit = spec.circuit
    expected = next(op for op in circuit.all_operations() if cirq.is_measurement(op))
    if noise is not None:
        model = cirq.NoiseModel.from_noise_model_like(noise)
        circuit = cirq.Circuit(model.noisy_moments(circuit, spec.qubit_order))
    readouts = [op for op in circuit.all_operations() if cirq.is_measurement(op)]
    if readouts != [expected] or any(
        cirq.control_keys(op) for op in circuit.all_operations()
    ):
        raise ValueError(
            "Noise must preserve the sole readout and cannot add measurements or classical controls."
        )
    prefix = []
    for moment in circuit:
        if any(cirq.is_measurement(op) for op in moment):
            # Operations in this moment are disjoint, so these commute with
            # readout and do not alter its probabilities. Keep their timing.
            remaining = [op for op in moment if not cirq.is_measurement(op)]
            if remaining:
                prefix.append(cirq.Moment(remaining))
            break
        prefix.append(moment)
    return cirq.Circuit(prefix)


@dataclass(frozen=True)
class CircuitSpec:
    """One terminal measurement circuit and its unambiguous output layout.

    ``qubit_order`` contains every qubit, including idle and purification qubits.
    ``measured_qubits`` gives the order of bits in ``key``. Outputs numbered
    ``num_outcomes`` and above remain visible as leakage. Metadata must be JSON
    serializable; use numeric arrays separately in :class:`Experiment`.
    """

    name: str
    circuit: cirq.Circuit
    qubit_order: tuple
    measured_qubits: tuple
    key: str
    num_outcomes: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        order, measured = tuple(self.qubit_order), tuple(self.measured_qubits)
        circuit = cirq.Circuit(self.circuit)
        if (
            not isinstance(self.name, str)
            or not isinstance(self.key, str)
            or not self.name
            or not self.key
        ):
            raise ValueError("Circuit name and measurement key must be nonempty.")
        if len(set(order)) != len(order) or len(set(measured)) != len(measured):
            raise ValueError("Qubit layouts must not contain duplicates.")
        if not measured or not set(measured).issubset(order):
            raise ValueError(
                "Measured qubits must be a nonempty subset of qubit_order."
            )
        if not circuit.all_qubits().issubset(order) or any(
            q.dimension != 2 for q in order
        ):
            raise ValueError(
                "qubit_order must contain every circuit qubit (all dimension 2)."
            )
        if (
            isinstance(self.num_outcomes, bool)
            or not isinstance(self.num_outcomes, (int, np.integer))
            or not 1 <= self.num_outcomes <= 2 ** len(measured)
        ):
            raise ValueError("num_outcomes must fit the measurement register.")
        measurements = [
            op for op in circuit.all_operations() if cirq.is_measurement(op)
        ]
        if (
            len(measurements) != 1
            or cirq.measurement_key_name(measurements[0]) != self.key
            or measurements[0].qubits != measured
            or not circuit.are_all_measurements_terminal()
        ):
            raise ValueError(
                "A CircuitSpec requires one terminal measurement matching its key and layout."
            )
        if any(getattr(measurements[0].gate, "invert_mask", ())):
            raise ValueError(
                "Invert masks are unsupported; make output relabeling explicit."
            )
        if getattr(measurements[0].gate, "confusion_map", {}):
            raise ValueError(
                "Measurement confusion maps are unsupported; model readout noise explicitly."
            )
        if any(cirq.control_keys(op) for op in circuit.all_operations()):
            raise ValueError(
                "Classical controls are unsupported in terminal-measurement experiments."
            )
        object.__setattr__(self, "circuit", circuit)
        object.__setattr__(self, "qubit_order", order)
        object.__setattr__(self, "measured_qubits", measured)
        object.__setattr__(self, "metadata", _json_data(dict(self.metadata)))

    @property
    def output_size(self):
        """Number of bit strings, including unused padded outputs."""
        return 2 ** len(self.measured_qubits)


@dataclass(frozen=True)
class OutcomeData:
    """Full output probabilities and optional counts from one circuit.

    ``probabilities`` sums to one across *all* binary outcomes; ``valid`` is the
    unnormalized physical portion. ``leakage`` is the unused-output mass. Exact
    results have ``counts=None`` and ``repetitions=None``.
    """

    probabilities: np.ndarray
    num_outcomes: int
    counts: Optional[np.ndarray] = None
    repetitions: Optional[int] = None

    def __post_init__(self):
        raw_probabilities = np.asarray(self.probabilities)
        if np.iscomplexobj(raw_probabilities) and np.any(raw_probabilities.imag != 0):
            raise ValueError("Probabilities must be real.")
        p = np.array(raw_probabilities.real, dtype=float, copy=True)
        if (
            p.ndim != 1
            or not np.all(np.isfinite(p))
            or np.any(p < -1e-10)
            or not np.isclose(p.sum(), 1)
        ):
            raise ValueError(
                "Probabilities must be a finite, normalized nonnegative vector."
            )
        if (
            isinstance(self.num_outcomes, (bool, np.bool_))
            or not isinstance(self.num_outcomes, (int, np.integer))
            or not 1 <= self.num_outcomes <= len(p)
        ):
            raise ValueError("Invalid physical outcome count.")
        object.__setattr__(self, "num_outcomes", int(self.num_outcomes))
        p = np.maximum(p, 0)
        # Avoid changing already-normalized data by an ulp on every JSON
        # round trip. Correct meaningful normalization errors only.
        if abs(p.sum() - 1.0) > 8 * np.finfo(float).eps:
            p /= p.sum()
        p.setflags(write=False)
        object.__setattr__(self, "probabilities", p)
        if self.counts is not None:
            raw = np.asarray(self.counts)
            if (
                raw.shape != p.shape
                or np.iscomplexobj(raw)
                or not np.all(np.isfinite(raw))
                or np.any(raw < 0)
                or not np.all(raw == np.floor(raw))
            ):
                raise ValueError(
                    "Counts must be nonnegative integers matching probabilities."
                )
            integer_counts = [int(count) for count in raw]
            if any(count > np.iinfo(np.int64).max for count in integer_counts):
                raise ValueError("Counts must fit signed 64-bit integers.")
            counts = np.asarray(integer_counts, dtype=np.int64)
            if (
                isinstance(self.repetitions, (bool, np.bool_))
                or not isinstance(self.repetitions, (int, np.integer))
                or not 0 < self.repetitions <= np.iinfo(np.int64).max
                or sum(integer_counts) != self.repetitions
            ):
                raise ValueError("Counts must sum to positive repetitions.")
            if not np.allclose(counts / self.repetitions, p):
                raise ValueError("Counts and probabilities disagree.")
            counts.setflags(write=False)
            object.__setattr__(self, "counts", counts)
            object.__setattr__(self, "repetitions", int(self.repetitions))
        elif self.repetitions is not None:
            raise ValueError("Repetitions require counts.")

    @property
    def valid(self):
        """Physical output probabilities; their sum is ``1 - leakage``."""
        return self.probabilities[: self.num_outcomes]

    @property
    def leakage(self):
        """Probability assigned to padded outcomes."""
        return float(self.probabilities[self.num_outcomes :].sum())

    @property
    def standard_errors(self):
        """Marginal binomial standard errors, or None for exact probabilities.

        These are plug-in errors, not confidence intervals; at zero counts they
        are zero even though uncertainty remains. Outcomes are correlated.
        """
        if self.repetitions is None:
            return None
        return np.sqrt(self.probabilities * (1 - self.probabilities) / self.repetitions)


@dataclass
class ExperimentResult:
    """Named outcomes plus execution provenance; serializable with an experiment."""

    outcomes: dict[str, OutcomeData]
    mode: str
    metadata: dict = field(default_factory=dict)

    def __getitem__(self, name):
        """Return one named circuit's OutcomeData."""
        return self.outcomes[name]

    def save(self, path):
        """Write a portable JSON result; no pickle or executable task objects."""
        document = {
            "schema_version": _SCHEMA_VERSION,
            "kind": "cirq_sic.result",
            "mode": self.mode,
            "metadata": _json_data(self.metadata),
            "outcomes": {
                name: {
                    "probabilities": data.probabilities.tolist(),
                    "num_outcomes": data.num_outcomes,
                    "counts": None if data.counts is None else data.counts.tolist(),
                    "repetitions": data.repetitions,
                }
                for name, data in self.outcomes.items()
            },
        }
        _write_json(path, document)

    @classmethod
    def load(cls, path):
        """Load a versioned JSON result and revalidate its distributions."""
        document = _read_json(path, "cirq_sic.result")
        return cls(
            {name: OutcomeData(**data) for name, data in document["outcomes"].items()},
            document["mode"],
            document["metadata"],
        )


def _write_json(path, document):
    """Atomically replace one explicitly named JSON artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A unique sibling prevents concurrent writers from sharing a partial file.
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.write("\n")
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def _read_json(path, kind):
    """Read and check the artifact format before decoding Cirq objects."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        document.get("schema_version") != _SCHEMA_VERSION
        or document.get("kind") != kind
    ):
        raise ValueError("Unsupported cirq_sic artifact kind or schema version.")
    return document


@dataclass
class Experiment:
    """Reusable circuits with numeric analysis inputs and JSON metadata.

    Build once, optionally transform circuits for a device, then call
    ``simulate()`` or ``run(sampler, repetitions=...)``. No cloud connection or
    compilation is implicit. ``save()`` stores Cirq JSON, layouts, metadata and
    complex arrays in one versioned file.
    """

    circuits: Sequence[CircuitSpec]
    name: str = "experiment"
    metadata: dict = field(default_factory=dict)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        self.circuits = tuple(self.circuits)
        names = [spec.name for spec in self.circuits]
        if not names or len(set(names)) != len(names):
            raise ValueError("An experiment requires uniquely named circuits.")
        self.metadata = _json_data(self.metadata)
        # Canonical complex arrays preserve the fingerprint through JSON
        # round-trips, including arrays supplied originally as integers.
        self.arrays = {
            key: np.array(value, dtype=complex, copy=True)
            for key, value in self.arrays.items()
        }
        if any(not np.all(np.isfinite(array)) for array in self.arrays.values()):
            raise ValueError("Experiment arrays must be finite.")

    def simulate(self, *, noise=None, seed=None):
        """Calculate exact terminal probabilities, marginalizing unmeasured qubits.

        Uses complex128 state vectors for unitary circuits and density matrices
        for noise/channels. Noise may be any Cirq NoiseModel or channel accepted
        by ``Circuit.with_noise``. A seed is only relevant to future stochastic
        extensions: this method computes distributions, never finite shots.
        """
        seed = _portable_seed(seed)
        fingerprint = self.fingerprint
        outcomes = {}
        for spec in self.circuits:
            body = _readout_prefix(spec, noise)
            if cirq.has_unitary(body):
                state = (
                    cirq.Simulator(dtype=np.complex128, seed=seed)
                    .simulate(body, qubit_order=spec.qubit_order)
                    .final_state_vector
                )
                probabilities = abs(state) ** 2
            else:
                state = (
                    cirq.DensityMatrixSimulator(dtype=np.complex128, seed=seed)
                    .simulate(body, qubit_order=spec.qubit_order)
                    .final_density_matrix
                )
                probabilities = np.diag(state).real
            axes = [spec.qubit_order.index(q) for q in spec.measured_qubits]
            remaining = [i for i in range(len(spec.qubit_order)) if i not in axes]
            marginal = probabilities.reshape([2] * len(spec.qubit_order)).transpose(
                axes + remaining
            )
            marginal = marginal.reshape(spec.output_size, -1).sum(axis=1)
            outcomes[spec.name] = OutcomeData(marginal, spec.num_outcomes)
        return ExperimentResult(
            outcomes,
            "exact",
            {
                "experiment": self.name,
                "noise": str(noise),
                "cirq_version": cirq.__version__,
                "experiment_fingerprint": fingerprint,
            },
        )

    def run(self, sampler=None, *, repetitions=1000, seed=None):
        """Sample circuits through any ``cirq.Sampler`` (local by default).

        Pass a hardware sampler explicitly. ``seed`` seeds only the default
        local simulator; supplying a sampler and a seed is rejected to avoid
        claiming reproducibility for a backend whose random state we do not own.
        """
        if (
            isinstance(repetitions, bool)
            or not isinstance(repetitions, (int, np.integer))
            or repetitions <= 0
        ):
            raise ValueError("repetitions must be a positive integer.")
        if sampler is not None and seed is not None:
            raise ValueError("Configure a supplied sampler's seed directly.")
        seed = _portable_seed(seed)
        fingerprint = self.fingerprint
        if sampler is None:
            sampler = cirq.Simulator(dtype=np.complex128, seed=seed)
        results = sampler.run_batch(
            [spec.circuit for spec in self.circuits], repetitions=int(repetitions)
        )
        if len(results) != len(self.circuits) or any(
            len(batch) != 1 for batch in results
        ):
            raise ValueError("Sampler returned an unexpected batch layout.")
        outcomes = {}
        for spec, batch in zip(self.circuits, results):
            bits = np.asarray(batch[0].measurements[spec.key])
            if bits.shape != (repetitions, len(spec.measured_qubits)) or not np.all(
                (bits == 0) | (bits == 1)
            ):
                raise ValueError("Sampler returned invalid measurement bits.")
            indexes = bits.astype(np.int64) @ (
                2 ** np.arange(bits.shape[1] - 1, -1, -1)
            )
            counts = np.bincount(indexes, minlength=spec.output_size)
            outcomes[spec.name] = OutcomeData(
                counts / repetitions, spec.num_outcomes, counts, int(repetitions)
            )
        return ExperimentResult(
            outcomes,
            "sampled",
            {
                "experiment": self.name,
                "seed": seed,
                "sampler": type(sampler).__name__,
                "cirq_version": cirq.__version__,
                "experiment_fingerprint": fingerprint,
            },
        )

    def transform(self, transformer):
        """Return a copy after ``transformer(circuit)`` compilation or noise insertion.

        The transformer must preserve qubits, the terminal measurement and its
        key/order. Device routing that changes qubits requires new CircuitSpecs
        with updated layouts, so remapping cannot silently corrupt decoding.
        """
        specs = [
            CircuitSpec(
                s.name,
                transformer(cirq.Circuit(s.circuit)),
                s.qubit_order,
                s.measured_qubits,
                s.key,
                s.num_outcomes,
                s.metadata,
            )
            for s in self.circuits
        ]
        return Experiment(specs, self.name, self.metadata, self.arrays)

    def with_noise(self, noise):
        """Return a reproducible experiment with explicit pre-readout noise.

        Insert a Cirq NoiseModel or channel on the original moments and full
        recorded qubit register, preserving idle moments. Operations after
        the sole readout are omitted because they cannot affect its recorded
        probability. The resulting circuits retain terminal measurements and
        work with both ``simulate()`` and ``run()``. Circuit fingerprints
        record the inserted operations. An external sampler can apply its own
        additional noise, so use a noiseless local sampler for model checks.
        """
        specs = []
        for spec in self.circuits:
            body = _readout_prefix(spec, noise)
            readout = next(
                op for op in spec.circuit.all_operations() if cirq.is_measurement(op)
            )
            circuit = cirq.Circuit([*body, cirq.Moment(readout)])
            specs.append(
                CircuitSpec(
                    spec.name,
                    circuit,
                    spec.qubit_order,
                    spec.measured_qubits,
                    spec.key,
                    spec.num_outcomes,
                    spec.metadata,
                )
            )
        return Experiment(specs, self.name, self.metadata, self.arrays)

    def _document(self):
        """Return the canonical experiment content, excluding its checksum."""
        return {
            "kind": "cirq_sic.experiment",
            "schema_version": _SCHEMA_VERSION,
            "name": self.name,
            "metadata": self.metadata,
            "arrays": {
                key: {"real": value.real.tolist(), "imag": value.imag.tolist()}
                for key, value in self.arrays.items()
            },
            "circuits": [
                {
                    "name": s.name,
                    "circuit": json.loads(cirq.to_json(s.circuit)),
                    "qubit_order": json.loads(cirq.to_json(list(s.qubit_order))),
                    "measured_qubits": json.loads(
                        cirq.to_json(list(s.measured_qubits))
                    ),
                    "key": s.key,
                    "num_outcomes": s.num_outcomes,
                    "metadata": s.metadata,
                }
                for s in self.circuits
            ],
        }

    @property
    def fingerprint(self):
        """SHA256 of circuit, layout, metadata, and numeric analysis content.

        The fingerprint binds results to the exact recorded experiment and
        survives save/load. Recompute it from current contents so later edits
        to a circuit or analysis array cannot silently reuse stale results.
        It detects accidental mismatches; it is not a cryptographic signature.
        """
        payload = json.dumps(
            self._document(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def save(self, path):
        """Save versioned Cirq JSON, numeric data, and a content fingerprint."""
        document = self._document()
        document["fingerprint"] = self.fingerprint
        _write_json(path, document)

    @classmethod
    def load(cls, path):
        """Load standard Cirq JSON and numeric data; unknown versions fail clearly."""
        document = _read_json(path, "cirq_sic.experiment")
        specs = []
        for entry in document["circuits"]:
            for key in ("circuit", "qubit_order", "measured_qubits"):
                entry[key] = cirq.read_json(json_text=json.dumps(entry[key]))
            specs.append(CircuitSpec(**entry))
        arrays = {
            key: np.asarray(value["real"]) + 1j * np.asarray(value["imag"])
            for key, value in document["arrays"].items()
        }
        experiment = cls(specs, document["name"], document["metadata"], arrays)
        if (
            "fingerprint" in document
            and document["fingerprint"] != experiment.fingerprint
        ):
            raise ValueError(
                "Experiment fingerprint does not match its saved contents."
            )
        return experiment

    def analyze(
        self, result, *, physical=False, leakage_tolerance=1e-8, allow_unbound=False
    ):
        """Analyze results using the builder's recorded mathematical inputs.

        Nonzero padded-output mass beyond ``leakage_tolerance`` is rejected;
        inspect ``OutcomeData.leakage`` to diagnose it. No conditional
        renormalization or clipping of experimental Born-rule residuals occurs.
        ``physical=True`` projects tomographic estimates onto density matrices
        and therefore changes the estimator (it is not maximum likelihood).
        Generated results must match this experiment's content fingerprint.
        To use manually imported data with no fingerprint, explicitly pass
        ``allow_unbound=True`` after checking its circuit identity yourself;
        a present but mismatched fingerprint is always rejected.
        """
        from .analysis import (
            linear_inversion,
            entropy_from_probabilities,
            born_rule_diagnostics,
        )

        if not np.isfinite(leakage_tolerance) or leakage_tolerance < 0:
            raise ValueError("leakage_tolerance must be finite and nonnegative.")
        expected = {s.name for s in self.circuits}
        if set(result.outcomes) != expected:
            raise ValueError("Results do not match this experiment's circuit names.")
        for spec in self.circuits:
            data = result[spec.name]
            if (
                len(data.probabilities) != spec.output_size
                or data.num_outcomes != spec.num_outcomes
            ):
                raise ValueError(
                    "Results do not match this experiment's output layouts."
                )
            if data.leakage > leakage_tolerance:
                raise ValueError(
                    f"Circuit {spec.name!r} has padded-output leakage {data.leakage:.6g}."
                )
        fingerprint = result.metadata.get("experiment_fingerprint")
        if fingerprint is None and not allow_unbound:
            raise ValueError(
                "Results have no experiment fingerprint; use allow_unbound=True for verified imported data."
            )
        if fingerprint is not None and fingerprint != self.fingerprint:
            raise ValueError("Results belong to a different experiment fingerprint.")
        kind = self.metadata.get("kind")
        if kind in ("tomography", "entropy"):
            p = result["measurement"].valid
            rho = linear_inversion(p, self.arrays["effects"], physical=physical)
            report = {
                "density_matrix": rho,
                "probabilities": p,
                "minimum_eigenvalue": float(np.linalg.eigvalsh(rho).min()),
                "purity": float(np.trace(rho @ rho).real),
                "physical_projection": physical,
            }
            if kind == "entropy":
                report["entropy"] = entropy_from_probabilities(
                    p,
                    self.arrays["effects"],
                    physical=physical,
                    basis=self.metadata["basis"],
                    alpha=self.metadata["alpha"],
                    mixed=self.metadata["mixed"],
                    base=self.metadata["base"],
                )
            return report
        if kind == "wh_entropy":
            from .analysis import wh_entropy_from_overlaps

            p = result["measurement"].valid
            return {
                "entropy": wh_entropy_from_overlaps(p, base=self.metadata["base"]),
                "purity": self.metadata["dimension"] * p[0],
                "probabilities": p,
            }
        if kind == "sky_ground":
            sky = np.stack(
                [result[f"sky/{i}"].valid for i in range(self.metadata["n_states"])]
            )
            ground = np.stack(
                [result[f"ground/{i}"].valid for i in range(self.metadata["n_states"])]
            )
            conditional = np.column_stack(
                [
                    result[f"conditional/{i}"].valid
                    for i in range(self.metadata["n_reference"])
                ]
            )
            reference = np.column_stack(
                [
                    result[f"reference/{i}"].valid
                    for i in range(self.metadata["n_reference"])
                ]
            )
            report = born_rule_diagnostics(
                sky, conditional, ground, reference=reference
            )
            report.update(
                sky=sky, ground=ground, conditional=conditional, reference=reference
            )
            return report
        if kind == "triples":
            from .triples import assemble_triple_products

            estimates = []
            for index in range(len(self.metadata["triplets"])):
                real = 2 * result[f"triple/{index}/real"].valid[0] - 1
                imag = 2 * result[f"triple/{index}/imag"].valid[0] - 1
                estimates.append(real + 1j * imag)
            estimates = np.asarray(estimates)
            report = {"triplets": self.metadata["triplets"], "estimates": estimates}
            if self.metadata["complete"]:
                report["tensor"] = assemble_triple_products(
                    estimates, self.metadata["triplets"], self.metadata["n_states"]
                )
            return report
        raise ValueError(
            "No built-in analysis for this experiment; use result.outcomes directly."
        )


def _measurement_spec(name, measurement, state, metadata=None):
    """Adapt a measurement circuit to the shared execution model."""
    built = build_measurement_circuit(measurement, state)
    return CircuitSpec(
        name,
        built.circuit,
        built.qubit_order,
        built.measured_qubits,
        built.measurement_key,
        measurement.num_outcomes,
        metadata or {},
    )


def tomography_experiment(state, measurement=None, *, name="tomography"):
    """Build one IC measurement of a ket or density matrix (SIC by default).

    The supplied state defines simulator/hardware preparation, not the answer
    used by analysis. Analysis reconstructs only from observed probabilities.
    """
    if measurement is None:
        measurement = sic_povm(np.asarray(state).shape[0])
    if (
        np.linalg.matrix_rank(measurement.effects.reshape(measurement.num_outcomes, -1))
        < measurement.dimension**2
    ):
        raise ValueError("Tomography requires an informationally complete POVM.")
    return Experiment(
        [_measurement_spec("measurement", measurement, state)],
        name,
        {"kind": "tomography", "dimension": measurement.dimension},
        {"effects": measurement.effects},
    )


def entropy_experiment(
    state,
    measurement=None,
    *,
    basis="pauli",
    alpha=2,
    mixed=False,
    base=2,
    name="entropy",
):
    """Measure a WH/SIC POVM and infer a stabilizer Rényi entropy by tomography.

    ``basis='pauli'`` means tensor-product qubit Paulis; ``'wh'`` means cyclic
    d-dimensional Weyl operators. The latter is different for d >= 4. Set
    ``mixed=True`` to use the documented purity-corrected extension. Finite-shot
    entropy is a nonlinear plug-in estimate; it is generally biased.
    """
    experiment = tomography_experiment(state, measurement, name=name)
    from .analysis import stabilizer_renyi_entropy

    # Validate the requested convention and input before running hardware.
    stabilizer_renyi_entropy(state, basis=basis, alpha=alpha, mixed=mixed, base=base)
    experiment.metadata.update(
        kind="entropy",
        basis=basis,
        alpha=float(alpha),
        mixed=bool(mixed),
        base=float(base),
    )
    return experiment


def sky_ground_experiment(states, reference, ground=None, *, name="sky_ground"):
    """Build a complete reference-device/sky/ground consistency battery.

    For each input: measure the reference (sky) and the ground POVM directly.
    For each normalized reference ket: measure both devices to estimate
    ``C[j,i]`` and ``P[k,i]``. Analysis compares ``q`` to ``C @ inv(P) @ p``.
    This holds for any minimal IC rank-one reference, including SICs where
    ``inv(P) = (d+1)I - J/d``. Independent preparations allow hardware errors
    in different circuits to violate the consistency relation.
    """
    states = tuple(states)
    if not states:
        raise ValueError("Supply at least one input state.")
    d = reference.dimension
    if (
        reference.num_outcomes != d * d
        or np.linalg.matrix_rank(reference.effects.reshape(d * d, -1)) < d * d
    ):
        raise ValueError(
            "Sky/ground requires a minimal informationally complete reference POVM."
        )
    ground = (
        Measurement(np.eye(d), name="computational basis") if ground is None else ground
    )
    if ground.dimension != d:
        raise ValueError(
            "Ground and reference measurements must have the same dimension."
        )
    specs = []
    for index, state in enumerate(states):
        specs.extend(
            [
                _measurement_spec(f"sky/{index}", reference, state),
                _measurement_spec(f"ground/{index}", ground, state),
            ]
        )
    for index, state in enumerate(reference.states.T):
        specs.extend(
            [
                _measurement_spec(f"conditional/{index}", ground, state),
                _measurement_spec(f"reference/{index}", reference, state),
            ]
        )
    return Experiment(
        specs,
        name,
        {
            "kind": "sky_ground",
            "dimension": d,
            "n_states": len(states),
            "n_reference": reference.num_outcomes,
        },
        {"reference_effects": reference.effects, "ground_effects": ground.effects},
    )


def triple_product_experiment(states, triplets=None, *, name="triples"):
    """Build real and imaginary cycle tests for selected ordered triples.

    States are normalized kets or density matrices. By default only sorted
    triples are measured; permutation/conjugation symmetries fill the tensor.
    There are ``2 * binomial(m+2, 3)`` circuits for m states, so pass a subset
    of ``triplets`` for larger frames. Each circuit uses three copies and a
    control qubit; mixed inputs also use purification qubits.
    """
    from .triples import triple_product_circuit

    states = tuple(states)
    if not states:
        raise ValueError("Supply at least one state.")
    complete = triplets is None
    selected = (
        list(combinations_with_replacement(range(len(states)), 3))
        if complete
        else list(triplets)
    )
    if not selected:
        raise ValueError("Supply at least one triplet.")
    specs = []
    normalized_triplets = []
    for index, triple in enumerate(selected):
        if len(triple) != 3 or any(
            isinstance(i, bool)
            or not isinstance(i, (int, np.integer))
            or not 0 <= i < len(states)
            for i in triple
        ):
            raise ValueError("Triplet indices must identify three supplied states.")
        triple = [int(i) for i in triple]
        normalized_triplets.append(triple)
        for component in ("real", "imag"):
            circuit = triple_product_circuit(
                [states[i] for i in triple], component=component, key="triple"
            )
            measurement = next(
                op for op in circuit.all_operations() if cirq.is_measurement(op)
            )
            order = tuple(sorted(circuit.all_qubits()))
            specs.append(
                CircuitSpec(
                    f"triple/{index}/{component}",
                    circuit,
                    order,
                    measurement.qubits,
                    "triple",
                    2,
                    {"triplet": triple, "component": component},
                )
            )
    return Experiment(
        specs,
        name,
        {
            "kind": "triples",
            "triplets": normalized_triplets,
            "n_states": len(states),
            "complete": complete,
        },
    )


def compile_circuit(circuit, *, gateset=None):
    """Synthesize dense gates into one-/two-qubit operations, then a Cirq gateset.

    Uses quantum Shannon decomposition for arbitrary-size unitaries, followed
    by ``optimize_for_target_gateset`` (CZ by default). Measurements are kept
    intact. This performs gate synthesis, not device placement or routing.
    Gate counts grow exponentially with register size; inspect them before
    sending an experiment to hardware. Nonunitary multi-qubit channels fail
    explicitly rather than being mistaken for compilable gates.
    """
    from ._synthesis import shannon_operations

    operations = []
    for op in cirq.Circuit(circuit).all_operations():
        if cirq.is_measurement(op) or len(op.qubits) <= 2:
            operations.append(op)
        elif cirq.has_unitary(op):
            operations.extend(
                cirq.flatten_to_ops(
                    shannon_operations(list(op.qubits), cirq.unitary(op))
                )
            )
        else:
            raise ValueError(f"Cannot synthesize nonunitary operation {op!r}.")
    target = cirq.CZTargetGateset() if gateset is None else gateset
    return cirq.optimize_for_target_gateset(
        cirq.Circuit(operations), gateset=target, ignore_failures=False
    )


def wh_entropy_experiment(state, *, base=2, name="wh_entropy"):
    """Build a direct two-copy cyclic-WH Rényi-2 entropy experiment.

    Prepare rho and its computational-basis conjugate, then measure their
    displaced overlaps ``s_ab = Tr(rho D_ab rho D_ab†)/d``. Analysis uses
    ``M2 = -log(sum(s**2)/s[0])`` and ``purity = d*s[0]``. This applies to
    pure and mixed states and needs no tomographic inversion. It requires
    the ability to prepare the conjugate; it is not a universal conjugation
    channel for unknown physical inputs. For tensor-Pauli conventions or
    other Rényi orders use :func:`entropy_experiment` instead.
    """
    from .measurements import bell_overlap_circuit

    if not np.isfinite(base) or base <= 1:
        raise ValueError("Logarithm base must be finite and greater than one.")
    built = bell_overlap_circuit(state)
    d = np.asarray(state).shape[0]
    spec = CircuitSpec(
        "measurement",
        built.circuit,
        built.qubit_order,
        built.measured_qubits,
        built.measurement_key,
        d * d,
    )
    return Experiment(
        [spec], name, {"kind": "wh_entropy", "dimension": d, "base": float(base)}
    )
