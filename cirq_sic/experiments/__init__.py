"""Lazy compatibility exports for the legacy experiment API."""

from importlib import import_module

_EXPORTS = {
    "BasisMeasurementAfterWHPOVMOnBasisStatesTask": "sky_ground",
    "BasisMeasurementOnBasisStatesTask": "sky_ground",
    "BasisMeasurementOnWHStatesTask": "sky_ground",
    "CharacterizeWHReferenceDeviceTask": "sky_ground",
    "CharacterizeWHTripleProductsTask": "sky_ground",
    "DEFAULT_BASE_DIR": "sky_ground",
    "EXPERIMENT_NAME": "sky_ground",
    "P_img": "sky_ground_analysis",
    "Phi_img": "sky_ground_analysis",
    "SIMPLE_WH_IMPLEMENTATIONS": "sky_ground",
    "WHPOVMOnBasisStatesTask": "sky_ground",
    "WHPOVMOnStatesTask": "sky_ground",
    "abbrev_grid_qubits": "experiment_utils",
    "abbrev_n_shots": "experiment_utils",
    "add_caption": "sky_ground_analysis",
    "are_all_measurements_terminal": "experiment_utils",
    "avg_negativity": "experiment_utils",
    "bqskit_machine_model": "experiment_utils",
    "bqskit_optimize_circuit": "experiment_utils",
    "cirq_optimize_circuit": "experiment_utils",
    "collect_circuit_files": "experiment_utils",
    "collect_circuits": "experiment_utils",
    "create_circuits": "sky_ground",
    "deep_match": "experiment_utils",
    "default_project_id": "experiment_utils",
    "exact_simulation": "experiment_utils",
    "exactify": "sky_ground",
    "get_gate_counts": "experiment_utils",
    "get_sampler": "experiment_utils",
    "get_wh_qubits": "sky_ground",
    "has_dynamic_circuit_features": "experiment_utils",
    "has_post_measurement_task": "sky_ground",
    "load_circuits": "sky_ground",
    "load_results": "sky_ground",
    "load_sky_ground_results": "sky_ground",
    "measurement_key_sort_key": "experiment_utils",
    "optimize_task_circuits": "sky_ground",
    "ordered_measurement_keys": "experiment_utils",
    "p_img": "sky_ground_analysis",
    "plot_matrix_comparison": "sky_ground_analysis",
    "pq_img": "sky_ground_analysis",
    "print_all_gate_counts": "experiment_utils",
    "process_task_results": "sky_ground",
    "push_measurements_to_end": "experiment_utils",
    "q_img": "sky_ground_analysis",
    "query_records": "experiment_utils",
    "remove_task_segment": "sky_ground_analysis",
    "result_measurements_matrix": "experiment_utils",
    "results_to_freqs": "experiment_utils",
    "run_circuits": "sky_ground",
    "run_sky_ground_task": "sky_ground",
    "run_sky_ground_task_from_specs": "sky_ground",
    "run_sky_ground_tasks": "sky_ground",
    "setup_logger": "experiment_utils",
    "shots_to_freqs": "experiment_utils",
    "sky_ground_images": "sky_ground_analysis",
    "sky_ground_img_funcs": "sky_ground_analysis",
    "sky_ground_metrics": "sky_ground_analysis",
    "sky_ground_tasks": "sky_ground",
    "task_from_specs": "sky_ground",
    "terminal_measurement_ops": "experiment_utils",
    "to_one_and_two_qubit_ops": "experiment_utils",
    "wh_povm_circuit": "sky_ground",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
