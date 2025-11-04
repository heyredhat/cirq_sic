from cirq_sic import *

d = 2
base_dir = "data/sky_ground"
wh_implementation = "simple"

psi_stab = np.eye(d)[0]
psi_sic = load_sic_fiducial(d)
interpolation = geodesic_interpolator(psi_stab, psi_sic)

T = np.linspace(0, 1, 15)
specs_t = {t: {"dataset_id": "interpolation",
               "processor_id": "willow_pink",
               "run_type": "noisy",
               "qubits": get_wh_qubits(d, wh_implementation),
               "n_shots": 50000,
               "optimizer": "cirq",
               "d": d,
               "fiducial": interpolation(t),
               "fiducial_description": f"interpolation_{np.round(t, 4)}",
               "wh_implementation": wh_implementation} for t in T}

for t, specs in specs_t.items():
    run_sky_ground_tasks(specs, base_dir=base_dir)