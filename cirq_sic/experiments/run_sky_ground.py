from cirq_sic import *

EXPERIMENT_NAME = "sky_ground"
BASE_DIR = f'data/{EXPERIMENT_NAME}'

def main():
    task_data = {"dataset_id": "9_19_25",
                 "processor_id": "willow_pink",
                 "d": 4}

    for run_type in ["clean", "noisy"]:
        for n_shots in [500, 1000, 5000, 10000, 20000, 50000, 100000]:
            for wh_implementation in ["ak", "simple"]:
                qubits = [cirq.GridQubit(5,9), cirq.GridQubit(6,9)] +\
                         [cirq.GridQubit(5,10), cirq.GridQubit(6,10)]                    
                if wh_implementation == "ak":
                    qubits.extend([cirq.GridQubit(5,11), cirq.GridQubit(6,11)])
                for program in sky_ground_programs:
                    task = WHSkyGroundTask(*{**task_data, "description": program.description,
                                                          "run_type": run_type, 
                                                          "wh_implementation": wh_implementation,
                                                          "n_shots": n_shots,
                                                          "qubits": qubits})
                    task.run(program, d4_sic_fiducial, base_dir=BASE_DIR)

if __name__ == '__main__':
    main()