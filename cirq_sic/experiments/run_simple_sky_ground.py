from cirq_sic import *

EXPERIMENT_NAME = "sky_ground"
BASE_DIR = f'data/{EXPERIMENT_NAME}'

def main():
    kwargs = {"dataset_id": 'good',
              "processor_id": "willow_pink",
              "run_type": "clean",
              "n_shots": 10000,
              "state_qubits": [cirq.GridQubit(5,9), cirq.GridQubit(6,9)],
              "fiducial_qubits": [cirq.GridQubit(5,10), cirq.GridQubit(6,10)]}

    programs = [SimpleSICOnSIC, SimpleSICOnBasisStates, SimpleBasisMeasurementOnSIC, SimpleBasisMeasurementOnBasisStates]
    for program in programs:
        task = SimpleAKTask(**{**kwargs, "desc": program.desc})
        task.run(program=program, base_dir=BASE_DIR)

if __name__ == '__main__':
    main()