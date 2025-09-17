from cirq_sic import *

def main():
    dataset_id = '9_17_25'

    n = 2
    data_collection_tasks = [
        SICOnSICTask(
            dataset_id=dataset_id,
            n_shots=5000,
            state_qubits = [cirq.GridQubit(5,9), cirq.GridQubit(6,9)],
            fiducial_qubits = [cirq.GridQubit(5,10), cirq.GridQubit(6,10)],
            processor_id = "willow_pink",
            run_type = "clean"
        )
    ]

    for dc_task in data_collection_tasks:
        run_sic_on_sic(dc_task)

if __name__ == '__main__':
    main()