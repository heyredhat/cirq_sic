import argparse

import cirq
import numpy as np

from cirq_sic import *

####################################################################################

EXPERIMENT_NAME = "sky_ground"
BASE_DIR = f'data/{EXPERIMENT_NAME}'

####################################################################################

def main():
    parser = argparse.ArgumentParser(prog="sky_ground")
    parser.add_argument("-dataset_id", type=str, required=True)
    parser.add_argument("-run_type", type=str, required=True)
    parser.add_argument("-wh_implementation", type=str, required=True)
    parser.add_argument("-n_shots", type=int, required=True)
    parser.add_argument("-d", type=int, required=True)
    parser.add_argument("-flag", type=str, required=True)
    params = vars(parser.parse_args())
    globals().update(params)

    params["processor_id"] = "willow_pink"

    n = int(np.log2(d))
    cols = 2 if wh_implementation == "simple" else 3
    params["qubits"] = cirq.GridQubit.rect(cols, n, top=4, left=2)

    if flag == "d4_monomial":
        prepare_fiducial = d4_sic_fiducial
    elif flag == "numerical_sic":
        phi = load_sic_fiducial(d)
        prepare_fiducial = ansatz_circuit(phi)

    for program, data_label in sky_ground_programs:
        task = WHSkyGroundTask(**{**params, "description": program.__name__})
        task.run(program, prepare_fiducial=prepare_fiducial, base_dir=BASE_DIR)

if __name__ == "__main__":
    main()