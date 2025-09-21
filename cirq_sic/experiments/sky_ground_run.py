import argparse
from cirq_sic import *

EXPERIMENT_NAME = "sky_ground"
BASE_DIR = f'data/{EXPERIMENT_NAME}'

parser = argparse.ArgumentParser(prog="sky_ground")
parser.add_argument("-dataset_id", type=str, required=True)
parser.add_argument("-run_type", type=str, required=True)
parser.add_argument("-wh_implementation", type=str, required=True)
parser.add_argument("-n_shots", type=int, required=True)
params = vars(parser.parse_args())
globals().update(params)

params["processor_id"] = "willow_pink"
qubits = [cirq.GridQubit(5,9), cirq.GridQubit(6,9)] +\
         [cirq.GridQubit(5,10), cirq.GridQubit(6,10)]                    
if wh_implementation == "ak":
    qubits.extend([cirq.GridQubit(5,11), cirq.GridQubit(6,11)])
params["qubits"] = qubits
params["d"] = 4

for program in sky_ground_programs:
    task = WHSkyGroundTask(**{**params, "description": program.description})
    task.run(program, d4_sic_fiducial, base_dir=BASE_DIR)
    
