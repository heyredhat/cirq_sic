## circ_sic

Implementing SIC's in `cirq`. 

### Installation

I recommend starting a new conda environment. Then install by navigating to the folder with `setup.py` and running `pip install -e .` Finally you can import from the library with `import cirq_sic`. Check out `notebooks/TaskExample.ipynb` for a walkthrough.

### A brief tour of the code

At the top level,

* `circuits.py` contains the building blocks for the `cirq` circuits, e.g. the controlled shift operator `CX` and at a higher level, `arthurs_kelly`. The functions are all generators which yield `cirq` gates. 

* `wh.py` contains helper funtions for explicitly constructing the matrix representations of Weyl-Heisenberg displacement operators, and WH-covariant POVMs from fiducial states. 

* `sics.py` contains `load_sic_fiducial` allowing you to instantly have a numerical representation of a SIC fiducial in all dimensions up to $d=151$. The fiducials are stored as text files in the `sic_povms` folder. It also contains an optimization routine for finding your own.

* `ansatz.py` implements an arbitrary state preparation ansatz which has the virtue that it is as easy to prepare the conjugate of a state as the state itself.

* `utils.py` contains a variety of useful methods, e.g. `rand_ket` for generating random states, `ptrace` for partial tracing, etc.

In `experiments`:

* `sky_ground.py` defines the basic `recirq` task functions which (a) specify the parameters for a battery of experiments, (b) construct the requisite circuits, and (c) process the data. So far we have: `CharacterizeWHReferenceDeviceTask`, `WHPOVMOnBasisStatesTask`, `BasisMeasurementOnWHStatesTask`, `BasisMeasurementOnBasisStatesTask`, `BasisMeasurementAfterWHPOVMOnBasisStatesTask`, and finally the more general, `WHPOVMOnStatesTask`. A task can be run with `run_sky_ground_task`: this builds the circuits, optimizes them, samples the circuits, and processes the results, saving the data, circuits, and logs in a particular file structure. 

* `sky_ground_analysis.py` provides functions for calculating the basic metrics for consistency in the QBist sky/ground scenario, and for generating plots of the results.

* `experiment_utils.py` provides helper functions for processing data, setting up samplers, and compiling circuits. In particular, `get_sampler` will have to be rewritten to run the circuits on the real quantum processor.

In `experiments/runners`:

* `sky_ground_run.py` is meant to be run from the command line. From its command line arguments, it builds a dictionary which specifies a sky/ground experiment (a full set of sky/ground tasks), runs all the sky/ground tasks, and then generates the plots of the results.

* `sky_ground_args.py`: edit this to generate an `args.txt` specifying all the experiments you want to run in parallel (as command line arguments for `sky_ground_run.py`).

* `parallel_runner.sh` is meant to be run from the command line e.g. as `./parallel_runner.sh args.txt sky_ground_run.py`. It spawns off a bunch of instances of `sky_ground_run.py`, one for each line of arguments in `args.txt`.

In `tests`:

* `test_circuits.py`, `test_sic.py`, and `test_tasks.py` provide unit tests for most of the functionality of `cirq_sic`. 

In `text`:

* TeX files for my notes `QuditArthursKelly` and for the slides `QuditArthursKellyPresentation`.

In `notebooks`:

* `TaskExample.ipynb`: Basic tutorial on how to run a sky/ground task or a whole battery of them, generating plots along the way. Also explains the directory structure of the saved data.

* `CircuitImages.ipynb`: generates images used in the TeX documents.

* `CircuitComparisons.ipynb`: work in progress. Comparing gate counts using the different optimizers.

* `EmbeddingTest.ipynb`: work in progress. Ironing out the kinks in doing a SIC in arbitrary dimensions.

* In `interpolation`: work in progress. Interpolating between stabilizer states and SICs and comparing sky/ground metrics.

### To come:

* Fix some bugs in the arbitrary dimension SIC code, and judge whether it is efficiently implementable.

* Nice visualization of gate count comparisons.

* Best way of generating a 1-parameter family of states with increasing magic?

* Best way of judging noncontextuality in the sky/ground scenario: average negativity? Which states have maximal negativity?








