# Haskap Pie: 

A Halo finding Algorithm with efficient Sampling, K-means clustering, tree-Assembly, Particle tracking, Python modules, Inter-code applicability, and Energy solving


version v1.0.0

## References

When using this code, please cite the following papers:

Haskap Pie: Barrow, Nguyen, & Scrabacz 2025, ApJ, [Edition], [Article], [Link] \
Haskap Pie II: Barrow, Nguyen, & Scrabacz 2025, ApJ, In Prep

A draft of Haskap Pie I is available [here](Haskap_Pie_Paper_Draft.pdf) or on the [arXiv](https://arxiv.org/abs/2505.22709)

## Installation

Haskap Pie requires python 3.12 or higher and yt 4.2 or higher. We recommend installing to a virtual environment using the following commands:

```
cd /path/to/install/
python -m venv venv
source venv/bin/activate
conda deactivate
python --version
```

Ensure that the version is greater than or equal to python 3.12.

Then, Haskap Pie can be installed in two ways.

Using git, Haskap Pie can be installed with the following:
```
cd /path/to/install/
git clone https://github.com/ksbarrow/Haskap_Pie.git
cd Haskap_Pie
pip install .
cd ..
```

To ensure it has installed correctly, run either `pip list` or `pip freeze` to ensure it is in the list of installed packages.

Using pip, Haskap Pie can be installed from PyPI with the following:
```
pip install haskap
```

We recommend using git, as it will contain the most recent version of Haskap Pie.

## Documentation

Haskap Pie can be used on both on an individual machine or a High Performance Computing (HPC) cluster. In [examples](examples/), there are example bash files to run Haskap Pie.

Dependencies should be installed with this package, but are also listed in [requirements.txt](requirements.txt)

To run Haskap Pie, your simulation must be one of the following codes:

- [`ENZO`](https://github.com/enzo-project/enzo-dev)
- [`GADGET3`](https://github.com/sbird/MP-Gadget3)
- [`AREPO`](https://arepo-code.org/wp-content/userguide/index.html)
- ART
- [`GIZMO`](http://www.tapir.caltech.edu/~phopkins/Site/GIZMO.html)
- [`CHANGA`](https://faculty.washington.edu/trq/hpcc/tools/changa.html)
- [`GEAR`](https://github.com/mladenivkovic/thesis_public)
- [`RAMSES`](https://github.com/ramses-organisation/ramses)

Your directory structure should following the following

```
project_dir
|
|-Haskap_Pie
|-venv
|
|-/path/to/sims/box1 (or name of simulation box here)
```

To create a script for intialize and run Haskap Pie, see the example script in [examples](examples/). This initializes and calls all revelvant functions from helper files, and calls the main `Evolve_Tree function`. 

To run Haskap Pie, use the example bash file `haskap.sh` in the following way from the root directory

Inside `haskap.sh` you will find an mpirun command. The 'X' will have to be replaced with the number of threads.

```
bash run_haskap.py /path/to/sims/box1 code_type /path/to/save/ num_skip
```

where `code_tp` is the code type as mentioned above and `num_skip` is the number of snapshots to skip.

To run on a High Performance Cluster with slurm scheduling, we provide `haskap_HPC.sh`. To run this, follow the same directory structure as above and use the following command:

```
sbatch run_haskap.py /path/to/sims/box1 code_type /path/to/save/ num_skip
```

changing the required account settings as needed.

For any questions regarding Haskap Pie, contact Dr. Kirk Barrow  at kbarrow [at] illinois [dot] edu