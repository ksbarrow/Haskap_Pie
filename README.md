# Haskap Pie: 

A Halo finding Algorithm with efficient Sampling, K-means clustering, tree-Assembly, Particle tracking, Python modules, Inter-code applicability, and Energy solving


## References

When using this code, please cite the following papers:

Haskap Pie: Barrow, Nguyen, & Scrabacz 2026, ApJ, 999, 72, [Link](http://doi.org/10.3847/1538-4357/ae2eb4) \
AGORA XI Halo Morphologies: Barrow et al, and the AGORA Collaboration, ApJ, In Prep


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


Using git, Haskap Pie can be installed with the following:
```
cd /path/to/install/
git clone https://github.com/ksbarrow/Haskap_Pie.git
cd Haskap_Pie
pip install .
cd ..
```

To ensure it has installed correctly, run either `pip list` or `pip freeze` to ensure it is in the list of installed packages.


We recommend using git, as it will contain the most recent version of Haskap Pie.

To use Haskap Pie on a Windows System, the package needs to be installed in the Windows Subsystem for Linux (WSL). Some dependencies for Haskap Pie do not compile on Windows systems.

## Documentation

Haskap Pie can be used on both on an individual machine or a High Performance Computing (HPC) cluster. In [examples](examples/), there are example bash files to run Haskap Pie.

Dependencies should be installed with this package. Haskap Pie also needs a minimum of 5 timesteps/snapshots of the simulation you are attemping to run it on.

To run Haskap Pie, your simulation must be one of the following codes:

- [`ENZO`](https://github.com/enzo-project/enzo-dev)
- [`GADGET3`](https://github.com/sbird/MP-Gadget3)
- [`AREPO`](https://arepo-code.org/wp-content/userguide/index.html)
- ART
- [`GIZMO`](http://www.tapir.caltech.edu/~phopkins/Site/GIZMO.html)
- [`CHANGA`](https://faculty.washington.edu/trq/hpcc/tools/changa.html)
- [`GEAR`](https://github.com/mladenivkovic/thesis_public)
- [`RAMSES`](https://github.com/ramses-organisation/ramses)
- GADGET4

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
bash haskap.sh /path/to/sims/box1 code_type /path/to/save/ num_skip
```

where `code_tp` is the code type as mentioned above and `num_skip` is the number of snapshots to skip.

To run on a High Performance Cluster with slurm scheduling, we provide `haskap_HPC.sh`. To run this, follow the same directory structure as above and use the following command:

```
sbatch haskap_HPC.sh /path/to/sims/box1 code_type /path/to/save/ num_skip
```

changing the required account settings as needed.

For any questions regarding Haskap Pie, contact Dr. Kirk Barrow  at kbarrow [at] illinois [dot] edu
