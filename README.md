# Haskap Pie: 

A Halo finding Algorithm with efficient Sampling, K-means clustering, tree-Assembly, Particle tracking, Python modules, Inter-code applicability, and Energy solving


version v1.0.0

## References

When using this code, please cite the following papers:

Haskap Pie: Barrow, Nguyen, & Scrabacz 2025, ApJ, [Edition], [Article], [Link]
Haskap Pie II: Barrow, Nguyen, & Scrabacz 2025, ApJ, In Prep

## Installation

Using git, Haskap Pie can be installed with the following:
```
cd /path/to/install/
git clone https://github.com/ksbarrow/Haskap_Pie.git
pip install .
```

Using pip, Haskap Pie can be installed with the following:
```
python -m pip install haskap --target /path/to/install/
```

## Documentation

Haskap Pie can be used on both on an individual machine or a High Performance Computing (HPC) cluster. In [examples](examples/), there are example bash files to run Haskap Pie.

Dependencies should be installed with this package, but are also listed in [requirements.txt](requirements.txt)

To run Haskap Pie, your simulation must be one of the following codes:

- [`ENZO`]()
- [`GADGET3`]()
- [`AREPO`]()
- [`ART`]()
- [`GIZMO`]()
- [`CHANGA`]()
- [`GEAR`]()
- [`RAMSES`]()

Your directory structure should following the following

```
project_dir
|
|-Haskap_Pie
|--src/haskap/
|
|-/path/to/sims/box1 (or name of simulation box here)
```

To run Haskap Pie, use the example bash file `haskap.sh` in the following way from the root directory

`bash Haskap_Pie/src/haskap/haskap.sh /path/to/sims/box1 code_type /path/to/save/ num_skip`

where `code_tp` is the code type as mentioned above and `num_skip` is the number of snapshots to skip.