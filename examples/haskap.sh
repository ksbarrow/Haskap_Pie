#!/bin/bash

## Recommended settings to run on an individual machine. This can also be used as an individual bash script.


## $1 is location of simulation snapshots: eg. /path/to/sim/box1/
## $2 is code type, one of ENZO, GADGET3, AREPO, GIZMO, ART, CHANGA, GEAR, RAMSES, manual
## $3 is same os save directory: eg. box1
## $4 is number of timesteps to skip, default should be 1.

mpirun -n X python run_haskap.py $1 $2 $3 $4