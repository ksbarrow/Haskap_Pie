#!/bin/bash

## Recommended settings to run on a High Performance Cluster (HPC). This can also be used as an individual bash script.

## Following are slurm settings

#SBATCH --exclusive
#SBATCH --mem=220g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=30
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu #enter HPC cpu partition here
#SBATCH --account= #enter allocation here
#SBATCH --job-name=tree_edit
#SBATCH --time=48:00:00   # hh:mm:ss for the job
#SBATCH --constraint="scratch"
#SBATCH --error=tree_edit.e%j
#SBATCH --output=tree_edit.o%j


## $1 is location of simulation snapshots: eg. /path/to/sim/box1/
## $2 is code type, one of ENZO, GADGET3, AREPO, GIZMO, ART, CHANGA, GEAR, RAMSES, manual
## $3 is same os save directory: eg. box1
## $4 is number of timesteps to skip, default should be 1.

srun python run_haskap.py $1 $2 $3 $4