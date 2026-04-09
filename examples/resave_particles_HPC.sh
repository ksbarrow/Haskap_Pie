#!/bin/bash

## Recommended settings to run on a High Performance Cluster (HPC). This can also be used as an individual bash script.

## Following are slurm settings

#SBATCH --mem=200g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=30
#SBATCH --cpus-per-task=1  
#SBATCH --partition=cpu   #enter HPC cpu partition here
#SBATCH --account= #enter allocation here
#SBATCH --job-name=resave_particles
#SBATCH --time=03:00:00   # hh:mm:ss for the job
#SBATCH --constraint="scratch"
#SBATCH --error=resave_particles.e%j
#SBATCH --output=resave_particles.o%j

## $1 is code type, one of ENZO, GADGET3, GADGET4, AREPO, GIZMO, ART, CHANGA, GEAR, RAMSES, manual
## $2 is os save directory: eg. box1

date
mpirun python resave_particles.py $1 $2
date
