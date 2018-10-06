#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16GB
#sbatch --output=lab1.out
#SBATCH --job-name=lab1
#SBATCH --mail-type=END
#SBATCH --mail-user=jma587@nyu.edu

SRCDIR=$HOME/hpc4ml/lab1

cd $SRCDIR

module purge

module load python3/intel/3.5.3

python3 lab1-c3-c4.py

cat /proc/cpuinfo
