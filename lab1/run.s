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

gcc -W -Wall -O3 ./lab1-c1-c2.c -o lab1

$SRCDIR/lab1

cat /proc/cpuinfo
