#!/bin/bash

#SBATCH --job-name=lab2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=10GB
#SBATCH --time=10:00:00
#SBATCH --partition=c32_41
#SBATCH --output=out.%j


##C3
/home/am9031/anaconda3/bin/python lab2.py /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train.csv /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train-jpg --n_workers 0
/home/am9031/anaconda3/bin/python lab2.py /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train.csv /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train-jpg --n_workers 1
## ^^^this command also produces the output needed in C4 and C5.1
/home/am9031/anaconda3/bin/python lab2.py /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train.csv /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train-jpg --n_workers 2
/home/am9031/anaconda3/bin/python lab2.py /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train.csv /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train-jpg --n_workers 4
/home/am9031/anaconda3/bin/python lab2.py /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train.csv /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train-jpg --n_workers 8
/home/am9031/anaconda3/bin/python lab2.py /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train.csv /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train-jpg --n_workers 12
/home/am9031/anaconda3/bin/python lab2.py /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train.csv /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train-jpg --n_workers 16
/home/am9031/anaconda3/bin/python lab2.py /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train.csv /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train-jpg --n_workers 20
/home/am9031/anaconda3/bin/python lab2.py /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train.csv /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train-jpg --n_workers 24
/home/am9031/anaconda3/bin/python lab2.py /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train.csv /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train-jpg --n_workers 28
/home/am9031/anaconda3/bin/python lab2.py /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train.csv /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train-jpg --n_workers 32

##C4
/home/am9031/anaconda3/bin/python -m cProfile -o lab2_1worker.prof lab2.py /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train.csv /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train-jpg --n_workers 1
/home/am9031/anaconda3/bin/python -m cProfile -o lab2_12worker.prof lab2.py /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train.csv /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train-jpg --n_workers 12
