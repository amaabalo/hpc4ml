#!/bin/bash

#SBATCH --job-name=lab2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30GB
#SBATCH --gres=gpu
#SBATCH --time=10:00:00
#SBATCH --output=out.%j

#C5.1
/home/am9031/anaconda3/bin/python lab2.py /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train.csv /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train-jpg --enable-cuda --n_workers 1 --optimiser 'sgd'

##C5.2
/home/am9031/anaconda3/bin/python lab2.py /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train.csv /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train-jpg --enable-cuda --n_workers 12 --optimiser 'sgd'
/home/am9031/anaconda3/bin/python lab2.py /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train.csv /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train-jpg --enable-cuda --n_workers 12 --optimiser 'sgdwithnesterov'
/home/am9031/anaconda3/bin/python lab2.py /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train.csv /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train-jpg --enable-cuda --n_workers 12 --optimiser 'adagrad'
/home/am9031/anaconda3/bin/python lab2.py /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train.csv /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train-jpg --enable-cuda --n_workers 12 --optimiser 'adadelta'
/home/am9031/anaconda3/bin/python lab2.py /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train.csv /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train-jpg --enable-cuda --n_workers 12 --optimiser 'adam'
