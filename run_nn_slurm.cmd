#!/bin/bash
#SBATCH -N 1
#SBATCH -p pbatch
#SBATCH -A ibronze
#SBATCH -t 720

cd /usr/WS1/hammel1/proj/
source gpu_venv/bin/activate

cd  /usr/WS1/hammel1/proj/bayes/

srun -N 1 -n 16 run_nn

