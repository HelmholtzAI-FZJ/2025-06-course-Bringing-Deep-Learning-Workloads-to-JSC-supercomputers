#!/bin/bash
#SBATCH --account=training2529
#SBATCH --mail-user=MYUSER@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --job-name=cat-detector
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --output=output.%j
#SBATCH --error=error.%j
#SBATCH --time=00:10:00
#SBATCH --partition=dc-gpu
##SBATCH --reservation=training2529 # For today only

cd $HOME/course/2025-06-course-Bringing-Deep-Learning-Workloads-to-JSC-supercomputers/code/fastai
source sc_venv_template/activate.sh # Now we finally use the fastai module

srun python cats.py
