#!/bin/bash -x

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1                    
#SBATCH --gres=gpu:4                  
#SBATCH --ntasks-per-node=4            
#SBATCH --cpus-per-task=24            
#SBATCH --time=00:10:00
#SBATCH --partition=dc-gpu
#SBATCH --account=training2529
#SBATCH --output=%j.out
#SBATCH --error=%j.err

#SBATCH --reservation=training2529

export CUDA_VISIBLE_DEVICES=0,1,2,3    # Very important to make the GPUs visible
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

source $HOME/course/sc_venv_template/activate.sh

time srun python3 imagenet_loaders.py --dset_type "fs" --data_root "/p/scratch/training2529/"