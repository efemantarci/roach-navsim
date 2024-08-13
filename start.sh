#!/bin/bash

#SBATCH --job-name=ep_and_reward            
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=1    
#SBATCH --cpus-per-task=16
#SBATCH --partition=long        
#SBATCH --qos=users        
#SBATCH --account=users    
#SBATCH --gres=gpu:tesla_t4:1    
#SBATCH --time=72:0:0        
#SBATCH --output=ep_and_reward.out    
python train_rl.py
