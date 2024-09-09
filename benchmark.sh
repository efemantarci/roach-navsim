#!/bin/bash
# Slurm script to benchmark rl agent. It benchmarks the model that has link in outputs/checkpoint.txt
#SBATCH --job-name=benchmark            
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=1    
#SBATCH --cpus-per-task=8
#SBATCH --partition=mid        
#SBATCH --qos=users        
#SBATCH --account=users    
#SBATCH --gres=gpu:tesla_t4:1    
#SBATCH --time=6:0:0        
#SBATCH --output=benchmark.out    
python benchmark.py
