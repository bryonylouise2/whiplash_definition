#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --partition=64gb_24core
#SBATCH --mem=62G
#SBATCH --output=spatial_%J_output.txt
#SBATCH --error=spatial_%J_error.txt
#SBATCH --time=24:00:00
#SBATCH --job-name=spatial_consistency
#SBATCH --mail-user=bryony.puxley@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bpuxley
#
#################################################
python -u /home/bpuxley/spatial_consistency.py
