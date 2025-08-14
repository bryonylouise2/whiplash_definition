#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --partition=256gb_64core
#SBATCH --mem=250G
#SBATCH --output=density_calculation_%J_output.txt
#SBATCH --error=density_calculation_%J_error.txt
#SBATCH --time=1:00:00
#SBATCH --job-name=density_calculation
#SBATCH --mail-user=bryony.puxley@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bpuxley
#
#################################################
python -u /home/bpuxley/Density_Calculation.py
