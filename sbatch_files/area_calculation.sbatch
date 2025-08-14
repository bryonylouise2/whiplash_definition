#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --partition=32gb_20core
#SBATCH --mem=30G
#SBATCH --output=area_calculation_%J_output.txt
#SBATCH --error=area_calculation_%J_error.txt
#SBATCH --time=6:00:00
#SBATCH --job-name=area_calculation
#SBATCH --mail-user=bryony.puxley@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bpuxley
#
#################################################
python -u /home/bpuxley/Area_Calculation.py
