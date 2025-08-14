#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --partition=64gb_24core
#SBATCH --mem=62G
#SBATCH --output=check_precip_%J_output.txt
#SBATCH --error=check_precip_%J_error.txt
#SBATCH --time=1:00:00
#SBATCH --job-name=check_precip
#SBATCH --mail-user=bryony.puxley@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bpuxley
#
#################################################
python -u /home/bpuxley/check_precip_data.py
