#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --partition=64gb_24core
#SBATCH --mem=62G
#SBATCH --output=calculate_spi_%J_output.txt
#SBATCH --error=calculate_spi_%J_error.txt
#SBATCH --time=18:00:00
#SBATCH --job-name=calculate_spi
#SBATCH --mail-user=bryony.puxley@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bpuxley
#
#################################################
python -u /home/bpuxley/CalculatingSPI.py
