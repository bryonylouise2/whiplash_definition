#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --partition=32gb_20core
#SBATCH --mem=30G
#SBATCH --output=whiplash_%J_output.txt
#SBATCH --error=whiplash_%J_error.txt
#SBATCH --time=1:00:00
#SBATCH --job-name=whiplash_identifcation_%J
#SBATCH --mail-user=bryony.puxley@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bpuxley
#
#################################################
python -u /home/bpuxley/Whiplash_Identification.py
