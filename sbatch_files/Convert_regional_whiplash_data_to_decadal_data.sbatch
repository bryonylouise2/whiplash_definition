#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --partition=normal_lake
#SBATCH --mem=126G
#SBATCH --output=combine_whiplash_%J_output.txt
#SBATCH --error=combine_whiplash_%J_error.txt
#SBATCH --time=2:00:00
#SBATCH --job-name=combine_whiplash
#SBATCH --mail-user=bryony.puxley@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bpuxley
#
#################################################
python -u /home/bpuxley/Convert_regional_whiplash_data_to_decadal_data.py
