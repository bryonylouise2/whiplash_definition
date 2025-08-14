#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --partition=normal_lake
#SBATCH --mem=126G
#SBATCH --output=combine_spi_%J_output.txt
#SBATCH --error=combine_spi_%J_error.txt
#SBATCH --time=12:00:00
#SBATCH --job-name=combine_spi
#SBATCH --mail-user=bryony.puxley@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bpuxley
#
#################################################
python -u /home/bpuxley/Convert_regional_spi_files_to_decadal_files.py
