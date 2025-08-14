#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --partition=32gb_20core
#SBATCH --mem=30G
#SBATCH --output=check_spi_%J_output.txt
#SBATCH --error=check_spi_%J_error.txt
#SBATCH --time=1:00:00
#SBATCH --job-name=check_spi
#SBATCH --mail-user=bryony.puxley@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bpuxley
#
#################################################
python -u /home/bpuxley/Check_SPI_Data.py
