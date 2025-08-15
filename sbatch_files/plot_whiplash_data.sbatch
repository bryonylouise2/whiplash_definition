#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --partition=32gb_20core
#SBATCH --mem=30G
#SBATCH --output=plot_whiplash_%J_output.txt
#SBATCH --error=plot_whiplash_%J_error.txt
#SBATCH --time=1:00:00
#SBATCH --job-name=plot_whiplash
#SBATCH --mail-user=bryony.puxley@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bpuxley
#
#################################################
python -u /home/bpuxley/plot_whiplash_data.py
