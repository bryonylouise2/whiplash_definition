#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --partition=largemem
#SBATCH --mem=300G
#SBATCH --output=independence_%J_output.txt
#SBATCH --error=independence_%J_error.txt
#SBATCH --time=48:00:00
#SBATCH --job-name=independence
#SBATCH --mail-user=bryony.puxley@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bpuxley
#
#################################################
python -u /home/bpuxley/independence.py
