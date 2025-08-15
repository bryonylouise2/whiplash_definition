#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --partition=32gb_20core
#SBATCH --mem=30G
#SBATCH --output=event_climo_%J_output.txt
#SBATCH --error=event_climo_%J_error.txt
#SBATCH --time=24:00:00
#SBATCH --job-name=event_climo
#SBATCH --mail-user=bryony.puxley@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bpuxley
#
#################################################
python -u /home/bpuxley/event_seasonal_climo.py
