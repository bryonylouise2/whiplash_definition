#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --partition=32gb_20core
#SBATCH --mem=30G
#SBATCH --output=event_examination_%J_output.txt
#SBATCH --error=event_examination_%J_error.txt
#SBATCH --time=12:00:00
#SBATCH --job-name=event_examination
#SBATCH --mail-user=bryony.puxley@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bpuxley
#
#################################################
python -u /home/bpuxley/Event_Examination.py
