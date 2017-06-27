#!/usr/bin/env bash

#SBATCH --partition=all
#SBATCH --time=5:00:00
#SBATCH --workdir   /home/kuhnm/sbatch_out
#SBATCH --output    gather_drscs-%N-%j.out   # File to which STDOUT will be written
#SBATCH --error     gather_drscs-%N-%j.err   # File to which STDERR will be written
#SBATCH --nodes=1
#SBATCH --mail-type END                   # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user manuela.kuhn@desy.de  # Email to which notifications will be sent

BASE_PATH = /home/kuhnm/agipd-calibration/agipdCalibration/batchProcessing

python ${BASE_PATH}/gatherCurrentSourceScanData_generic.py --module M310_m7 --temperature temperature_m20C --current itestc80 --column_spec 5 6 7 8

