#!/usr/bin/env bash

#SBATCH --partition=all
#SBATCH --time=5:00:00
#SBATCH --workdir   /home/mmuster/sbatch_out
#SBATCH --output    gather_drscs-%N-%j.out   # File to which STDOUT will be written
#SBATCH --error     gather_drscs-%N-%j.err   # File to which STDERR will be written
#SBATCH --nodes=1
#SBATCH --mail-type END                   # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user max.muster@desy.de  # Email to which notifications will be sent

BASE_PATH=/home/mmuster/agipd-calibration/agipdCalibration/batchProcessing
INPUT_PATH=7-modules
#INPUT_PATH=302-303-314-305

python ${BASE_PATH}/gatherCurrentSourceScanData_generic.py --input_path ${INPUT_PATH} --module M309_m3 --temperature temperature_m25C --current itestc150

