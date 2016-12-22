#!/usr/bin/env bash

#SBATCH --partition=cfel
#SBATCH --time=48:00:00                 
#SBATCH --nodes=1
#SBATCH --cores-per-socket=16

mokalphaDataFileName=$1
gatheredMokalphaFileName=$2
photonSpacingFileName=$3

pulsedCapacitorDataFolder=$4
pulsedCapacitorModuleNumber=$5
gatheredPulsedCapacitorDataSaveFileName=$6
analogGainsSaveFileName=$7
digitalMeansSaveFileName=$8

processingFilesFolder=$9


python ${processingFilesFolder}batchProcessing/gatherMokalphaData.py ${mokalphaDataFileName} ${gatheredMokalphaFileName}

python ${processingFilesFolder}batchProcessing/batchProcessMokalpha.py ${gatheredMokalphaFileName} ${photonSpacingFileName}

python ${processingFilesFolder}batchProcessing/gatherPulsedCapacitorData.py ${pulsedCapacitorDataFolder} ${pulsedCapacitorModuleNumber} ${gatheredPulsedCapacitorDataSaveFileName}

python ${processingFilesFolder}batchProcessing/batchProcessPulsedCapacitor.py ${gatheredPulsedCapacitorDataSaveFileName} ${analogGainsSaveFileName} ${digitalMeansSaveFileName}