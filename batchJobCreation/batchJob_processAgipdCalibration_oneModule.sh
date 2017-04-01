#!/usr/bin/env bash

#SBATCH --partition=cfel
#SBATCH --time=48:00:00                 
#SBATCH --nodes=1
#SBATCH --cores-per-socket=16

xRayTubeDataFileName=$1
gatheredXRayTubeDataFileName=$2
photonSpacingFileName=$3

currentSourceScanFileName=$4
gatheredCurrentSourceScanFileName=$5
analogGainsFileName=$6
digitalMeansFileName=$7

processingFilesFolder=$8


python ${processingFilesFolder}batchProcessing/gatherXRayTubeData.py ${xRayTubeDataFileName} ${gatheredXRayTubeDataFileName}

python ${processingFilesFolder}batchProcessing/batchProcessXRayTubeData.py ${gatheredXRayTubeDataFileName} ${photonSpacingFileName}

python ${processingFilesFolder}batchProcessing/gatherPulsedCapacitorData.py ${currentSourceScanFileName} ${gatheredCurrentSourceScanFileName}

#python ${processingFilesFolder}batchProcessing/batchProcessPulsedCapacitor.py ${gatheredPulsedCapacitorDataSaveFileName} ${analogGainsSaveFileName} ${digitalMeansSaveFileName}
python ${processingFilesFolder}batchProcessing/batchProcessCurrentSourceScan.py ${gatheredCurrentSourceScanFileName} ${analogGainsFileName} ${digitalMeansFileName}