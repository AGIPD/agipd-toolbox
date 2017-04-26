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

darkDataFileName=$8
gatheredDarkDataFileName=$9
darkOffsetFileName=${10}

photonSpacingCellNumber=${11}
keV_perPhoton=${12}

combinedCalibrationConstantsFileName=${13}

processingFilesFolder=${14}

python ${processingFilesFolder}batchProcessing/gatherDarkData.py ${darkDataFileName} ${gatheredDarkDataFileName}

python ${processingFilesFolder}batchProcessing/batchProcessDarkData.py ${gatheredDarkDataFileName} ${darkOffsetFileName}

python ${processingFilesFolder}batchProcessing/gatherXRayTubeData.py ${xRayTubeDataFileName} ${gatheredXRayTubeDataFileName}

python ${processingFilesFolder}batchProcessing/batchProcessXRayTubeData.py ${gatheredXRayTubeDataFileName} ${photonSpacingFileName}

python ${processingFilesFolder}batchProcessing/gatherPulsedCapacitorData.py ${currentSourceScanFileName} ${gatheredCurrentSourceScanFileName}

python ${processingFilesFolder}batchProcessing/batchProcessCurrentSourceScan.py ${gatheredCurrentSourceScanFileName} ${analogGainsFileName} ${digitalMeansFileName}

python ${processingFilesFolder}combineCalibrationData.py ${analogGainsFileName} ${digitalMeansFileName} ${darkOffsetFileName} ${photonSpacingFileName} ${photonSpacingCellNumber} ${keV_perPhoton} ${combinedCalibrationConstantsFileName}




