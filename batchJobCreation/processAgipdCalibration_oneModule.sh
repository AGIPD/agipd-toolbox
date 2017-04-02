#!/usr/bin/env bash

xRayTubeDataFileName=$1
gatheredXRayTubeDataFileName=$2
photonSpacingFileName=$3

currentSourceScanFileName=$4
gatheredCurrentSourceScanFileName=$5
analogGainsFileName=$6
digitalMeansFileName=$7
processingFilesFolder=$8

darkDataFileName=$9
gatheredDarkDataFileName=${10}
darkOffsetFileName=${11}

photonSpacingCellNumber=${12}
keV_perPhoton=${13}

workspaceFolder=${14}

moduleNumber=${15}

batchJobCreationFolder=${16}

sbatch --job-name=processAgipdCalibration_module${moduleNumber} \
		--output=${workspaceFolder}batchJobProcessAgipdCalibration_m${moduleNumber}.output \
		--error=${workspaceFolder}batchJobProcessAgipdCalibration_m${moduleNumber}.error \
		${batchJobCreationFolder}batchJob_processAgipdCalibration_oneModule.sh \
		${xRayTubeDataFileName} \
		${gatheredXRayTubeDataFileName} \
		${photonSpacingFileName} \
		${currentSourceScanFileName} \
		${gatheredCurrentSourceScanFileName} \
		${analogGainsFileName} \
		${digitalMeansFileName} \
		${darkDataFileName} \
		${gatheredDarkDataFileName} \
        ${darkOffsetFileName} \
        ${photonSpacingCellNumber} \
        ${keV_perPhoton} \
		${processingFilesFolder} 
		

