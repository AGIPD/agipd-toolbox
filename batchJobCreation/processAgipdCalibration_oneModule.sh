#!/usr/bin/env bash

xRayTubeDataFileName=$1
gatheredXRayTubeDataFileName=$2
photonSpacingFileName=$3

currentSourceScanFileName=$4
gatheredCurrentSourceScanFileName=$5
analogGainsFileName=$6
digitalMeansFileName=$7
processingFilesFolder=$8

workspaceFolder=${9}

moduleNumber=${10}

batchJobCreationFolder=${11}

sbatch --job-name=processAgipdCalibration_module${moduleNumber} \
		--output=${workspaceFolder}batchJobProcessMokalphaData_m${moduleNumber}.output \
		--error=${workspaceFolder}batchJobProcessMokalphaData_m${moduleNumber}.error \
		${batchJobCreationFolder}batchJob_processAgipdCalibration_oneModule.sh \
		${xRayTubeDataFileName} \
		${gatheredXRayTubeDataFileName} \
		${photonSpacingFileName} \
		${currentSourceScanFileName} \
		${gatheredCurrentSourceScanFileName} \
		${analogGainsFileName} \
		${digitalMeansFileName} \
		${processingFilesFolder} 
		

