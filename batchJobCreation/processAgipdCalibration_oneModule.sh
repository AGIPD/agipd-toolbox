#!/usr/bin/env bash

mokalphaDataFileName=$1
gatheredMokalphaFileName=$2
photonSpacingFileName=$3

pulsedCapacitorDataFolder=$4
pulsedCapacitorModuleNumber=$5
gatheredPulsedCapacitorDataSaveFileName=$6
analogGainsSaveFileName=$7
digitalMeansSaveFileName=$8
processingFilesFolder=$9

workspaceFolder=${10}

moduleNumber=${11}

batchJobCreationFolder=${12}

sbatch --job-name=processAgipdCalibration_module${moduleNumber} \
		--output=${workspaceFolder}batchJobProcessMokalphaData_m${moduleNumber}.output \
		--error=${workspaceFolder}batchJobProcessMokalphaData_m${moduleNumber}.error \
		${batchJobCreationFolder}batchJob_processAgipdCalibration_oneModule.sh \
		${mokalphaDataFileName} \
		${gatheredMokalphaFileName} \
		${photonSpacingFileName} \
		${pulsedCapacitorDataFolder} \
		${pulsedCapacitorModuleNumber} \
		${gatheredPulsedCapacitorDataSaveFileName} \
		${analogGainsSaveFileName} \
		${digitalMeansSaveFileName} \
		${processingFilesFolder} 
		

