#!/usr/bin/env bash

xRayTubeDataFileName=$1
gatheredXRayTubeDataFileName=$2
photonSpacingFileName=$3

currentSourceScanFileName_column1and5=$4
currentSourceScanFileName_column2and6=$5
currentSourceScanFileName_column3and7=$6
currentSourceScanFileName_column4and8=$7
gatheredCurrentSourceScanFileName=$8
analogGainsFileName=$9
digitalMeansFileName=${10}
processingFilesFolder=${11}

darkDataFileName=${12}
gatheredDarkDataFileName=${13}
darkOffsetFileName=${14}

photonSpacingCellNumber=${15}
keV_perPhoton=${16}

combinedCalibrationConstantsFileName=${17}

workspaceFolder=${18}

moduleNumber=${19}

batchJobCreationFolder=${20}

sbatch --job-name=processAgipdCalibration_module${moduleNumber} \
		--output=${workspaceFolder}batchJobProcessAgipdCalibration_m${moduleNumber}.output \
		--error=${workspaceFolder}batchJobProcessAgipdCalibration_m${moduleNumber}.error \
		${batchJobCreationFolder}batchJob_processAgipdCalibration_oneModule.sh \
		${xRayTubeDataFileName} \
		${gatheredXRayTubeDataFileName} \
		${photonSpacingFileName} \
		${currentSourceScanFileName_column1and5} \
        ${currentSourceScanFileName_column2and6} \
        ${currentSourceScanFileName_column3and7} \
        ${currentSourceScanFileName_column4and8} \
		${gatheredCurrentSourceScanFileName} \
		${analogGainsFileName} \
		${digitalMeansFileName} \
		${darkDataFileName} \
		${gatheredDarkDataFileName} \
        ${darkOffsetFileName} \
        ${photonSpacingCellNumber} \
        ${keV_perPhoton} \
        ${combinedCalibrationConstantsFileName} \
		${processingFilesFolder} 
		

