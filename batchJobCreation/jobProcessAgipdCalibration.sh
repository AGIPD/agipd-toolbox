#!/usr/bin/env bash

workspaceFolder=/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/

batchJobCreationFolder=${AGIPDCALIBRATION_INSTALLATION_DIR}batchJobCreation/
processingFilesFolder=${AGIPDCALIBRATION_INSTALLATION_DIR}agipdCalibration/

keV_perPhoton=1
photonSpacingCellNumber=175

moduleNumbersToProcess=(1 2 3 4)

for moduleNumber in ${moduleNumbersToProcess[*]}
do
    darkDataFileName=/gpfs/cfel/fsds/labs/processed/M213_dark/data_klyuev/m${moduleNumber}_dark1kclk100im_00003.nxs
    gatheredDarkDataFileName=/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/darkData_m${moduleNumber}.h5
    darkOffsetFileName=/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/darkOffset_m${moduleNumber}.h5

    xRayTubeDataFileName=/gpfs/cfel/fsds/labs/calibration/current/1Mpix_calib/wing2/xray/m${moduleNumber}_xray_00003.nxs
    gatheredXRayTubeDataFileName=${workspaceFolder}xRayTubeData_m${moduleNumber}.h5
    photonSpacingFileName=${workspaceFolder}photonSpacing_m${moduleNumber}.h5

    currentSourceScanFileName=/gpfs/cfel/fsds/labs/processed/m${moduleNumber}_m233_drscsvr160_i80_00002.nxs
    gatheredCurrentSourceScanFileName=${workspaceFolder}currentSourceScanData_m${moduleNumber}_chunked.h5
    analogGainsFileName=${workspaceFolder}analogGains_m${moduleNumber}.h5
    digitalMeansFileName=${workspaceFolder}digitalMeans_m${moduleNumber}.h5

    ./processAgipdCalibration_oneModule.sh ${xRayTubeDataFileName} \
                                                ${gatheredXRayTubeDataFileName} \
                                                ${photonSpacingFileName} \
                                                ${currentSourceScanFileName} \
                                                ${gatheredCurrentSourceScanFileName} \
                                                ${analogGainsFileName} \
                                                ${digitalMeansFileName} \
                                                ${processingFilesFolder} \
                                                ${darkDataFileName} \
                                                ${gatheredDarkDataFileName} \
                                                ${darkOffsetFileName} \
                                                ${photonSpacingCellNumber} \
                                                ${keV_perPhoton} \
                                                ${workspaceFolder} \
                                                ${moduleNumber} \
                                                ${batchJobCreationFolder}
done

