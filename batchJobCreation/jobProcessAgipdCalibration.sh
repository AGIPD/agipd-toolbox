#!/usr/bin/env bash

workspaceFolder=/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/

batchJobCreationFolder=${AGIPDCALIBRATION_INSTALLATION_DIR}batchJobCreation/
processingFilesFolder=${AGIPDCALIBRATION_INSTALLATION_DIR}agipdCalibration/



moduleNumbersToProcess=(1 2 3 4)

for moduleNumber in ${moduleNumbersToProcess[*]}
do
    xRayTubeDataFileName=/gpfs/cfel/fsds/labs/calibration/current/1Mpix_calib/wing2/xray/m${moduleNumber}_xray_00003.nxs
    gatheredXRayTubeDataFileName=${workspaceFolder}mokalphaData_m${moduleNumber}.h5
    photonSpacingFileName=${workspaceFolder}photonSpacing_m${moduleNumber}.h5

    currentSourceScanFileName=/gpfs/cfel/fsds/labs/processed/m${moduleNumber}_m233_drscsvr160_i80_00002.nxs
    gatheredCurrentSourceScanFileName=${workspaceFolder}pulsedCapacitorData_m${moduleNumber}_chunked.h5
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
                                                ${workspaceFolder} \
                                                ${moduleNumber} \
                                                ${batchJobCreationFolder}
done

