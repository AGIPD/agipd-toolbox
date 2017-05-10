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

    currentSourceScanFileName_column1and5=/gpfs/cfel/fsds/labs/processed/calibration_1.1/drscs_step/m${moduleNumber}_cdslow_col1and5_00000.nxs
    currentSourceScanFileName_column2and6=/gpfs/cfel/fsds/labs/processed/calibration_1.1/drscs_step/m${moduleNumber}_cdslow_col2and6_00000.nxs
    currentSourceScanFileName_column3and7=/gpfs/cfel/fsds/labs/processed/calibration_1.1/drscs_step/m${moduleNumber}_cdslow_col3and7_00000.nxs
    currentSourceScanFileName_column4and8=/gpfs/cfel/fsds/labs/processed/calibration_1.1/drscs_step/m${moduleNumber}_cdslow_col4and8_00000.nxs
    gatheredCurrentSourceScanFileName=${workspaceFolder}currentSourceScanData_m${moduleNumber}_chunked.h5
    analogGainsFileName=${workspaceFolder}analogGains_m${moduleNumber}.h5
    digitalMeansFileName=${workspaceFolder}digitalMeans_m${moduleNumber}.h5

    combinedCalibrationConstantsFileName=combinedCalibrationConstants_m${moduleNumber}.h5

    ${batchJobCreationFolder}processAgipdCalibration_oneModule.sh ${xRayTubeDataFileName} \
                                                ${gatheredXRayTubeDataFileName} \
                                                ${photonSpacingFileName} \
                                                ${currentSourceScanFileName_column1and5} \
                                                ${currentSourceScanFileName_column2and6} \
                                                ${currentSourceScanFileName_column3and7} \
                                                ${currentSourceScanFileName_column4and8} \
                                                ${gatheredCurrentSourceScanFileName} \
                                                ${analogGainsFileName} \
                                                ${digitalMeansFileName} \
                                                ${processingFilesFolder} \
                                                ${darkDataFileName} \
                                                ${gatheredDarkDataFileName} \
                                                ${darkOffsetFileName} \
                                                ${photonSpacingCellNumber} \
                                                ${keV_perPhoton} \
                                                ${combinedCalibrationConstantsFileName} \
                                                ${workspaceFolder} \
                                                ${moduleNumber} \
                                                ${batchJobCreationFolder}
done

