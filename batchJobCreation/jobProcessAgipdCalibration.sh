#!/usr/bin/env bash

workspaceFolder=/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_full/

batchJobCreationFolder=${AGIPDCALIBRATION_INSTALLATION_DIR}batchJobCreation/
processingFilesFolder=${AGIPDCALIBRATION_INSTALLATION_DIR}agipdCalibration/

keV_perPhoton=8.04
photonSpacingCellNumber=175

moduleNumbersToProcess=(1)

for moduleNumber in ${moduleNumbersToProcess[*]}
do
    darkDataFileName=/gpfs/cfel/fsds/labs/processed/calibration_1.1/M310_full_calibration/03_conventional_dark/m${moduleNumber}_dark_00000.nxs
    gatheredDarkDataFileName=${workspaceFolder}darkData_m${moduleNumber}.h5
    darkOffsetFileName=${workspaceFolder}darkOffset_m${moduleNumber}.h5

    xRayTubeDataFileName=/gpfs/cfel/fsds/labs/processed/calibration_1.1/M310_full_calibration/01_xray/m${moduleNumber}_xray_memcell175_00000.nxs
    gatheredXRayTubeDataFileName=${workspaceFolder}xRayTubeData_m${moduleNumber}.h5
    photonSpacingFileName=${workspaceFolder}photonSpacing_m${moduleNumber}.h5

    currentSourceScanFileName_column1and5=/gpfs/cfel/fsds/labs/processed/calibration_1.1/M310_full_calibration/04_drscs/m${moduleNumber}_15_00000.nxs
    currentSourceScanFileName_column2and6=/gpfs/cfel/fsds/labs/processed/calibration_1.1/M310_full_calibration/04_drscs/m${moduleNumber}_26_00000.nxs
    currentSourceScanFileName_column3and7=/gpfs/cfel/fsds/labs/processed/calibration_1.1/M310_full_calibration/04_drscs/m${moduleNumber}_37_00000.nxs
    currentSourceScanFileName_column4and8=/gpfs/cfel/fsds/labs/processed/calibration_1.1/M310_full_calibration/04_drscs/m${moduleNumber}_48_00000.nxs
    gatheredCurrentSourceScanFileName=${workspaceFolder}currentSourceScanData_m${moduleNumber}_chunked.h5
    analogGainsFileName=${workspaceFolder}analogGains_m${moduleNumber}.h5
    digitalMeansFileName=${workspaceFolder}digitalMeans_m${moduleNumber}.h5

    combinedCalibrationConstantsFileName=${workspaceFolder}combinedCalibrationConstants_m${moduleNumber}.h5

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

