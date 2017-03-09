#!/usr/bin/env bash

workspaceFolder=/gpfs/cfel/cxi/scratch/user/gevorkov/agipdCalibration_workspace/

batchJobCreationFolder=${AGIPDCALIBRATION_INSTALLATION_DIR}batchJobCreation/
processingFilesFolder=${AGIPDCALIBRATION_INSTALLATION_DIR}agipdCalibration/



moduleNumber=3
workaroundModuleNumber=4 #calibration data mixed up, need workaround for this beamtime

mokalphaDataFileName=/gpfs/cfel/fsds/labs/calibration/current/1Mpix_calib/wing2/xray/m${workaroundModuleNumber}_xray_00003.nxs
gatheredMokalphaFileName=${workspaceFolder}mokalphaData_m${moduleNumber}.h5
photonSpacingFileName=${workspaceFolder}photonSpacing_m${moduleNumber}.h5

pulsedCapacitorDataFolder=/gpfs/cfel/fsds/labs/calibration/current/1Mpix_calib/wing2/drspc/
pulsedCapacitorModuleNumber=${workaroundModuleNumber}
gatheredPulsedCapacitorDataSaveFileName=${workspaceFolder}pulsedCapacitorData_m${moduleNumber}_chunked.h5
analogGainsSaveFileName=${workspaceFolder}analogGains_m${moduleNumber}.h5
digitalMeansSaveFileName=${workspaceFolder}digitalMeans_m${moduleNumber}.h5

./processAgipdCalibration_oneModule.sh ${mokalphaDataFileName} \
											${gatheredMokalphaFileName} \
											${photonSpacingFileName} \
											${pulsedCapacitorDataFolder} \
											${pulsedCapacitorModuleNumber} \
											${gatheredPulsedCapacitorDataSaveFileName} \
											${analogGainsSaveFileName} \
											${digitalMeansSaveFileName} \
											${processingFilesFolder} \
											${workspaceFolder} \
											${moduleNumber} \
											${batchJobCreationFolder}



moduleNumber=4
workaroundModuleNumber=3 #calibration data mixed up, need workaround for this beamtime

mokalphaDataFileName=/gpfs/cfel/fsds/labs/calibration/current/1Mpix_calib/wing2/xray/m${workaroundModuleNumber}_xray_00003.nxs
gatheredMokalphaFileName=${workspaceFolder}mokalphaData_m${moduleNumber}.h5
photonSpacingFileName=${workspaceFolder}photonSpacing_m${moduleNumber}.h5

pulsedCapacitorDataFolder=/gpfs/cfel/fsds/labs/calibration/current/1Mpix_calib/wing2/drspc/
pulsedCapacitorModuleNumber=${workaroundModuleNumber}
gatheredPulsedCapacitorDataSaveFileName=${workspaceFolder}pulsedCapacitorData_m${moduleNumber}_chunked.h5
analogGainsSaveFileName=${workspaceFolder}analogGains_m${moduleNumber}.h5
digitalMeansSaveFileName=${workspaceFolder}digitalMeans_m${moduleNumber}.h5

./processAgipdCalibration_oneModule.sh ${mokalphaDataFileName} \
											${gatheredMokalphaFileName} \
											${photonSpacingFileName} \
											${pulsedCapacitorDataFolder} \
											${pulsedCapacitorModuleNumber} \
											${gatheredPulsedCapacitorDataSaveFileName} \
											${analogGainsSaveFileName} \
											${digitalMeansSaveFileName} \
											${processingFilesFolder} \
											${workspaceFolder} \
											${moduleNumber} \
											${batchJobCreationFolder}



moduleNumber=1
workaroundModuleNumber=2 #calibration data mixed up, need workaround for this beamtime

mokalphaDataFileName=/gpfs/cfel/fsds/labs/calibration/current/1Mpix_calib/wing2/xray/m${workaroundModuleNumber}_xray_00003.nxs
gatheredMokalphaFileName=${workspaceFolder}mokalphaData_m${moduleNumber}.h5
photonSpacingFileName=${workspaceFolder}photonSpacing_m${moduleNumber}.h5

pulsedCapacitorDataFolder=/gpfs/cfel/fsds/labs/calibration/current/1Mpix_calib/wing2/drspc/
pulsedCapacitorModuleNumber=${workaroundModuleNumber}
gatheredPulsedCapacitorDataSaveFileName=${workspaceFolder}pulsedCapacitorData_m${moduleNumber}_chunked.h5
analogGainsSaveFileName=${workspaceFolder}analogGains_m${moduleNumber}.h5
digitalMeansSaveFileName=${workspaceFolder}digitalMeans_m${moduleNumber}.h5

./processAgipdCalibration_oneModule.sh ${mokalphaDataFileName} \
											${gatheredMokalphaFileName} \
											${photonSpacingFileName} \
											${pulsedCapacitorDataFolder} \
											${pulsedCapacitorModuleNumber} \
											${gatheredPulsedCapacitorDataSaveFileName} \
											${analogGainsSaveFileName} \
											${digitalMeansSaveFileName} \
											${processingFilesFolder} \
											${workspaceFolder} \
											${moduleNumber} \
											${batchJobCreationFolder}



moduleNumber=2
workaroundModuleNumber=1 #calibration data mixed up, need workaround for this beamtime

mokalphaDataFileName=/gpfs/cfel/fsds/labs/calibration/current/1Mpix_calib/wing2/xray/m${workaroundModuleNumber}_xray_00003.nxs
gatheredMokalphaFileName=${workspaceFolder}mokalphaData_m${moduleNumber}.h5
photonSpacingFileName=${workspaceFolder}photonSpacing_m${moduleNumber}.h5

pulsedCapacitorDataFolder=/gpfs/cfel/fsds/labs/calibration/current/1Mpix_calib/wing2/drspc/
pulsedCapacitorModuleNumber=${workaroundModuleNumber}
gatheredPulsedCapacitorDataSaveFileName=${workspaceFolder}pulsedCapacitorData_m${moduleNumber}_chunked.h5
analogGainsSaveFileName=${workspaceFolder}analogGains_m${moduleNumber}.h5
digitalMeansSaveFileName=${workspaceFolder}digitalMeans_m${moduleNumber}.h5

./processAgipdCalibration_oneModule.sh ${mokalphaDataFileName} \
											${gatheredMokalphaFileName} \
											${photonSpacingFileName} \
											${pulsedCapacitorDataFolder} \
											${pulsedCapacitorModuleNumber} \
											${gatheredPulsedCapacitorDataSaveFileName} \
											${analogGainsSaveFileName} \
											${digitalMeansSaveFileName} \
											${processingFilesFolder} \
											${workspaceFolder} \
											${moduleNumber} \
											${batchJobCreationFolder}