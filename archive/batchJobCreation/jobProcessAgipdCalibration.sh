#!/usr/bin/env bash


batchJobCreationFolder=${AGIPDCALIBRATION_INSTALLATION_DIR}batchJobCreation/
processingFilesFolder=${AGIPDCALIBRATION_INSTALLATION_DIR}agipdCalibration/

moduleID=M304
moduleNumbersToProcess=(2)
temperature=30C
itestc=itestc80

dataFolder=/gpfs/cfel/fsds/labs/agipd/calibration/raw/315-304-309-314-316-306-307/temperature_${temperature}
processedFolder=/gpfs/cfel/fsds/labs/agipd/calibration/processed/${moduleID}/temperature_${temperature}

keV_perPhoton=8.04
element=Mo
photonSpacingCellNumber=175

nPartsDark=10
nPartsCS=13


for moduleNumber in ${moduleNumbersToProcess[*]}
do
    darkDataFileName=${dataFolder}/dark/${moduleID}_m${moduleNumber}_dark_tint150ns_00012_part000
    gatheredDarkDataFileName=${processedFolder}/dark/darkData_${moduleID}_m${moduleNumber}_tint150ns.h5
    darkOffsetFileName=${processedFolder}/dark/darkOffset_${moduleID}_m${moduleNumber}_tint150ns.h5

    xRayTubeDataFileName=${dataFolder}/xray/${moduleID}_m${moduleNumber}_xray_${element}_00008.nxs
    gatheredXRayTubeDataFileName=${processedFolder}/xray/xRayTubeData_${moduleID}_m${moduleNumber}_xray_${element}.h5
    photonSpacingFileName=${processedFolder}/xray/photonSpacing_${moduleID}_m${moduleNumber}_xray_${element}.h5

    currentSourceScanFileName_column1and5=${dataFolder}/drscs/${itestc}/${moduleID}_m${moduleNumber}_drscs_${itestc}_col15_00009_part000
    currentSourceScanFileName_column2and6=${dataFolder}/drscs/${itestc}/${moduleID}_m${moduleNumber}_drscs_${itestc}_col26_00010_part000
    currentSourceScanFileName_column3and7=${dataFolder}/drscs/${itestc}/${moduleID}_m${moduleNumber}_drscs_${itestc}_col37_00011_part000
    currentSourceScanFileName_column4and8=${dataFolder}/drscs/${itestc}/${moduleID}_m${moduleNumber}_drscs_${itestc}_col48_00012_part000
    gatheredCurrentSourceScanFileName=${processedFolder}/drscs/${itestc}/currentSourceScanData_${moduleID}_m${moduleNumber}_chunked.h5
    analogGainsFileName=${processedFolder}/drscs/${itestc}/analogGains_${moduleID}_m${moduleNumber}.h5
    digitalMeansFileName=${processedFolder}/drscs/${itestc}/digitalMeans_${moduleID}_m${moduleNumber}.h5

    combinedCalibrationConstantsFileName=${processedFolder}/combinedCalibrationConstants_${moduleID}_m${moduleNumber}.h5

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
                                                ${processedFolder} \
                                                ${moduleNumber} \
                                                ${batchJobCreationFolder} \
                                                ${nPartsCS} \
						${nPartsDark} \
	                                        ${moduleID} \
	                                        ${temperature}
done
