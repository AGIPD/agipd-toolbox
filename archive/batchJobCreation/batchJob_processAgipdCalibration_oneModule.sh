#!/usr/bin/env bash

#SBATCH --partition=all
#SBATCH --time=48:00:00                 
#SBATCH --nodes=1
#SBATCH --cores-per-socket=16
#SBATCH --threads-per-core=2
#SBATCH --sockets-per-node=2
#SBATCH --mail-type=END
#SBATCH --mail-user=jennifer.poehlsen@desy.de

source /etc/profile.d/modules.sh
source /gpfs/cfel/cxi/common/cfelsoft-rh7/setup.sh
module load cfel-anaconda/py3-4.3.0

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

darkDataFileName=${11}
gatheredDarkDataFileName=${12}
darkOffsetFileName=${13}

photonSpacingCellNumber=${14}
keV_perPhoton=${15}

combinedCalibrationConstantsFileName=${16}

processingFilesFolder=${17}

nPartsCS=${18}
nPartsDark=${19}

python ${processingFilesFolder}batchProcessing/gatherDarkData.py ${nPartsDark} ${darkDataFileName} ${gatheredDarkDataFileName}

python ${processingFilesFolder}batchProcessing/batchProcessDarkData.py ${gatheredDarkDataFileName} ${darkOffsetFileName}

python ${processingFilesFolder}batchProcessing/gatherXRayTubeData.py ${xRayTubeDataFileName} ${gatheredXRayTubeDataFileName}

python ${processingFilesFolder}batchProcessing/batchProcessXRayTubeData.py ${gatheredXRayTubeDataFileName} ${photonSpacingFileName}

python ${processingFilesFolder}batchProcessing/gatherCurrentSourceScanData.py ${nPartsCS} ${currentSourceScanFileName_column1and5} ${currentSourceScanFileName_column2and6} ${currentSourceScanFileName_column3and7} ${currentSourceScanFileName_column4and8} ${gatheredCurrentSourceScanFileName}

python ${processingFilesFolder}batchProcessing/batchProcessCurrentSourceScan.py ${gatheredCurrentSourceScanFileName} ${analogGainsFileName} ${digitalMeansFileName}

python ${processingFilesFolder}combineCalibrationData.py ${analogGainsFileName} ${digitalMeansFileName} ${darkOffsetFileName} ${photonSpacingFileName} ${photonSpacingCellNumber} ${keV_perPhoton} ${combinedCalibrationConstantsFileName}



