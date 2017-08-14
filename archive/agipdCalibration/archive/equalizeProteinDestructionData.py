import h5py
import numpy as np
import sys
from agipdCalibration.tests.h5py_display import h5disp
import os

import matplotlib.pyplot as plt
import pyqtgraph as pg

dataPathInFile = '/entry/instrument/detector/data'
dataFolder = '/asap3/petra3/gpfs/p11/2016/data/11001485/raw/ProteinDestruction/'
allFiles = os.listdir(dataFolder)

for i in np.arange(len(allFiles)):
    print(i / len(allFiles))

    dataFileName = dataFolder + allFiles[i]
    moduleNumber = allFiles[i][1]

    combinedCalibrationConstantsFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/combinedCalibrationConstants_m' + moduleNumber + '.h5'
    maskFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/mask_m' + moduleNumber + '.h5'
    saveFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/correctedProteinDestructionData/' + allFiles[i] + '_equalized.h5'

    maskFile = h5py.File(maskFileName, 'r')
    badCellMask = maskFile['/badCellMask'][...]
    maskFile.close()

    dataFile = h5py.File(dataFileName, 'r')
    rawData = dataFile[dataPathInFile][...]
    dataFile.close()

    if rawData.size == 0:
        continue

    analog = rawData[::2, ...].astype('float32')
    digital = rawData[1::2, ...]

    analog.shape = (352, 128, 512)
    digital.shape = (352, 128, 512)

    # a = pg.image(analog.transpose(0, 2, 1))
    # i = 0

    saveFile = h5py.File(saveFileName, "w", libver='latest')
    dset_analogCorrected = saveFile.create_dataset("analogCorrected", shape=(352, 128, 512), dtype='float32')
    dset_gainStage = saveFile.create_dataset("digitalGainStage", shape=(352, 128, 512), dtype='int8')

    combinedCalibrationConstantsFile = h5py.File(combinedCalibrationConstantsFileName, 'r', libver='latest')
    analogGains_keV = combinedCalibrationConstantsFile['/analogGains_keV']
    digitalThresholds = combinedCalibrationConstantsFile['/digitalThresholds']
    darkOffset = combinedCalibrationConstantsFile['/darkOffset']

    gainStage = np.zeros((352, 128, 512), dtype='uint8')
    gainStage[digital > digitalThresholds[0, ...]] = 1
    # gainStage[digital > digitalThresholds[1, ...]] = 2

    # following code mimics "analogCorrected = (analog - darkOffset) * analogGains_keV[gainStage]"
    analogCorrected = (analog - darkOffset)
    analogCorrected.ravel()[gainStage.ravel() == 0] *= analogGains_keV[0, ...].ravel()[gainStage.ravel() == 0]
    analogCorrected.ravel()[gainStage.ravel() == 1] *= analogGains_keV[1, ...].ravel()[gainStage.ravel() == 1]
    analogCorrected.ravel()[gainStage.ravel() == 2] *= analogGains_keV[2, ...].ravel()[gainStage.ravel() == 2]
    # analogCorrected[gainStage == 0] *= analogGains_keV[0, gainStage == 0]
    # analogCorrected[gainStage == 1] *= analogGains_keV[1, gainStage == 1]
    # analogCorrected[gainStage == 2] *= analogGains_keV[2, gainStage == 2]

    analogCorrected[badCellMask] = 0
    gainStage[badCellMask] = 0

    analogCorrected[gainStage > 0] = 0

    dset_gainStage[...] = gainStage
    dset_analogCorrected[...] = analogCorrected

    saveFile.flush()
    saveFile.close()
    combinedCalibrationConstantsFile.close()

    # b = pg.image(analogCorrected.transpose(0, 2, 1))
    # c = pg.image(gainStage.transpose(0, 2, 1))
    # i = 0
