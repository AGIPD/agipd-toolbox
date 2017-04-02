import h5py
import numpy as np
import sys

from agipdCalibration.algorithms.rawDataEqualization import equalizeRawData_oneBurst

import matplotlib.pyplot as plt
import pyqtgraph as pg

moduleNumber = sys.argv[1]
dataFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/python_saved_workspace/lysozymeData_m' + moduleNumber + '_correctionChunked.h5'
combinedCalibrationConstantsFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/combinedCalibrationConstants_m' + moduleNumber + '.h5'
maskFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/mask_m' + moduleNumber + '.h5'
saveFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/python_saved_workspace/lysozymeData_m' + moduleNumber + '_equalized.h5'

dataFile = h5py.File(dataFileName, 'r', libver='latest')
dset_analog = dataFile['/analog']
dset_digital = dataFile['/digital']

maskFile = h5py.File(maskFileName, 'r')
badCellMask = maskFile['/badCellMask'][...]
maskFile.close()

combinedCalibrationConstantsFile = h5py.File(combinedCalibrationConstantsFileName, 'r', libver='latest')
analogGains_keV = combinedCalibrationConstantsFile['/analogGains_keV']
digitalThresholds = combinedCalibrationConstantsFile['/digitalThresholds']
darkOffsets = combinedCalibrationConstantsFile['/darkOffsets']

saveFile = h5py.File(saveFileName, "w", libver='latest')
dset_analogCorrected = saveFile.create_dataset("analogCorrected", shape=dset_analog.shape, chunks=(1, 352, 128, 512), dtype='float32')
dset_gainStage = saveFile.create_dataset("digitalGainStage", shape=dset_digital.shape, chunks=(1, 352, 128, 512), dtype='int8')

for i in np.arange(dset_analog.shape[0]):
    analog = dataFile['/analog'][i, ...].astype('float32')
    digital = dataFile['/digital'][i, ...]

    (analogCorrected, gainStage) = equalizeRawData_oneBurst(analog, digital, analogGains_keV, digitalThresholds, darkOffsets)

    analogCorrected[badCellMask] = 0
    gainStage[badCellMask] = -1

    dset_gainStage[i, ...] = gainStage
    dset_analogCorrected[i, ...] = analogCorrected

    print(i)

saveFile.flush()
saveFile.close()
combinedCalibrationConstantsFile.close()
dataFile.close()
