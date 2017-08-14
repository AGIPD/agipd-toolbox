import h5py
import numpy as np
import sys

from agipdCalibration.algorithms.rawDataEqualization import equalizeRawData_oneBurst

import matplotlib.pyplot as plt
import pyqtgraph as pg

dataFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/python_saved_workspace/gathered_MXX_m6_Mo_tube_burst_data_00001.h5'
combinedCalibrationConstantsFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/python_saved_workspace/combinedCalibrationConstants_m6_mixed.h5'
saveFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/python_saved_workspace/equalized_MXX_m6_Mo_tube_burst_data_00001.h5'

# dataFileName = sys.argv[1]
# combinedCalibrationConstantsFileName = sys.argv[2]
# saveFileName = sys.argv[3]

dataFile = h5py.File(dataFileName, 'r', libver='latest')
dset_analog = dataFile['/analog']
dset_digital = dataFile['/digital']

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

    dset_gainStage[i, ...] = gainStage
    dset_analogCorrected[i, ...] = analogCorrected

    print(i)

saveFile.flush()
saveFile.close()
combinedCalibrationConstantsFile.close()
dataFile.close()
