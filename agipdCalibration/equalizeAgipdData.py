import h5py
import numpy as np
import sys

import matplotlib.pyplot as plt
import pyqtgraph as pg

dataFileName = '/gpfs/cfel/cxi/scratch/user/gevorkov/python_saved_workspace/lysozymeData_m' + sys.argv[1] + '_correctionChunked.h5'
combinedCalibrationConstantsFileName = '/gpfs/cfel/cxi/scratch/user/gevorkov/agipdCalibration_workspace/combinedCalibrationConstants_m' + sys.argv[1] + '.h5'
saveFileName = '/gpfs/cfel/cxi/scratch/user/gevorkov/python_saved_workspace/lysozymeData_m' + sys.argv[1] + '_equalized.h5'

dataFile = h5py.File(dataFileName, 'r', libver='latest')
dset_analog = dataFile['/analog']
dset_digital = dataFile['/digital']

saveFile = h5py.File(saveFileName, "w", libver='latest')
dset_analogCorrected = saveFile.create_dataset("analogCorrected", shape=dset_analog.shape, chunks=(1, 352, 128, 512), dtype='float32')
dset_gainStage = saveFile.create_dataset("digitalGainStage", shape=dset_digital.shape, chunks=(1, 352, 128, 512), dtype='int8')

combinedCalibrationConstantsFile = h5py.File(combinedCalibrationConstantsFileName, 'r', libver='latest')
analogGains_keV = combinedCalibrationConstantsFile['/analogGains_keV']
digitalThresholds = combinedCalibrationConstantsFile['/digitalThresholds']
darkOffset = combinedCalibrationConstantsFile['/darkOffset']

for i in np.arange(dset_analog.shape[0]):
    analog = dataFile['/analog'][i, ...].astype('float32')
    digital = dataFile['/digital'][i, ...]

    gainStage = np.zeros((352, 128, 512), dtype='uint8')
    gainStage[digital > digitalThresholds[0, ...]] = 1
    # gainStage[digital > digitalThresholds[1, ...]] = 2

    # analogCorrected = (analog - darkOffset) * analogGains_keV[gainStage]
    analogCorrected = (analog - darkOffset)
    analogCorrected.ravel()[gainStage.ravel() == 0] *= analogGains_keV[0, ...].ravel()[gainStage.ravel() == 0]
    analogCorrected.ravel()[gainStage.ravel() == 1] *= analogGains_keV[1, ...].ravel()[gainStage.ravel() == 1]
    analogCorrected.ravel()[gainStage.ravel() == 2] *= analogGains_keV[2, ...].ravel()[gainStage.ravel() == 2]
    # analogCorrected[gainStage == 0] *= analogGains_keV[0, gainStage == 0]
    # analogCorrected[gainStage == 1] *= analogGains_keV[1, gainStage == 1]
    # analogCorrected[gainStage == 2] *= analogGains_keV[2, gainStage == 2]

    dset_gainStage[i, ...] = gainStage
    dset_analogCorrected[i, ...] = analogCorrected

    print(i)

saveFile.flush()
saveFile.close()
combinedCalibrationConstantsFile.close()
dataFile.close()
