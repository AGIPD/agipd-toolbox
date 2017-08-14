import h5py
import numpy as np
from agipdCalibration.tests.h5py_display import h5disp

import matplotlib.pyplot as plt
import pyqtgraph as pg

from agipdCalibration.algorithms.rawDataEqualization import equalizeRawData_oneCell

dataFileName = '/gpfs/cfel/fsds/labs/processed/calibration_1.1/aschkan_stash/3_module_MC_imaging/raw/m6_MC_image_input.h5'
combinedCalibrationConstantsFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/python_saved_workspace/combinedCalibrationConstants_m6.h5'
maskFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/python_saved_workspace/mask_m6.h5'
saveFileName = '/gpfs/cfel/fsds/labs/processed/calibration_1.1/aschkan_stash/3_module_MC_imaging/m6_MC_image_equalized_2.h5'

# dataFileName = sys.argv[1]
# combinedCalibrationConstantsFileName = sys.argv[2]
# maskFileName = sys.argv[3]
# saveFileName = sys.argv[4]

dataFile = h5py.File(dataFileName, 'r', libver='latest')
dset_analog = dataFile['/analog']
dset_digital = dataFile['/digital']

maskFile = h5py.File(maskFileName, 'r')
badCellMask = maskFile['/badCellMask'][...]
maskFile.close()

combinedCalibrationConstantsFile = h5py.File(combinedCalibrationConstantsFileName, 'r', libver='latest')
analogGains_keV = combinedCalibrationConstantsFile['/analogGains_keV'][...]
digitalThresholds = combinedCalibrationConstantsFile['/digitalThresholds'][...]
darkOffsets = combinedCalibrationConstantsFile['/darkOffsets'][...]
#darkOffsets[2, ...] = darkOffsets[0, ...]
#darkOffsets[1, ...] = darkOffsets[0, ...]

saveFile = h5py.File(saveFileName, "w", libver='latest')
dset_analogCorrected = saveFile.create_dataset("analogCorrected", shape=dset_analog.shape, chunks=(1, 128, 512), dtype='float32')
dset_gainStage = saveFile.create_dataset("digitalGainStage", shape=dset_digital.shape, chunks=(1, 128, 512), dtype='int8')
dset_mean = saveFile.create_dataset("mean", shape=(128, 512), dtype='int8')
dset_median = saveFile.create_dataset("median", shape=(128, 512), dtype='int8')

cellNumber = 175

for i in np.arange(dset_analog.shape[0]):
    analog = dataFile['/analog'][i, ...].astype('float32')
    digital = dataFile['/digital'][i, ...]

    (analogCorrected, gainStage) = equalizeRawData_oneCell(analog, digital, analogGains_keV, digitalThresholds, darkOffsets, cellNumber)

    analogCorrected[badCellMask[cellNumber, ...]] = 0
    gainStage[badCellMask[cellNumber, ...]] = -1

    dset_gainStage[i, ...] = gainStage
    dset_analogCorrected[i, ...] = analogCorrected

    print(i)

tmp = dset_analogCorrected[...]
a = np.median(dset_analogCorrected, axis=0)
dset_mean[...] = np.mean(tmp, axis=0)
dset_median[...] = np.median(tmp, axis=0)

saveFile.flush()
saveFile.close()
combinedCalibrationConstantsFile.close()
dataFile.close()
