import h5py
import numpy as np
import sys

import matplotlib.pyplot as plt
import pyqtgraph as pg

dataFileName = '/gpfs/cfel/cxi/scratch/user/gevorkov/python_saved_workspace/lysozymeData_m' + sys.argv[1] + '_equalized.h5'
maskFileName = '/gpfs/cfel/cxi/scratch/user/gevorkov/agipdCalibration_workspace/mask_m' + sys.argv[1] + '.h5'
saveFileName_filtered = '/gpfs/cfel/cxi/scratch/user/gevorkov/python_saved_workspace/lysozymeData_m' + sys.argv[1] + '_averageFiltered.h5'
saveFileName_badPixelMask = '/gpfs/cfel/cxi/scratch/user/gevorkov/python_saved_workspace/lysozymeData_m' + sys.argv[1] + '_badPixelMask.h5'

minValidDataCount = 30

dataFile = h5py.File(dataFileName, 'r')
dset_analogCorrected = dataFile['/analogCorrected']
dset_digitalGainStage = dataFile['/digitalGainStage']
dataCount = dset_analogCorrected.shape[0]

badCellMask = h5py.File(maskFileName, 'r')['/badCellMask'][...]
badCellMask = np.transpose(badCellMask, (1, 2, 0))

saveFile_filtered = h5py.File(saveFileName_filtered, "w", libver='latest')
dset_medianFiltered = saveFile_filtered.create_dataset("medianFiltered", shape=(dataCount, 128, 512), dtype='float32')
dset_meanFiltered = saveFile_filtered.create_dataset("meanFiltered", shape=(dataCount, 128, 512), dtype='float32')

saveFile_badPixelMask = h5py.File(saveFileName_badPixelMask, "w", libver='latest')
dset_badPixelMask = saveFile_badPixelMask.create_dataset("badPixelMask", shape=(128, 512), dtype='float32')

badPixelMask = np.zeros((128, 512), dtype=bool)

for i in np.arange(dset_analogCorrected.shape[0]):
    analogCorrected = dset_analogCorrected[i, ...]
    analogCorrected = np.transpose(analogCorrected, (1, 2, 0))

    digitalGainStage = dset_digitalGainStage[i, ...]
    digitalGainStage = np.transpose(digitalGainStage, (1, 2, 0))

    medianFiltered = np.zeros((128, 512))
    meanFiltered = np.zeros((128, 512))

    for x in np.arange(512):
        for y in np.arange(128):
            validData = analogCorrected[y, x, np.logical_and(~badCellMask[y, x, :], digitalGainStage[y, x, :] == 0)]
            if validData.size < minValidDataCount:
                badPixelMask[y, x] = 0
            else:
                medianFiltered[y, x] = np.median(validData)
                meanFiltered[y, x] = np.mean(validData)

    dset_medianFiltered[i, ...] = medianFiltered
    dset_meanFiltered[i, ...] = meanFiltered

    print(i)

dataFile.close()
saveFile_filtered.flush()
saveFile_filtered.close()
saveFile_badPixelMask.flush()
saveFile_badPixelMask.close()
