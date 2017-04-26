import h5py
import numpy as np
import sys

import matplotlib.pyplot as plt
import pyqtgraph as pg

# moduleNumber = 4
# analogGainsFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/analogGains_m' + str(moduleNumber) + '.h5'
# digitalMeansFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/digitalMeans_m' + str(moduleNumber) + '.h5'
# darkOffsetFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/lysozymeData_m' + str(moduleNumber) + '_darkcal_inSitu.h5'
# photonSpacingFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/photonSpacing_m' + str(moduleNumber) + '_inSitu.h5'
# photonSpacingCellNumber = 175
# keV_perPhoton = 1
# saveFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/combinedCalibrationConstants_m' + str(moduleNumber) + '.h5'

analogGainsFileName = sys.argv[1]
digitalMeansFileName = sys.argv[2]
darkOffsetFileName = sys.argv[3]
photonSpacingFileName = sys.argv[4]
photonSpacingCellNumber = int(float(sys.argv[5]))
keV_perPhoton = float(sys.argv[6])
saveFileName = sys.argv[7]

print('\n\n\nstart gatherCurrentSourceScanData')
print('analogGainsFileName = ', analogGainsFileName)
print('digitalMeansFileName = ', digitalMeansFileName)
print('darkOffsetFileName = ', darkOffsetFileName)
print('photonSpacingFileName = ', photonSpacingFileName)
print('photonSpacingCellNumber = ', photonSpacingCellNumber)
print('keV_perPhoton = ', keV_perPhoton)
print('saveFileName = ', saveFileName)
print('')

analogGains = h5py.File(analogGainsFileName, 'r', libver='latest')['/analogGains'][...]
anlogLineOffsets = h5py.File(analogGainsFileName, 'r', libver='latest')['/anlogLineOffsets'][...]
digitalThresholds = h5py.File(digitalMeansFileName, 'r', libver='latest')['/digitalThresholds'][...]
darkOffset = h5py.File(darkOffsetFileName, 'r', libver='latest')['/darkOffset'][...]
photonSpacing = h5py.File(photonSpacingFileName, 'r', libver='latest')['/photonSpacing'][...]

saveFile = h5py.File(saveFileName, "w", libver='latest')
dset_analogGains_keV = saveFile.create_dataset("analogGains_keV", shape=(3, 352, 128, 512), dtype='float32')
dset_digitalThresholds = saveFile.create_dataset("digitalThresholds", shape=(2, 352, 128, 512), dtype='int16')
dset_darkOffsets = saveFile.create_dataset("darkOffsets", shape=(3, 352, 128, 512), dtype='int16')

analogGains_keV = np.zeros(analogGains.shape, dtype='float32')
photonSpacingAllCells = photonSpacing.astype('float32') / analogGains[0, photonSpacingCellNumber, ...] * analogGains[0, :, ...]
analogGains_keV[0, :, :, :] = keV_perPhoton / photonSpacingAllCells
analogGains_keV[1, :, :, :] = keV_perPhoton / photonSpacingAllCells / analogGains[0, ...] * analogGains[1, ...]
analogGains_keV[2, :, :, :] = keV_perPhoton / photonSpacingAllCells / analogGains[0, ...] * analogGains[2, ...]
analogGains_keV[~np.isfinite(analogGains_keV)] = -1  # just for easier plotting, should be masked anyway!

dset_analogGains_keV[...] = analogGains_keV
dset_digitalThresholds[...] = digitalThresholds
dset_darkOffsets[0, ...] = darkOffset
dset_darkOffsets[1, ...] = np.round(anlogLineOffsets[1,...]).astype('int16')
dset_darkOffsets[2, ...] = np.round(anlogLineOffsets[2,...]).astype('int16')

saveFile.close()
