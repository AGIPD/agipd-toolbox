import h5py
import numpy as np

import matplotlib.pyplot as plt
import pyqtgraph as pg


analogGainsFileName = '/gpfs/cfel/cxi/scratch/user/gevorkov/agipdCalibration_workspace/analogGains_m3.h5'
digitalMeansFileName = '/gpfs/cfel/cxi/scratch/user/gevorkov/agipdCalibration_workspace/digitalMeans_m3.h5'
darkOffsetFileName = '/gpfs/cfel/cxi/scratch/user/gevorkov/agipdCalibration_workspace/lysozymeData_m3_darkcal_inSitu.h5'
photonSpacingFileName = '/gpfs/cfel/cxi/scratch/user/gevorkov/agipdCalibration_workspace/photonSpacing_m3_inSitu.h5'
photonSpacingCellNumber = 175
keV_perPhoton = 1

saveFileName = '/gpfs/cfel/cxi/scratch/user/gevorkov/agipdCalibration_workspace/combinedCalibrationConstants_m4.h5'

analogGains = h5py.File(analogGainsFileName, 'r', libver='latest')['/analogGains'][...]
digitalThresholds = h5py.File(digitalMeansFileName, 'r', libver='latest')['/digitalThresholds'][...]
darkOffset = h5py.File(darkOffsetFileName, 'r', libver='latest')['/darkOffset'][...]
photonSpacing = h5py.File(photonSpacingFileName, 'r', libver='latest')['/photonSpacing'][...]

saveFile = h5py.File(saveFileName, "w", libver='latest')
dset_analogGains_keV = saveFile.create_dataset("analogGains_keV", shape=(3, 352, 128, 512), dtype='float32')
dset_digitalThresholds = saveFile.create_dataset("digitalThresholds", shape=(2, 352, 128, 512), dtype='int16')
dset_darkOffset = saveFile.create_dataset("darkOffset", shape=(352, 128, 512), dtype='int16')

analogGains_keV = np.zeros(analogGains.shape, dtype='float32')
photonSpacingAllCells = photonSpacing.astype('float32') / analogGains[0, photonSpacingCellNumber, ...] * analogGains[0, :, ...]
analogGains_keV[0, :, :, :] = 1/photonSpacingAllCells
analogGains_keV[1, :, :, :] = photonSpacingAllCells / analogGains[0, ...] * analogGains[1, ...]
analogGains_keV[2, :, :, :] = photonSpacingAllCells / analogGains[0, ...] * analogGains[2, ...]
analogGains_keV[~np.isfinite(analogGains_keV)] = -1  # just for easier plotting, should be masked anyway!

dset_analogGains_keV[...] = analogGains_keV
dset_digitalThresholds[...] = digitalThresholds
dset_darkOffset[...] = darkOffset

saveFile.close()
