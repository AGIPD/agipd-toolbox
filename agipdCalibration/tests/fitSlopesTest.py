import h5py

from agipdCalibration.algorithms.rangeScansFitting import *

dataFileName = '/gpfs/cfel/cxi/scratch/user/gevorkov/python_saved_workspace/pulsedCapacitor_m4_chunked.h5'
dataPathInFile = '/entry/instrument/detector/data'

interestingPixelsY = (120, 128)
interestingPixelsX = (20, 30)

f = h5py.File(dataFileName, 'r', libver='latest')
analog = f['/analog'][:, :, interestingPixelsY[0]:interestingPixelsY[1],
         interestingPixelsX[0]:interestingPixelsX[1]]
digital = f['/digital'][:, :, interestingPixelsY[0]:interestingPixelsY[1],
         interestingPixelsX[0]:interestingPixelsX[1]]
f.close()

(fitLineParameters, digitalMeanValues, analogFitError, (digitalStdDev_highGain, digitalStdDev_mediumGain)) = fit2DynamicScanSlopes(analog[:, 0, 1, 0], digital[:, 0, 1, 0])
