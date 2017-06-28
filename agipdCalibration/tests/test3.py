import h5py
import sys
import time
import numpy as np
from scipy.signal.wavelets import cwt, ricker
import peakutils
from peakutils.plot import plot as pplot
from matplotlib import pyplot as plt
import pyqtgraph as pg

from agipdCalibration.algorithms.xRayTubeDataFitting import *

# fileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/python_saved_workspace/xRay175.h5'
fileName = '/gpfs/cfel/fsds/labs/processed/calibration_1.1/aschkan_stash/302-303-314-305/temperature_40C/xray/xRayTubeData_m3_Cu.h5'
f = h5py.File(fileName, 'r', libver='latest')
print('start loading analog from', fileName)
analog = f['analog'][()]
print('loading done')
f.close()

analog = analog[1:, ...]  # first value is always wrong

data = analog[:, 2, 5]

localityRadius = 800
samplePointsCount = 1000
(photonSpacing, quality, peakStdDevs) = getOnePhotonAdcCountsXRayTubeData(data, localityRadius, samplePointsCount)

i = 0
