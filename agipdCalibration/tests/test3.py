import h5py
import sys
import time
import numpy as np
from scipy.signal.wavelets import cwt, ricker
import peakutils
from peakutils.plot import plot as pplot
from matplotlib import pyplot as plt
import pyqtgraph as pg



fileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/xRay175.h5'
f = h5py.File(fileName, 'r', libver='latest')
print('start loading analog from', fileName)
analog = f['analog'][()]
print('loading done')
f.close()

analog = analog[1:, ...]  # first value is always wrong

data = analog[:, 30, 35]
photonHistogramBins = np.arange(np.min(data), np.max(data), 1, dtype='int16')
(photonHistoramValues, _) = np.histogram(data, photonHistogramBins)
x = photonHistogramBins[1:]
y = photonHistoramValues

plt.plot(x, y)
plt.title("Data with noise")

indexes = peakutils.indexes(y, thres=0.05, min_dist=50)
print(indexes)
print(x[indexes], y[indexes])
plt.figure(figsize=(10, 6))
pplot(x, y, indexes)
plt.title('First estimate')

peaks_x = peakutils.interpolate(x, y, ind=indexes, width=15)
print(peaks_x)

base = peakutils.baseline(y, 10)
plt.figure(figsize=(10,6))
plt.plot(x, base)
plt.title("Data with baseline removed")

i = 0
