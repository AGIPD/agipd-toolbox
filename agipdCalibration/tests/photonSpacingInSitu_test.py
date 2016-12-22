from multiprocessing import Pool
import h5py
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

from agipdCalibration.algorithms.photonSpacingWorkaroundInSitu import *

dataFileName = '/gpfs/cfel/cxi/scratch/user/gevorkov/python_saved_workspace/lysozymeData_m3.h5'

print('loading data')
t = time.time()
f = h5py.File(dataFileName, 'r', libver='latest')
analog = f['/analog'][:, :, 0:64, 0:64]
f.close()
print('took time:  ' + str(time.time() - t))

data = analog[:,0,55,5]
adcCount = getOnePhotonAdcCountsInSitu(data)

localBinEdges = np.arange(np.min(data), np.max(data))
(localHistogram, _) = np.histogram(data, localBinEdges)

plt.plot(localBinEdges[0:-1], localHistogram)
