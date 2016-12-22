import h5py
import numpy as np

import matplotlib.pyplot as plt
import pyqtgraph as pg

# dataFileName = '/gpfs/cfel/cxi/scratch/user/gevorkov/python_saved_workspace/lysozymeData_m3_equalized.h5'
#
# dataFile = h5py.File(dataFileName, 'r', libver='latest')
# dset_analogCorrected = dataFile['/analogCorrected']
# dset_digitalGainStage = dataFile['/digitalGainStage']
#
# # print(dset_analogCorrected.shape)
#
# # pg.image(dset_analogCorrected[0,0,...].T)
# # pg.image(dset_digitalGainStage[0,0,...])
# tmp = dset_digitalGainStage[...]
# # plt.hist(tmp.flatten(), 3)
# # plt.show()
#
# # tmp = dset_analogCorrected[0,...].transpose(0,2,1)
# # tmp_median = np.median(tmp, axis=0)
# # pg.image(tmp_median)
#
# print('max = ', tmp.max())


dataFileName = '/gpfs/cfel/cxi/scratch/user/gevorkov/python_saved_workspace/lysozymeData_m3_averageFiltered.h5'

dataFile = h5py.File(dataFileName, 'r', libver='latest')
medianFiltered = dataFile['/medianFiltered'][...]
meanFiltered = dataFile['/meanFiltered'][...]

#print(dset_analogCorrected.shape)

# pg.image(dset_analogCorrected[0,0,...].T)

a = pg.image(medianFiltered.transpose(0,2,1))
b = pg.image(meanFiltered.transpose(0,2,1))

i = 1
