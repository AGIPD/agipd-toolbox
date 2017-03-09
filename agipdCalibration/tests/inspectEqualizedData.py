import h5py
import numpy as np

import matplotlib.pyplot as plt
import pyqtgraph as pg

dataFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/correctedProteinDestructionData/m3_nilyso_3a_30percent_135keV_pos_-640.000_00000.nxs'

dataFile = h5py.File(dataFileName, 'r', libver='latest')
dset_analogCorrected = dataFile['/analogCorrected']
dset_digitalGainStage = dataFile['/digitalGainStage']

# print(dset_analogCorrected.shape)

a = pg.image(dset_analogCorrected[...].transpose(0, 2, 1))
# b = pg.image(dset_digitalGainStage[...].transpose(0, 2, 1))

# pg.image(dset_analogCorrected[0,0,...].T)
# pg.image(dset_digitalGainStage[0,0,...])
# tmp = dset_digitalGainStage[...]
# plt.hist(tmp.flatten(), 3)
# plt.show()

# tmp = dset_analogCorrected[0,...].transpose(0,2,1)
# tmp_median = np.median(tmp, axis=0)
# pg.image(tmp_median)

# print('max = ', tmp.max())


# dataFileName = '/gpfs/cfel/cxi/scratch/user/gevorkov/agipdCalibration_workspace/digitalMeans_m2.h5'
#
# dataFile = h5py.File(dataFileName, 'r', libver='latest')
# digitalMeans  = dataFile['/digitalMeans'][...] #shape=(352, 3, 128, 512)
#
# #print(dset_analogCorrected.shape)
#
# # pg.image(dset_analogCorrected[0,0,...].T)
#
# plt.hist(digitalMeans[:,0,64:128,0:64].flatten(),300)
# plt.hist(digitalMeans[:,1,64:128,0:64].flatten(),300)
# # plt.gca().set_yscale("log")
# plt.show()

i = 1
