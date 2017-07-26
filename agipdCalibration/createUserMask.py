"""
Create user-defined mask

For example: beam stop, ASIC edges, known bad regions, etc.

combinedMask: logical OR of all sub-masks

Note: when no pixels/cells are masked, plotting throws an exception!

"""

import h5py
import numpy as np
import os

import matplotlib.pyplot as plt
import pyqtgraph as pg


moduleID = 'M314'
moduleNumber = 'm7'
temperature = 'temperature_m15C'
baseDir = "/gpfs/cfel/fsds/labs/agipd/calibration/processed/{}/{}".format(moduleID, temperature)

# Flags for which masks to create
manual = False
systematic = True # memory cell 0
asicEdge = False

# Input files


# Output mask file
saveFileName = os.path.join(baseDir, 'userMask_{}_{}_{}.h5'.format(moduleID, moduleNumber, temperature))

# Create output file
saveFile = h5py.File(saveFileName, "w", libver='latest')
dset_userMask = saveFile.create_dataset("combinedMask", shape=(352, 128, 512), dtype=bool)
subMask = saveFile.create_group("subMasks")

userMask = np.zeros((352, 128, 512), dtype=bool)


################# manual mask - please edit (e.g. beamstop) ############
manualMask = np.zeros((128, 512), dtype=bool)
dset_manualMask = subMask.create_dataset("manualMask", shape=(352, 128, 512), dtype=bool)

if manual:
    # !!Here is an example, edit to reflect your needs/setup!!
    manualMask[65:105, 200:260] = True
    manualMask[80:105, 200:511] = True

    manualMask[80:86, 22:43] = True
    manualMask[75:100, 31:43] = True

    manualMask[64:127, 127:195] = True

dset_manualMask[...] = manualMask

userMask = np.logical_or(manualMask, userMask)
########################################################################


######### systematic mask - please edit ################################
systematicMask = np.zeros((128, 512, 11, 32), dtype=bool)
dset_systMask = subMask.create_dataset("systMask", shape=(352, 128, 512), dtype=bool)

if systematic:
    # systematicMask[..., 27:29] = True  # 32-column tips
    systematicMask[..., 0, 0] = True # memory cell 0

systematicMask = systematicMask.reshape((128, 512, 352)).transpose(2, 0, 1)
dset_systMask[...] = systematicMask

userMask = np.logical_or(systematicMask, userMask)
########################################################################


######### bad asic borders mask - please edit ##########################
badAsicBordersMask = np.zeros((352, 128, 512), dtype=bool)
dset_asicEdgeMask = subMask.create_dataset("asicEdgeMask", shape=(352, 128, 512), dtype=bool)

if asicEdge:
    badAsicBordersMask[:, (0, 63, 64, 127), :] = True
    for column in np.arange(8):
        badAsicBordersMask[:, :, (column * 64, column * 64 + 63)] = True

dset_asicEdgeMask[...] = badAsicBordersMask

userMask = np.logical_or(badAsicBordersMask, userMask)
########################################################################


##################### the rest can be left untouched ###################

dset_userMask[...] = userMask
saveFile.flush()
saveFile.close()
print('userMask saved in ', saveFileName)

userMaskFile = h5py.File(saveFileName, "r", libver='latest')
userMask = userMaskFile['combinedMask'][...]
pg.image(userMask.transpose(0, 2, 1))

print('\n\npercentage of masked cells: ', 100 * userMask.flatten().sum() / userMask.size)
print('\n\n\npress enter to quit')
tmp = input()
