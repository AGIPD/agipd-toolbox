import h5py
import numpy as np
from agipdCalibration.tests.h5py_display import h5disp

import matplotlib.pyplot as plt
import pyqtgraph as pg

dataFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/darkOffset_m3.h5'
# dataFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/python_saved_workspace/darkOffset_agipd11.h5'
dataFile = h5py.File(dataFileName, 'r', libver='latest')
standardDeviation = dataFile['/darkStandardDeviation'][...]
offset = dataFile['/darkOffset'][...]  # shape = (352, 128, 512)

# tmp = standardDeviation[:,0:64,0:64]  #shape = (352, 64, 64)
# tmp = tmp.transpose(1,2,0) #shape = (64, 64, 352)
# tmp = tmp.reshape([64*64, 11, 32])
# pg.image(tmp.transpose(0,2,1))


# standardDeviation2D_64x64 = np.empty([64 * 64, 11, 32])
# standardDeviation2D_64x64 = standardDeviation[:, 0:64, 0:64]
# standardDeviation2D_64x64 = standardDeviation2D_64x64.transpose(1, 2, 0) #shape = (64, 64, 352)
# standardDeviation2D_64x64 = standardDeviation2D_64x64.reshape([64, 64, 11, 32])
#
# offset2D_64x64 = np.empty([64 * 64, 11, 32])
# offset2D_64x64 = offset[:, 0:64, 0:64]
# offset2D_64x64 = offset2D_64x64.transpose(1, 2, 0) #shape = (64, 64, 352)
# offset2D_64x64 = offset2D_64x64.reshape([64, 64, 11, 32])
#
# overviewStdDev_64x64 = np.empty((64 * 11, 64 * 32))
# overviewOffset_64x64 = np.empty((64 * 11, 64 * 32))
# for y in np.arange(64):
#     for x in np.arange(64):
#         overviewStdDev_64x64[y * 11:(y + 1) * 11, x * 32:(x + 1) * 32] = standardDeviation2D_64x64[y, x, ...]
#         overviewOffset_64x64[y * 11:(y + 1) * 11, x * 32:(x + 1) * 32] = offset2D_64x64[y, x, ...]
#
#
# # pg.image(overviewStdDev64x64.transpose(1,0))
# # pg.image(overviewOffset64x64.transpose(1,0))
#
# mask = np.zeros((128,512,11,32))
# mask[...,27:29] = 1 #32-column tips
#
# lineMediansStdDev2D_64x64 = np.empty((64 * 11, 64))
# lineMediansOffset2D_64x64 = np.empty((64 * 11, 64))
# for y in np.arange(64):
#     for x in np.arange(64):
#         lineMediansStdDev2D_64x64[y * 11:(y + 1) * 11, x] = np.median(standardDeviation2D_64x64[y, x, ...], axis=1)
#         lineMediansOffset2D_64x64[y * 11:(y + 1) * 11, x] = np.median(offset2D_64x64[y, x, ...], axis=1)
#
# pg.image(lineMediansStdDev2D_64x64.transpose(1, 0))
# pg.image(lineMediansOffset2D_64x64.transpose(1, 0))

standardDeviation = standardDeviation.transpose(1, 2, 0)  # shape = (128, 512, 352)
standardDeviation = standardDeviation.reshape([128, 512, 11, 32])

offset = offset.transpose(1, 2, 0)  # shape = (128, 512, 352)
offset = offset.reshape([128, 512, 11, 32])

overviewStdDev = np.empty((128 * 11, 512 * 32))
overviewOffset = np.empty((128 * 11, 512 * 32))
for y in np.arange(128):
    for x in np.arange(512):
        overviewStdDev[y * 11:(y + 1) * 11, x * 32:(x + 1) * 32] = standardDeviation[y, x, ...]
        overviewOffset[y * 11:(y + 1) * 11, x * 32:(x + 1) * 32] = offset[y, x, ...]

pg.image(overviewStdDev.transpose(1,0))
pg.image(overviewOffset.transpose(1,0))

# lineMediansStdDev2D = np.empty((128 * 11, 512))
# lineMediansOffset2D = np.empty((128 * 11, 512))
# for y in np.arange(128):
#     for x in np.arange(512):
#         lineMediansStdDev2D[y * 11:(y + 1) * 11, x] = np.median(standardDeviation2D[y, x, ...], axis=1)
#         lineMediansOffset2D[y * 11:(y + 1) * 11, x] = np.median(offset2D[y, x, ...], axis=1)
#
# pg.image(lineMediansStdDev2D.transpose(1, 0))
# pg.image(lineMediansOffset2D.transpose(1, 0))

lineMediansStdDev = np.empty((128, 512, 11))
lineMediansOffset = np.empty((128, 512, 11))
for y in np.arange(128):
    for x in np.arange(512):
        lineMediansStdDev[y, x, :] = np.median(standardDeviation[y, x, ...], axis=1)
        lineMediansOffset[y, x, :] = np.median(offset[y, x, ...], axis=1)

# pg.image(lineMediansStdDev.transpose(2, 1, 0))
# pg.image(lineMediansOffset.transpose(2, 1, 0))

mask = np.zeros((128, 512, 11, 32), dtype=bool)
mask[..., 27:29] = 1  # 32-column tips

stdDevRange = [10, 20]
offsetRange = [3800, 5800]

badStdDevPixelMask = np.zeros((128, 512, 11, 32), dtype=bool)
badStdDevPixelMask[standardDeviation < stdDevRange[0]] = True
badStdDevPixelMask[standardDeviation > stdDevRange[1]] = True

badOffsetPixelMask = np.zeros((128, 512, 11, 32), dtype=bool)
badOffsetPixelMask[offset < offsetRange[0]] = True
badOffsetPixelMask[offset > offsetRange[1]] = True

for y in np.arange(128):
    for x in np.arange(512):
        tmp = np.median(lineMediansStdDev[y, x, :])
        if tmp > stdDevRange[1] or tmp < stdDevRange[0]:
            badStdDevPixelMask[y, x, ...] = True
        else:
            for line in np.arange(11):
                if lineMediansStdDev[y, x, line]  > stdDevRange[1] or lineMediansStdDev[y, x, line]  < stdDevRange[0]:
                    badStdDevPixelMask[y, x, line, :] = True

        tmp = np.median(lineMediansOffset[y, x, :])
        if tmp > offsetRange[1] or tmp < offsetRange[0]:
            badOffsetPixelMask[y, x, ...] = True
        else:
            for line in np.arange(11):
                if lineMediansOffset[y, x, line] > offsetRange[1] or lineMediansOffset[y, x, line] < offsetRange[0]:
                    badOffsetPixelMask[y, x, line, :] = True

overviewStdDev = np.empty((128 * 11, 512 * 32))
overviewOffset = np.empty((128 * 11, 512 * 32))
for y in np.arange(128):
    for x in np.arange(512):
        overviewStdDev[y * 11:(y + 1) * 11, x * 32:(x + 1) * 32] = badStdDevPixelMask[y, x, ...]
        overviewOffset[y * 11:(y + 1) * 11, x * 32:(x + 1) * 32] = badOffsetPixelMask[y, x, ...]

pg.image(overviewStdDev.transpose(1, 0))
pg.image(overviewOffset.transpose(1, 0))

i = 0
