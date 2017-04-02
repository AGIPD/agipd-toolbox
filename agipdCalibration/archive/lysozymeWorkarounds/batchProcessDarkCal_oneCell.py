import time
from multiprocessing import Pool

import h5py

from agipdCalibration.algorithms.helperFunctions import *


def computeDarkCalsOneCell(analog, linearIndex):

    darkCal = np.mean(analog)

    print('cell', linearIndex, ' done')
    return darkCal


if __name__ == '__main__':
    dataFileName = '/gpfs/cfel/cxi/scratch/user/gevorkov/python_saved_workspace/m2_spatial_position3_dark.h5'
    saveFileName = '/gpfs/cfel/cxi/scratch/user/gevorkov/python_saved_workspace/m2_spatial_position3_darkcal.h5'

    frontCutoffCount = 250

    saveFile = h5py.File(saveFileName, "w", libver='latest')
    dset_darkOffset = saveFile.create_dataset("darkOffset", shape=(128, 512), dtype='int16')

    print('loading data from', saveFileName)
    f = h5py.File(dataFileName, 'r', libver='latest')
    t = time.time()
    analog = f['/analog'][...]
    f.close()
    print('took time:  ' + str(time.time() - t))

    analog = analog[frontCutoffCount:,...]

    print('creating linear index')
    linearIndex = np.arange(128 * 512)
    (matrixIndexY, matrixIndexX) = np.unravel_index(linearIndex, (128, 512))

    pixelList = []
    for i in np.arange(linearIndex.size):
        pixelList.append((analog[:, matrixIndexY[i], matrixIndexX[i]], i))

    print('start parallel pool')

    ##########
    # for i in np.arange(3700, linearIndex.size):
    #     print 'i = ' + str(i)
    #     computeDarkCalsOnePixel(pixelList[i])
    ##########

    t = time.time()
    p = Pool(3)
    parallelResult = p.starmap(computeDarkCalsOneCell, pixelList)
    print('took time:  ' + str(time.time() - t))

    print('all calculation done')

    print('start collecting parallel results')
    t = time.time()
    darkOffsets = np.empty((128, 512), dtype='int16')
    for i in np.arange(linearIndex.size):
        darkOffsets[matrixIndexY[i], matrixIndexX[i]] = parallelResult[i]
    print('took time:  ' + str(time.time() - t))

    print('start saving')

    dset_darkOffset[...] = darkOffsets

    saveFile.flush()

    print('saved')

    saveFile.close()
