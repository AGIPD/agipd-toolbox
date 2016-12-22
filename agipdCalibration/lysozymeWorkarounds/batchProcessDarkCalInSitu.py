import sys
import time
from multiprocessing import Pool

import h5py

from agipdCalibration.lysozymeWorkarounds.darkCalWorkaroundInSitu import *


def computeDarkCalsOnePixel(analog, linearIndex):
    darkCalsOnePixel = np.zeros(352)

    for i in np.arange(352):
        darkCalsOnePixel[i] = getDarkCalInSitu(analog[:, i])

    # print('subpixel ' + str(linearIndex) + ' done')
    return darkCalsOnePixel


if __name__ == '__main__':
    moduleNumber = sys.argv[1]
    dataFileName = '/gpfs/cfel/cxi/scratch/user/gevorkov/python_saved_workspace/lysozymeData_m' + moduleNumber + '.h5'
    saveFileName = '/gpfs/cfel/cxi/scratch/user/gevorkov/python_saved_workspace/lysozymeData_m' + moduleNumber + '_darkcal_inSitu.h5'

    saveFile = h5py.File(saveFileName, "w", libver='latest')
    dset_darkOffset = saveFile.create_dataset("darkOffset", shape=(352, 128, 512), chunks=(1, 128, 512), dtype='int16')

    p = Pool()
    columnsToLoadPerIteration = 64
    rowsToLoadPerIteration = 64
    for column in np.arange(512 / columnsToLoadPerIteration):
        for row in np.arange(128 / rowsToLoadPerIteration):
            interestingPixelsY = (int(row * rowsToLoadPerIteration), int((row + 1) * rowsToLoadPerIteration))
            interestingPixelsX = (int(column * columnsToLoadPerIteration), int((column + 1) * columnsToLoadPerIteration))

            print('loading data row ' + str(row) + ' column ' + str(column))
            t = time.time()
            f = h5py.File(dataFileName, 'r', libver='latest')
            analog = f['/analog'][:, :, interestingPixelsY[0]:interestingPixelsY[1], interestingPixelsX[0]:interestingPixelsX[1]]
            f.close()
            print('took time:  ' + str(time.time() - t))

            print('creating linear index row ' + str(row) + ' column ' + str(column))
            linearIndex = np.arange((interestingPixelsY[1] - interestingPixelsY[0]) * (interestingPixelsX[1] - interestingPixelsX[0]))
            (matrixIndexY, matrixIndexX) = np.unravel_index(linearIndex,
                                                            ((interestingPixelsY[1] - interestingPixelsY[0]), (interestingPixelsX[1] - interestingPixelsX[0])))

            pixelList = []
            for i in np.arange(linearIndex.size):
                pixelList.append((analog[:, :, matrixIndexY[i], matrixIndexX[i]], i))

            print('start parallel pool row ' + str(row) + ' column ' + str(column))

            ##########
            # for i in np.arange(3700, linearIndex.size):
            #     print 'i = ' + str(i)
            #     computeDarkCalsOnePixel(pixelList[i])
            ##########

            parallelResult = p.starmap(computeDarkCalsOnePixel, pixelList)

            print('all calculation done row ' + str(row) + ' column ' + str(column))

            darkOffsets = np.empty((352, (interestingPixelsY[1] - interestingPixelsY[0]), (interestingPixelsX[1] - interestingPixelsX[0])), dtype='int16')
            for i in np.arange(linearIndex.size):
                darkOffsets[:, matrixIndexY[i], matrixIndexX[i]] = parallelResult[i]

            print('start saving row ' + str(row) + ' column ' + str(column))

            dset_darkOffset[:, interestingPixelsY[0]:interestingPixelsY[1], interestingPixelsX[0]:interestingPixelsX[1]] = darkOffsets

            saveFile.flush()

            print('saved row ' + str(row) + ' column ' + str(column))

    saveFile.close()
    p.close()
