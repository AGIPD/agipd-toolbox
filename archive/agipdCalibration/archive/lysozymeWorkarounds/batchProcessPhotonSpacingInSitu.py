import time
from multiprocessing import Pool

import h5py

from agipdCalibration.archive.photonSpacingWorkaroundInSitu import *


def computePhotonSpacingOnePixel_inSitu(analog, linearIndex, perMillInterval):
    (photonSpacing, quality) = getOnePhotonAdcCountsInSitu(analog)

    if np.mod(linearIndex, perMillInterval) == 0:
        print(0.1 * linearIndex / perMillInterval, '%')

    return (photonSpacing, quality)


if __name__ == '__main__':
    fileName = '/gpfs/cfel/cxi/scratch/user/gevorkov/python_saved_workspace/lysozymeData_m1.h5'
    saveFileName = '/gpfs/cfel/cxi/scratch/user/gevorkov/python_saved_workspace/photonSpacing_m1_inSitu.h5'
    # fileName = sys.argv[1]
    # saveFileName = sys.argv[2]

    saveFile = h5py.File(saveFileName, "w", libver='latest')
    dset_photonSpacing = saveFile.create_dataset("photonSpacing", shape=(128, 512), dtype='int16')
    dset_quality = saveFile.create_dataset("quality", shape=(128, 512), dtype='int16')

    print('start loading')
    f = h5py.File(fileName, 'r', libver='latest')
    t = time.time()
    analog = f['analog'][:, 175, ...]
    f.close()
    print('loading done, took time:', time.time() - t)

    print('creating linear index')
    linearIndices = np.arange(128 * 512)
    (matrixIndexY, matrixIndexX) = np.unravel_index(linearIndices, (128, 512))

    pixelList = []
    perMillInterval = int(np.round(linearIndices.size / 1000))
    for i in np.arange(linearIndices.size):
        pixelList.append((analog[:, matrixIndexY[i], matrixIndexX[i]], i, perMillInterval))

    print('start parallel pool ')

    p = Pool()
    parallelResult = p.starmap(computePhotonSpacingOnePixel_inSitu, pixelList, chunksize=10)

    print('all calculation done')

    # parallelResult = []
    # for i in np.arange(linearIndices.size):
    #     print('idx', i)
    #     i = 43773
    #     parallelResult.append(computePhotonSpacingOnePixel_inSitu(*pixelList[i]))

    photonSpacing = np.zeros((128, 512))
    quality = np.zeros((128, 512))
    for i in np.arange(linearIndices.size):
        (photonSpacing[matrixIndexY[i], matrixIndexX[i]], quality[matrixIndexY[i], matrixIndexX[i]]) = parallelResult[i]

    print('saving')

    dset_photonSpacing[...] = photonSpacing
    dset_quality[...] = quality

    print('saved')

    saveFile.flush()
    saveFile.close()
