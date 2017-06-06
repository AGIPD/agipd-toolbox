from multiprocessing import Pool
import h5py
import sys
import time

from agipdCalibration.algorithms.xRayTubeDataFitting import *


def computePhotonSpacingOnePixel(analog, linearIndex, perMillInterval):
    localityRadius = 800
    samplePointsCount = 1000

    (photonSpacing, quality) = getOnePhotonAdcCountsXRayTubeData(analog, localityRadius, samplePointsCount)

    # if np.mod(linearIndex, perMillInterval) == 0:
    #     print(0.1 * linearIndex / perMillInterval, '%')

    return (photonSpacing, quality)


if __name__ == '__main__':
    # fileName = '/gpfs/cfel/fsds/labs/processed/calibration_1.1/mokalphaData_m1_new_v2.h5'
    # saveFileName ='/gpfs/cfel/fsds/labs/processed/calibration_1.1/photonSpacing_m1_new_v2.h5'
    # fileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/xRay200.h5'
    # saveFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/photonSpacing200.h5'
    fileName = sys.argv[1]
    saveFileName = sys.argv[2]
    print('\n\n\nstart batchProcessXRayTubeData')
    print('fileName = ', fileName)
    print('saveFileName = ', saveFileName)

    saveFile = h5py.File(saveFileName, "w", libver='latest')
    dset_photonSpacing = saveFile.create_dataset("photonSpacing", shape=(128, 512), dtype='int16')
    dset_quality = saveFile.create_dataset("quality", shape=(128, 512), dtype='int16')

    totalTime = time.time()

    f = h5py.File(fileName, 'r', libver='latest')
    print('start loading analog from', fileName)
    analog = f['analog'][()]
    print('loading done')
    f.close()

    analog = analog[1:,...] #first value is always wrong

    print('creating linear index')
    linearIndices = np.arange(128 * 512)
    (matrixIndexY, matrixIndexX) = np.unravel_index(linearIndices, (128, 512))

    pixelList = []
    perMillInterval = int(np.round(linearIndices.size / 1000))
    for i in np.arange(linearIndices.size):
        pixelList.append((analog[:, matrixIndexY[i], matrixIndexX[i]], i, perMillInterval))

    # computePhotonSpacingOnePixel(*pixelList[0])
    computePhotonSpacingOnePixel(analog[:, 5, 20], 1, 1)

    # for i in np.arange(linearIndices.size):
    #     print(i)
    #     computePhotonSpacingOnePixel(*pixelList[i])


    print('start parallel computation')
    p = Pool()
    parallelResult = p.starmap(computePhotonSpacingOnePixel, pixelList, chunksize=10)
    p.close()
    print('all computation done')

    photonSpacing = np.zeros((128, 512))
    quality = np.zeros((128, 512))
    for i in np.arange(linearIndices.size):
        (photonSpacing[matrixIndexY[i], matrixIndexX[i]], quality[matrixIndexY[i], matrixIndexX[i]]) = parallelResult[i]
    print('start saving results at', saveFileName)

    dset_photonSpacing[...] = photonSpacing
    dset_quality[...] = quality

    print('saved')

    saveFile.flush()
    saveFile.close()

    print('batchProcessXRayTubeData took time:  ', time.time() - totalTime, '\n\n')
