from multiprocessing import Pool
import h5py
import sys
import time

from .algorithms.xRayTubeDataFitting import *


def computePhotonSpacingOnePixel(analog, linearIndex, perMillInterval):
    localityRadius = 800
    samplePointsCount = 1000

    (photonSpacing, quality, peakStdDevs, peakErrors, spacingError) = getOnePhotonAdcCountsXRayTubeData(analog, localityRadius, samplePointsCount)

    # if np.mod(linearIndex, perMillInterval) == 0:
    #     print(0.1 * linearIndex / perMillInterval, '%')

    return (photonSpacing, quality, peakStdDevs, peakErrors, spacingError)


class BatchProcessXRayTubeData():
    def __init__(self, fileName, saveFileName):

        self.fileName = fileName
        self.saveFileName = saveFileName
        print('\n\n\nstart batchProcessXRayTubeData')
        print('fileName = ', self.fileName)
        print('saveFileName = ', self.saveFileName)

        self.run()

    def run(self):

        saveFile = h5py.File(self.saveFileName, "w", libver='latest')
        dset_photonSpacing = saveFile.create_dataset("photonSpacing", shape=(128, 512), dtype='int16')
        dset_quality = saveFile.create_dataset("quality", shape=(128, 512), dtype='int16')
        dset_peakStdDevs = saveFile.create_dataset("peakStdDevs", shape=(128, 512, 2), dtype='int16')
        dset_peakErrors = saveFile.create_dataset("peakErrors", shape=(128, 512, 2), dtype='float32')
        dset_spacingError = saveFile.create_dataset("spacingError", shape=(128, 512), dtype='float32')

        totalTime = time.time()

        f = h5py.File(self.fileName, 'r', libver='latest')
        print('start loading analog from', self.fileName)
        analog = f['analog'][()]
        print('loading done')
        f.close()

        analog = analog[1:, ...]  # first value is always wrong

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
        peakStdDevs = np.zeros((128, 512, 2))
        peakErrors = np.zeros((128, 512, 2))
        spacingError = np.zeros((128, 512))
        for i in np.arange(linearIndices.size):
            (photonSpacing[matrixIndexY[i], matrixIndexX[i]], quality[matrixIndexY[i], matrixIndexX[i]], peakStdDevs[matrixIndexY[i], matrixIndexX[i], :], peakErrors[matrixIndexY[i], matrixIndexX[i], :], spacingError[matrixIndexY[i], matrixIndexX[i]]) = \
            parallelResult[i]
        print('start saving results at', self.saveFileName)

        dset_photonSpacing[...] = photonSpacing
        dset_quality[...] = quality
        dset_peakStdDevs[...] = peakStdDevs
        dset_peakErrors[...] = peakErrors
        dset_spacingError[...] = spacingError

        print('saved')

        saveFile.flush()
        saveFile.close()

        print('batchProcessXRayTubeData took time:  ', time.time() - totalTime, '\n\n')
