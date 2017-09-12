from multiprocessing import Pool
import h5py
import sys
import time

from algorithms.xRayTubeDataFitting import *


def computePhotonSpacingOnePixel(analog, linearIndex, perMillInterval, memcell):
    localityRadius = 800
    #localityRadius = 800
    samplePointsCount = 1000

    (photonSpacing,
     quality,
     peakStdDevs,
     peakErrors,
     spacingError) = getOnePhotonAdcCountsXRayTubeData(analog,
                                                       localityRadius=localityRadius,
                                                       lwopassSamplePointsCount=samplePointsCount)

    # if np.mod(linearIndex, perMillInterval) == 0:
    #     print(0.1 * linearIndex / perMillInterval, '%')

    return (photonSpacing, quality, peakStdDevs, peakErrors, spacingError)


class BatchProcessXRayTubeData():
    def __init__(self, input_fname, output_fname):

        self.input_fname = input_fname
        self.output_fname = output_fname
        print('\n\n\nstart batchProcessXRayTubeData')
        print('input_fname = ', self.input_fname)
        print('output_fname = ', self.output_fname)

        self.run()

    def run(self):

        totalTime = time.time()

        f = h5py.File(self.input_fname, 'r', libver='latest')
        print('start loading analog from', self.input_fname)
        analog_data = f['analog'][()]
        print('loading done')
        f.close()

        analog_data = analog_data[1:, ...]  # first value is always wrong

        # is of shape (<frames>, <memcells>, <pixel rows>, <pixel cols>)
        n_memcells = analog_data.shape[1]
        n_memcells = 1

        photonSpacing = np.zeros((128, 512, n_memcells))
        quality = np.zeros((128, 512, n_memcells))
        peakStdDevs = np.zeros((128, 512, 2, n_memcells))
        peakErrors = np.zeros((128, 512, 2, n_memcells))
        spacingError = np.zeros((128, 512, n_memcells))

        print('creating linear index')
        linearIndices = np.arange(128 * 512)
        (matrixIndexY, matrixIndexX) = np.unravel_index(linearIndices, (128, 512))

        for memcell in range(n_memcells):
            print("Processing memcell", memcell)

            analog = analog_data[:, memcell, ...]
            print("analog.shape", analog.shape)

            print("create perMillInterval")
            pixelList = []
            perMillInterval = int(np.round(linearIndices.size / 1000))
            for i in np.arange(linearIndices.size):
                pixelList.append((analog[:, matrixIndexY[i], matrixIndexX[i]],
                                  i,
                                  perMillInterval, memcell))
            print("pixelList")
            print(len(pixelList))

            # computePhotonSpacingOnePixel(*pixelList[0])
            #computePhotonSpacingOnePixel(analog[:, 5, 20], 1, 1)

            # for i in np.arange(linearIndices.size):
            #     print(i)
            #     computePhotonSpacingOnePixel(*pixelList[i])


        print('start parallel computation')
        p = Pool()
        parallelResult = p.starmap(computePhotonSpacingOnePixel, pixelList, chunksize=10)
        p.close()

        print('all computation done')

        for i in np.arange(linearIndices.size):
            (photonSpacing[matrixIndexY[i], matrixIndexX[i], memcell],
             quality[matrixIndexY[i], matrixIndexX[i], memcell],
             peakStdDevs[matrixIndexY[i], matrixIndexX[i], :, memcell],
             peakErrors[matrixIndexY[i], matrixIndexX[i], :, memcell],
             spacingError[matrixIndexY[i], matrixIndexX[i], memcell]) = \
            parallelResult[i]

        print('start saving results at', self.output_fname)

        saveFile = h5py.File(self.output_fname, "w", libver='latest')
        dset_photonSpacing = saveFile.create_dataset("photonSpacing",
                                                     shape=(128, 512, n_memcells),
                                                     dtype='int16')
        dset_quality = saveFile.create_dataset("quality",
                                               shape=(128, 512, n_memcells),
                                               dtype='int16')
        dset_peakStdDevs = saveFile.create_dataset("peakStdDevs",
                                                   shape=(128, 512, 2, n_memcells),
                                                   dtype='int16')
        dset_peakErrors = saveFile.create_dataset("peakErrors",
                                                  shape=(128, 512, 2, n_memcells),
                                                  dtype='float32')
        dset_spacingError = saveFile.create_dataset("spacingError",
                                                    shape=(128, 512, n_memcells),
                                                    dtype='float32')


        dset_photonSpacing[...] = photonSpacing
        dset_quality[...] = quality
        dset_peakStdDevs[...] = peakStdDevs
        dset_peakErrors[...] = peakErrors
        dset_spacingError[...] = spacingError

        print('saved')

        saveFile.flush()
        saveFile.close()

        print('batchProcessXRayTubeData took time:  ', time.time() - totalTime, '\n\n')

if __name__ == "__main__":
    import os

    base_dir = "/gpfs/cfel/fsds/labs/agipd/calibration/processed/M302/temperature_m15C/xray/"

    #input_fname = os.path.join(base_dir, "test.h5")
    #output_fname = os.path.join(base_dir, "test_processed.h5")

    input_fname = os.path.join(base_dir, "test_AGIPD00_s00000.h5")
    output_fname = os.path.join(base_dir, "test_AGIPD00_s00000_processed.h5")

    obj = BatchProcessXRayTubeData(input_fname, output_fname)
