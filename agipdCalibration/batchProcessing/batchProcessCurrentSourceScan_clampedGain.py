import gc
import time
from multiprocessing import Pool
import sys
import h5py

from agipdCalibration.algorithms.rangeScansFitting import *


def computeCurrentSourceCalibrationOneMemoryCell(analog, digital, digitalMeanValues, linearIndex, perMillInterval):
    fitSlopesResult = fit3DynamicScanSlopes_precomputedDigitalMeans(analog, digital, digitalMeanValues)

    # if np.mod(linearIndex, perMillInterval) == 0:
    #     print(0.1 * linearIndex / perMillInterval, '%')

    return fitSlopesResult


if __name__ == '__main__':
    # workspacePath = '/gpfs/cfel/fsds/labs/processed/Yaroslav/python_saved_workspace/'
    # currentSourceDataFileName = workspacePath + 'currentSource_chunked.h5'
    # clampedDigitalMeansFileName = workspacePath + 'clampedDigitalMeans.h5'
    # saveFileName_analogGains = workspacePath + 'analogGains_currentSource.h5'
    # saveFileName_digitalMeans = workspacePath + 'digitalMeans_currentSource.h5'

    dataFileName = sys.argv[1]
    saveFileName_analogGains = sys.argv[2]
    saveFileName_digitalMeans = sys.argv[3]

    print('\n\n\nstart batchProcessCurrentSourceScan')
    print('dataFileName = ', currentSourceDataFileName)
    print('saveFileName_analogGains = ', saveFileName_analogGains)
    print('saveFileName_digitalMeans = ', saveFileName_digitalMeans)
    print(' ')

    workerCount = 3  # python parallelizes the rest! One worker makes 30 threads!

    totalTime = time.time()

    saveFile_analogGains = h5py.File(saveFileName_analogGains, "w", libver='latest')
    dset_analogGains = saveFile_analogGains.create_dataset("analogGains", shape=(3, 352, 128, 512), dtype='float32')
    dset_analogLineOffsets = saveFile_analogGains.create_dataset("anlogLineOffsets", shape=(3, 352, 128, 512), dtype='float32')
    dset_analogFitStdDevs = saveFile_analogGains.create_dataset("analogFitStdDevs", shape=(3, 352, 128, 512), dtype='float32')

    saveFile_digitalMeans = h5py.File(saveFileName_digitalMeans, "w", libver='latest')
    dset_digitalMeans = saveFile_digitalMeans.create_dataset("digitalMeans", shape=(352, 3, 128, 512), dtype='uint16')
    dset_digitalThresholds = saveFile_digitalMeans.create_dataset("digitalThresholds", shape=(2, 352, 128, 512), dtype='uint16')
    dset_digitalStdDeviations = saveFile_digitalMeans.create_dataset("digitalStdDeviations", shape=(352, 3, 128, 512), dtype='float32')
    dset_digitalSpacingsSafetyFactors = saveFile_digitalMeans.create_dataset("digitalSpacingsSafetyFactors", shape=(352, 2, 128, 512), dtype='float32')

    clampedDigitalMeansFile = h5py.File(clampedDigitalMeansFileName, 'r', libver='latest')
    clampedDigitalMeans = clampedDigitalMeansFile['clampedDigitalMeans'][...]
    clampedDigitalStandardDeviations = clampedDigitalMeansFile['clampedDigitalStandardDeviations'][...]

    p = Pool(workerCount)
    currentSourceDataFile = h5py.File(currentSourceDataFileName, 'r', libver='latest')
    columnsToLoadPerIteration = 64
    rowsToLoadPerIteration = 64
    for column in np.arange(512 / columnsToLoadPerIteration):  # np.arange(1):
        for row in np.arange(128 / rowsToLoadPerIteration):  # np.arange(1):
            consideredPixelsY = (int(row * rowsToLoadPerIteration), int((row + 1) * rowsToLoadPerIteration))
            consideredPixelsX = (int(column * columnsToLoadPerIteration), int((column + 1) * columnsToLoadPerIteration))

            print('loading data, rows ', consideredPixelsY[0], '-', consideredPixelsY[1], ' columns ', + consideredPixelsX[0], '-', consideredPixelsX[1],
                  'from', currentSourceDataFileName)
            t = time.time()
            analog = currentSourceDataFile['/analog'][:, :, consideredPixelsY[0]:consideredPixelsY[1], consideredPixelsX[0]:consideredPixelsX[1]]
            digital = currentSourceDataFile['/digital'][:, :, consideredPixelsY[0]:consideredPixelsY[1], consideredPixelsX[0]:consideredPixelsX[1]]
            print('took time:  ', time.time() - t)
            digitalMeans_local = clampedDigitalMeans[:, :, consideredPixelsY[0]:consideredPixelsY[1], consideredPixelsX[0]:consideredPixelsX[1]]
            digitalStandardDeviations_local = clampedDigitalStandardDeviations[:, :, consideredPixelsY[0]:consideredPixelsY[1],
                                              consideredPixelsX[0]:consideredPixelsX[1]]

            # #####################debug
            # import matplotlib.pyplot as plt
            # cell = 300
            # y = 30
            # x = 60
            # plt.plot(digital[:, cell, y, x])
            # plt.axhline(digitalMeans_local[0, cell, y, x])
            # plt.axhline(digitalMeans_local[1, cell, y, x])
            # plt.axhline(digitalMeans_local[2, cell, y, x])
            # plt.axhline(np.mean((digitalMeans_local[0, cell, y, x], digitalMeans_local[1, cell, y, x])), color='r')
            # plt.axhline(np.mean((digitalMeans_local[1, cell, y, x], digitalMeans_local[2, cell, y, x])), color='r')
            # plt.show()
            # ######################################

            print('creating linear index, rows ', consideredPixelsY[0], '-', consideredPixelsY[1], ' columns ', + consideredPixelsX[0], '-',
                  consideredPixelsX[1])
            linearIndices = np.arange(352 * columnsToLoadPerIteration * rowsToLoadPerIteration)
            matrixIndices = np.unravel_index(linearIndices, (352, columnsToLoadPerIteration, rowsToLoadPerIteration))

            pixelList = []
            perMillInterval = int(np.round(linearIndices.size / 1000))
            for i in np.arange(linearIndices.size):
                idx = (slice(None), matrixIndices[0][i], matrixIndices[1][i], matrixIndices[2][i])
                pixelList.append((analog[idx], digital[idx], digitalMeans_local[idx], i, perMillInterval))

            # ############debug
            # for i in np.arange(15*64*64+25*64 + 25,linearIndices.size):
            #     print(i, (slice(None), matrixIndices[0][i], matrixIndices[1][i], matrixIndices[2][i]))
            #     computeCurrentSourceCalibrationOneMemoryCell(*(pixelList[i]))
            # ##################

            print('start manual garbage collection')
            t = time.time()
            gc.collect()
            print('took time:  ', time.time() - t)

            print('start parallel computations, rows ', consideredPixelsY[0], '-', consideredPixelsY[1], ' columns ', + consideredPixelsX[0], '-',
                  consideredPixelsX[1])
            t = time.time()
            parallelResult = p.starmap(computeCurrentSourceCalibrationOneMemoryCell, pixelList, chunksize=352 * 4)
            print('took time:  ', time.time() - t)

            print('all calculation done, rows ', consideredPixelsY[0], '-', consideredPixelsY[1], ' columns ', + consideredPixelsX[0], '-',
                  consideredPixelsX[1])

            resultSize = (352, (consideredPixelsY[1] - consideredPixelsY[0]), (consideredPixelsX[1] - consideredPixelsX[0]))
            highGain = np.empty(resultSize, dtype='float32')
            mediumGain = np.empty(resultSize, dtype='float32')
            lowGain = np.empty(resultSize, dtype='float32')
            offset_highGain = np.empty(resultSize, dtype='float32')
            offset_mediumGain = np.empty(resultSize, dtype='float32')
            offset_lowGain = np.empty(resultSize, dtype='float32')
            fitError_highGain = np.empty(resultSize, dtype='float32')
            fitError_mediumGain = np.empty(resultSize, dtype='float32')
            fitError_lowGain = np.empty(resultSize, dtype='float32')

            for i in np.arange(linearIndices.size):
                idx = (matrixIndices[0][i], matrixIndices[1][i], matrixIndices[2][i])
                (((highGain[idx], offset_highGain[idx]), (mediumGain[idx], offset_mediumGain[idx]), (lowGain[idx], offset_lowGain[idx])),
                 (fitError_highGain[idx], fitError_mediumGain[idx], fitError_lowGain[idx])
                 ) = parallelResult[i]

            print('start saving, rows ', consideredPixelsY[0], '-', consideredPixelsY[1], ' columns ', + consideredPixelsX[0], '-', consideredPixelsX[1],
                  'in', saveFileName_analogGains, 'and', saveFileName_digitalMeans)

            t = time.time()
            idx = (slice(0, 3), slice(None), slice(consideredPixelsY[0], consideredPixelsY[1]), slice(consideredPixelsX[0], consideredPixelsX[1]))
            dset_analogGains[idx] = np.stack((highGain, mediumGain, lowGain), axis=0)
            dset_analogLineOffsets[idx] = np.stack((offset_highGain, offset_mediumGain, offset_lowGain), axis=0)

            print('gains and offsets saved')

            dset_analogFitStdDevs[idx] = np.stack((fitError_highGain, fitError_mediumGain, fitError_lowGain), axis=0)

            print('analog fit errors saved')

            idx = (slice(0, 352), slice(0, 3), slice(consideredPixelsY[0], consideredPixelsY[1]), slice(consideredPixelsX[0], consideredPixelsX[1]))
            dset_digitalMeans[idx] = digitalMeans_local.transpose((1, 0, 2, 3))
            dset_digitalStdDeviations[idx] = digitalStandardDeviations_local.transpose((1, 0, 2, 3))

            print('means and standard deviations saved')

            print('flushing')

            saveFile_analogGains.flush()
            saveFile_digitalMeans.flush()

            print('took time:  ', time.time() - t)

            print('saved, rows ', consideredPixelsY[0], '-', consideredPixelsY[1], ' columns ', + consideredPixelsX[0], '-', consideredPixelsX[1])
            print(' ')

    digitalMeans = dset_digitalMeans[...]
    lowThreshold = ((digitalMeans[:, 0, ...].astype('float32') + digitalMeans[:, 1, ...].astype('float32')) / 2).astype('uint16')
    dset_digitalThresholds[0, ...] = lowThreshold
    highThreshold = ((digitalMeans[:, 1, ...].astype('float32') + digitalMeans[:, 2, ...].astype('float32')) / 2).astype('uint16')
    dset_digitalThresholds[1, ...] = highThreshold

    print('digital threshold computed and saved')

    digitalStdDeviations = dset_digitalStdDeviations[...]
    dset_digitalSpacingsSafetyFactors[:, 0, ...] = (digitalMeans[:, 1, ...] - digitalMeans[:, 0, ...]) / (
        digitalStdDeviations[:, 1, ...] + digitalStdDeviations[:, 0, ...])
    dset_digitalSpacingsSafetyFactors[:, 1, ...] = (digitalMeans[:, 2, ...] - digitalMeans[:, 1, ...]) / (
        digitalStdDeviations[:, 2, ...] + digitalStdDeviations[:, 1, ...])

    print('digital spacings safety factors computed and saved')

    currentSourceDataFileName.close()
    clampedDigitalMeansFileName.close()
    p.close()
    saveFile_analogGains.close()
    saveFile_digitalMeans.close()

    print('\ndone with processing ', currentSourceDataFileName)
    print('generated output: ', saveFileName_analogGains)
    print('generated output: ', saveFileName_digitalMeans)
    print('batchProcessPulsedCapacitor took time:  ', time.time() - totalTime, '\n\n')
