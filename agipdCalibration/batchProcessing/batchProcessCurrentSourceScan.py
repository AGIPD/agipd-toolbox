import gc
import time
from multiprocessing import Pool
import sys
import h5py

from agipdCalibration.algorithms.rangeScansFitting import *


def computeCurrentSourceCalibrationOneMemoryCell(analog, digital, linearIndex, perMillInterval):
    fitSlopesResult = fit3DynamicScanSlopes(analog, digital)

    # if np.mod(linearIndex, perMillInterval) == 0:
    #     print(0.1 * linearIndex / perMillInterval, '%')

    return fitSlopesResult


if __name__ == '__main__':
    # workspacePath = '/gpfs/cfel/fsds/labs/processed/Yaroslav/python_saved_workspace/'
    # dataFileName = workspacePath + 'currentSource_chunked.h5'
    # saveFileName_analogGains = workspacePath + 'analogGains_currentSource.h5'
    # saveFileName_digitalMeans = workspacePath + 'digitalMeans_currentSource.h5'

    dataFileName = sys.argv[1]
    saveFileName_analogGains = sys.argv[2]
    saveFileName_digitalMeans = sys.argv[3]

    print('\n\n\nstart batchProcessPulsedCapacitor')
    print('dataFileName = ', dataFileName)
    print('saveFileName_analogGains = ', saveFileName_analogGains)
    print('saveFileName_digitalMeans = ', saveFileName_digitalMeans)
    print(' ')

    workerCount = 3  # python parallelizes the rest! One worker makes 30 threads!

    totalTime = time.time()

    saveFile_analogGains = h5py.File(saveFileName_analogGains, "w", libver='latest')
    dset_analogGains = saveFile_analogGains.create_dataset("analogGains", shape=(3, 352, 128, 512), dtype='float32')
    dset_analogLineOffsets = saveFile_analogGains.create_dataset("anlogLineOffsets", shape=(3, 352, 128, 512), dtype='float32')
    dset_analogFitError = saveFile_analogGains.create_dataset("analogFitError", shape=(3, 352, 128, 512), dtype='float32')

    saveFile_digitalMeans = h5py.File(saveFileName_digitalMeans, "w", libver='latest')
    dset_digitalMeans = saveFile_digitalMeans.create_dataset("digitalMeans", shape=(352, 3, 128, 512), dtype='uint16')
    dset_digitalThresholds = saveFile_digitalMeans.create_dataset("digitalThresholds", shape=(2, 352, 128, 512), dtype='uint16')
    dset_digitalStdDeviations = saveFile_digitalMeans.create_dataset("digitalStdDeviations", shape=(352, 3, 128, 512), dtype='float32')
    dset_digitalSpacingsSafetyFactors = saveFile_digitalMeans.create_dataset("digitalSpacingsSafetyFactors", shape=(352, 3, 128, 512), dtype='float32')

    p = Pool(workerCount)
    dataFile = h5py.File(dataFileName, 'r', libver='latest')
    columnsToLoadPerIteration = 64
    rowsToLoadPerIteration = 64
    for column in np.arange(512 / columnsToLoadPerIteration): #np.arange(1):
        for row in np.arange(128 / rowsToLoadPerIteration): #np.arange(1):
            consideredPixelsY = (int(row * rowsToLoadPerIteration), int((row + 1) * rowsToLoadPerIteration))
            consideredPixelsX = (int(column * columnsToLoadPerIteration), int((column + 1) * columnsToLoadPerIteration))

            print('loading data, rows ', consideredPixelsY[0], '-', consideredPixelsY[1], ' columns ', + consideredPixelsX[0], '-', consideredPixelsX[1],
                  'from', dataFileName)
            t = time.time()
            analog = dataFile['/analog'][:, :, consideredPixelsY[0]:consideredPixelsY[1], consideredPixelsX[0]:consideredPixelsX[1]]
            digital = dataFile['/digital'][:, :, consideredPixelsY[0]:consideredPixelsY[1], consideredPixelsX[0]:consideredPixelsX[1]]
            print('took time:  ', time.time() - t)

            print('creating linear index, rows ', consideredPixelsY[0], '-', consideredPixelsY[1], ' columns ', + consideredPixelsX[0], '-',
                  consideredPixelsX[1])
            linearIndices = np.arange(352 * columnsToLoadPerIteration * rowsToLoadPerIteration)
            matrixIndices = np.unravel_index(linearIndices, (352, columnsToLoadPerIteration, rowsToLoadPerIteration))

            pixelList = []
            perMillInterval = int(np.round(linearIndices.size / 1000))
            for i in np.arange(linearIndices.size):
                idx = (slice(None), matrixIndices[0][i], matrixIndices[1][i], matrixIndices[2][i])
                pixelList.append((analog[idx], digital[idx], i, perMillInterval))

            # for i in np.arange(linearIndices.size):
            #     print(i, (slice(None), matrixIndices[0][i], matrixIndices[1][i], matrixIndices[2][i]))
            #     computeCurrentSourceCalibrationOneMemoryCell(*(pixelList[i]))

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
            digitalStdDev_highGain = np.empty(resultSize, dtype='float32')
            digitalStdDev_mediumGain = np.empty(resultSize, dtype='float32')
            digitalStdDev_lowGain = np.empty(resultSize, dtype='float32')

            digitalMeans_highGain = np.empty(resultSize, dtype='uint16')
            digitalMeans_mediumGain = np.empty(resultSize, dtype='uint16')
            digitalMeans_lowGain = np.empty(resultSize, dtype='uint16')

            for i in np.arange(linearIndices.size):
                idx = (matrixIndices[0][i], matrixIndices[1][i], matrixIndices[2][i])
                (((highGain[idx], offset_highGain[idx]), (mediumGain[idx], offset_mediumGain[idx]),(lowGain[idx], offset_lowGain[idx])),
                 (digitalMeans_highGain[idx], digitalMeans_mediumGain[idx], digitalMeans_lowGain[idx]),
                 (fitError_highGain[idx], fitError_mediumGain[idx], fitError_lowGain[idx]),
                 (digitalStdDev_highGain[idx], digitalStdDev_mediumGain[idx], digitalStdDev_lowGain[idx])
                 ) = parallelResult[i]

            print('start saving, rows ', consideredPixelsY[0], '-', consideredPixelsY[1], ' columns ', + consideredPixelsX[0], '-', consideredPixelsX[1],
                  'in', saveFileName_analogGains, 'and', saveFileName_digitalMeans)

            t = time.time()
            idx = (slice(0, 3), slice(None), slice(consideredPixelsY[0], consideredPixelsY[1]), slice(consideredPixelsX[0], consideredPixelsX[1]))
            dset_analogGains[idx] = np.stack((highGain, mediumGain, lowGain), axis=0)
            dset_analogLineOffsets[idx] = np.stack((offset_highGain, offset_mediumGain, offset_lowGain), axis=0)

            print('gains and offsets saved')

            dset_analogFitError[idx] = np.stack((fitError_highGain, fitError_mediumGain, fitError_lowGain), axis=0)

            print('analog fit errors saved')

            idx = (slice(None), slice(0, 3), slice(consideredPixelsY[0], consideredPixelsY[1]), slice(consideredPixelsX[0], consideredPixelsX[1]))
            dset_digitalMeans[idx] = np.stack((digitalMeans_highGain, digitalMeans_mediumGain, digitalMeans_lowGain), axis=1)
            dset_digitalStdDeviations[idx] = np.stack((digitalStdDev_highGain, digitalStdDev_mediumGain, digitalStdDev_lowGain), axis=1)

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
    dset_digitalSpacingsSafetyFactors[:, 2, ...] = (digitalMeans[:, 2, ...] - digitalMeans[:, 1, ...]) / (
        digitalStdDeviations[:, 2, ...] + digitalStdDeviations[:, 1, ...])
    dset_digitalSpacingsSafetyFactors[:, 1, ...] = np.maximum(dset_digitalSpacingsSafetyFactors[:, 0, ...], dset_digitalSpacingsSafetyFactors[:, 2, ...])

    print('digital spacings safety factors computed and saved')

    dataFile.close()
    p.close()
    saveFile_analogGains.close()
    saveFile_digitalMeans.close()

    print('\ndone with processing ', dataFileName)
    print('generated output: ', saveFileName_analogGains)
    print('generated output: ', saveFileName_digitalMeans)
    print('batchProcessPulsedCapacitor took time:  ', time.time() - totalTime, '\n\n')
