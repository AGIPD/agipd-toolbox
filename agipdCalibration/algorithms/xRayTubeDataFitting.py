import numpy as np
from scipy.signal import convolve
from scipy.signal import find_peaks_cwt

from agipdCalibration.algorithms.helperFunctions import orderFilterCourse

import matplotlib.pyplot as plt


def getPhotonHistogramLowpassCorrected(analog, localityRadius, lowpassSamplePointsCount):
    data = analog

    # plt.plot(data,'.')
    # plt.show()

    localData = data[2 * localityRadius:6 * localityRadius]
    localBinEdges = np.arange(np.min(localData), np.max(localData))

    (localHistogram, _) = np.histogram(localData, localBinEdges)
    smoothWindowSize = 11
    localHistogramSmooth = convolve(localHistogram, np.ones((smoothWindowSize,)), mode='same')

    mostFrequentLocalValue = np.mean(localBinEdges[np.argmax(localHistogramSmooth)]).astype(int)

    # set order to be the same order as the maximum of the histogram in the local data. Should be least affected by noise
    order = (2 * localityRadius * np.mean(np.nonzero(np.sort(localData) == mostFrequentLocalValue)) / localData.shape[0])

    if np.isnan(order):
        order = np.array(0.4 * 2 * localityRadius)

    order = order.astype(int)

    # plt.figure(1)
    # plt.plot(localBinEdges[0:-1], localHistogramSmooth)


    lowPassData = orderFilterCourse(data, localityRadius, order, lowpassSamplePointsCount)

    # plt.figure(2)
    # plt.plot(lowPassData)


    correctedData = data - lowPassData

    # plt.figure(3)
    # plt.plot(correctedData,'*')

    binEdges = np.arange(np.min(correctedData), np.max(correctedData))

    return np.histogram(correctedData, binEdges)


def getOnePhotonAdcCountsXRayTubeData(analog, applyLowpass=True, localityRadius=801, lwopassSamplePointsCount=1000):
    if applyLowpass:
        (photonHistoramValues, photonHistogramBins) = getPhotonHistogramLowpassCorrected(analog, localityRadius, lwopassSamplePointsCount)
    else:
        photonHistogramBins = np.arange(np.min(analog), np.max(analog), 1, dtype='int16')
        (photonHistoramValues, _) = np.histogram(analog, photonHistogramBins)

    # plt.figure(4)
    # plt.plot(photonHistogramBins[0:-1],photonHistoramValues)
    # plt.show()

    smoothWindowRange = (5, 50)
    smoothWindowStep = 4
    peakIndices = np.unique(np.array(find_peaks_cwt(photonHistoramValues, np.arange(smoothWindowRange[0], smoothWindowRange[1], smoothWindowStep))))
    if peakIndices.size < 2:
        return (0, 0)

    smoothWindowSize = np.round(smoothWindowRange[0] + 0.5 * (smoothWindowRange[1] - smoothWindowRange[0])).astype(int)
    photonHistogramValuesSmooth = convolve(photonHistoramValues, np.ones((smoothWindowSize,)), mode='same')
    # plt.plot(photonHistogramBins[0:-1], photonHistogramValuesSmooth)
    # plt.show()

    peakSizes = photonHistogramValuesSmooth[peakIndices]
    sizeSortIndices = np.argsort(peakSizes)[::-1]
    sizeSortedPeakLocations = photonHistogramBins[peakIndices[sizeSortIndices]]
    sizeSortedPeakIndices = peakIndices[sizeSortIndices]
    peakSizes = peakSizes[sizeSortIndices]

    onePhotonAdcCounts = sizeSortedPeakLocations[1] - sizeSortedPeakLocations[0]

    if onePhotonAdcCounts < 0:
        return (0, 0)

    valleyDepthBetweenPeaks = peakSizes[1] - np.min(
        photonHistogramValuesSmooth[sizeSortedPeakIndices[0]:sizeSortedPeakIndices[1]])

    return (onePhotonAdcCounts, valleyDepthBetweenPeaks)
