import numpy as np
from scipy.signal import convolve
from scipy.optimize import curve_fit
import peakutils

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


def manual_gaussian_fit(x, y):
    initial = [np.max(y), x[0], (x[1] - x[0]) * 5]
    try:
        params, pcov = curve_fit(peakutils.peak.gaussian, x, y, initial)
    except:
        return 0

    return params[1]

def indexes_peakutilsManuallyAdjusted(y, thres=0.3, min_dist=1):
    '''Peak detection routine.

    Finds the peaks in *y* by taking its first order difference. By using
    *thres* and *min_dist* parameters, it is possible to reduce the number of
    detected peaks. *y* must be signed.

    Parameters
    ----------
    y : ndarray (signed)
        1D amplitude data to search for peaks.
    thres : float between [0., 1.]
        Normalized threshold. Only the peaks with amplitude higher than the
        threshold will be detected.
    min_dist : int
        Minimum distance between each detected peak. The peak with the highest
        amplitude is preferred to satisfy this constraint.

    Returns
    -------
    ndarray
        Array containing the indexes of the peaks that were detected
        
    '''
    if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.unsignedinteger):
        raise ValueError("y must be signed")

    thres = thres * (np.max(y) - np.min(y)) + np.min(y)
    min_dist = int(min_dist)

    # find the peaks by using the first order difference
    dy = np.diff(y)
    peaks = np.where((np.hstack([dy, 0.]) <= 0.)    #Addition by yaroslav.gevorkov@desy.de: instead of <, here <=
                     & (np.hstack([0., dy]) >= 0.)
                     & (y > thres))[0]

    if peaks.size > 1 and min_dist > 1:
        highest = peaks[np.argsort(y[peaks])][::-1]
        rem = np.ones(y.size, dtype=bool)
        rem[peaks] = False

        for peak in highest:
            if not rem[peak]:
                sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                rem[sl] = True
                rem[peak] = False

        peaks = np.arange(y.size)[~rem]

    return peaks

def getOnePhotonAdcCountsXRayTubeData(analog, applyLowpass=True, localityRadius=801, lwopassSamplePointsCount=1000):
    if applyLowpass:
        (photonHistoramValues, photonHistogramBins) = getPhotonHistogramLowpassCorrected(analog, localityRadius, lwopassSamplePointsCount)
    else:
        photonHistogramBins = np.arange(np.min(analog), np.max(analog), 1, dtype='int16')
        (photonHistoramValues, _) = np.histogram(analog, photonHistogramBins)

    smoothWindowSize = 11
    photonHistogramValuesSmooth = convolve(photonHistoramValues, np.ones((smoothWindowSize,)), mode='same')
    # plt.plot(photonHistogramBins[0:-1], photonHistogramValuesSmooth)
    # plt.show()


    # plt.figure(4)
    # plt.plot(photonHistogramBins[0:-1],photonHistoramValues)
    # plt.show()

    x = np.arange(len(photonHistoramValues))
    y = photonHistogramValuesSmooth #photonHistoramValues
    roughPeakLocations = indexes_peakutilsManuallyAdjusted(y, thres=0.05, min_dist=50)

    peakWidth = 21
    try:
        peakLocations = peakutils.interpolate(x, y, ind=roughPeakLocations, width=peakWidth, func=manual_gaussian_fit)
    except:
        return (0, 0)
    if peakLocations.size < 2:
        return (0, 0)

    peakLocations = np.clip(peakLocations, 0, len(photonHistoramValues) - 1)


    peakIndices = np.round(peakLocations).astype(int)

    peakSizes = photonHistogramValuesSmooth[peakIndices]
    sizeSortIndices = np.argsort(peakSizes)[::-1]
    sizeSortedPeakLocations = peakLocations[sizeSortIndices]
    sizeSortedPeakIndices = peakIndices[sizeSortIndices]
    sizeSortPeakSizes = peakSizes[sizeSortIndices]

    onePhotonAdcCounts = np.abs(sizeSortedPeakLocations[1] - sizeSortedPeakLocations[0])

    if onePhotonAdcCounts <= 10:
        return (0, 0)

    indicesBetweenPeaks = np.sort(sizeSortedPeakIndices[0:2])
    valleyDepthBetweenPeaks = sizeSortPeakSizes[1] - np.min(photonHistogramValuesSmooth[indicesBetweenPeaks[0]:indicesBetweenPeaks[1]])

    return (onePhotonAdcCounts, valleyDepthBetweenPeaks)
