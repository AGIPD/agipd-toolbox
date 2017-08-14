import numpy as np
from scipy.signal import convolve
from scipy.signal import find_peaks_cwt

import matplotlib.pyplot as plt


def getDarkCalInSitu(analog):
    darkThrowAwayPercentile = 0.03
    brightThrowAwayPercentile = 0.15
    inspectionRange = (np.round(darkThrowAwayPercentile*analog.size).astype(int), np.round((1-brightThrowAwayPercentile)*analog.size).astype(int))


    analog_cleaned = np.sort(analog)[inspectionRange[0]:inspectionRange[1]]

    analogHistogramEdges = np.arange(np.min(analog_cleaned), np.max(analog_cleaned))
    (analogHistogramValues, _) = np.histogram(analog, analogHistogramEdges)

    # plt.plot(analogHistogramEdges[0:-1], analogHistogramValues)
    # plt.show()

    smoothWindowRange = (20, 50)
    smoothWindowStep = 4
    peakIndices = np.array(
        find_peaks_cwt(analogHistogramValues, np.arange(smoothWindowRange[0], smoothWindowRange[1], smoothWindowStep)))

    smoothWindowSize = np.round(smoothWindowRange[0] + 0.5 * (smoothWindowRange[1] - smoothWindowRange[0])).astype(int)
    analogHistogramValuesSmooth = convolve(analogHistogramValues, np.ones((smoothWindowSize,)), mode='same')
    # plt.plot(analogHistogramEdges[0:-1], analogHistogramValuesSmooth)
    # plt.show()

    if peakIndices.size == 0:
        return 0

    peakSizes = analogHistogramValuesSmooth[peakIndices]

    #leftmost peak, that is big enough, is dark peak. assumption: low average illumination
    threshold = 0.5*peakSizes.max()
    for i in np.arange(peakIndices.size):
        darkOffset = peakSizes[i]
        if darkOffset > threshold:
            return analogHistogramEdges[peakIndices[i]]


