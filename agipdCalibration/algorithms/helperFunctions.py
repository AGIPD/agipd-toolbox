import numpy as np


def orderFilterCourse(data, windowRadius, order, lowpassSamplePointsCount):
    if data.ndim != 1:
        raise ValueError('data must be one dimensional')

    if windowRadius%2 == 0:
        windowRadius -= 1

    samplePositions = np.round(np.linspace(windowRadius, data.size - windowRadius, lowpassSamplePointsCount)).astype(int)
    sampleValues = np.zeros((samplePositions.size,))
    for i in np.arange(samplePositions.size):
        neighbourhood = data[(samplePositions[i] - windowRadius):(samplePositions[i] + windowRadius + 1)]
        sampleValues[i] = np.sort(neighbourhood)[order]

    lowPassData = np.interp(np.arange(data.size), samplePositions, sampleValues, left = sampleValues[0], right = sampleValues[-1])
    return lowPassData