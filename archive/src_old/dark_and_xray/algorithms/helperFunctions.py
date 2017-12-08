import numpy as np


# fast course order filter.
# Instead of taking 2x windowRadius values, take only lowpassSamplePointsCount
# and thus speed up the computation
def orderFilterCourse(data, windowRadius, order, lowpassSamplePointsCount):
    if data.ndim != 1:
        raise ValueError('data must be one dimensional')

    if windowRadius % 2 == 0:
        windowRadius -= 1

    samplePositions = (
        np.round(np.linspace(windowRadius,
                             data.size - windowRadius,
                             lowpassSamplePointsCount)).astype(int))
    sampleValues = np.zeros((samplePositions.size,))

    for i in np.arange(samplePositions.size):
        idx = slice((samplePositions[i] - windowRadius),
                    (samplePositions[i] + windowRadius + 1))
        neighbourhood = data[idx]
        sampleValues[i] = np.sort(neighbourhood)[order]

    lowPassData = np.interp(np.arange(data.size),
                            samplePositions,
                            sampleValues,
                            left=sampleValues[0], right=sampleValues[-1])
    return lowPassData
