import numpy as np
from scipy.signal import convolve


def fitScanSlopes(analog, digital):
    ##### find_peaks_cwt
    # digitalHistogramEdges = np.arange(np.min(digital), np.max(digital))
    # (digitalHistogram, _) = np.histogram(digital, digitalHistogramEdges)
    #
    # plt.plot(digitalHistogramEdges[0:-1], digitalHistogram)
    #
    # smoothWindowRange = (4, 50)  # expect peak widths to be in the range of 4-50 ADC counts
    # smoothWindowStep = 8
    # peakLocations = np.array(
    #     find_peaks_cwt(digitalHistogram, np.arange(smoothWindowRange[0], smoothWindowRange[1], smoothWindowStep)))
    #
    # peakValues = digitalHistogram[peakLocations]
    # if peakValues.size < 2:
    #     return ([], float("inf"))
    # valueSortedPeakLocations = peakLocations[np.argsort(peakValues)[::-1]]
    # digitalMeanValues = digitalHistogramEdges[valueSortedPeakLocations[0:2].astype(int)][::-1]
    #####

    ##### k-means
    # digitalMeanValues = np.array(
    #     KMeans(n_clusters=2, max_iter=5, init=np.array([[4000], [7000]]), precompute_distances=True, tol=1e-6, n_jobs=1, copy_x=False)
    #         .fit(digital.reshape(-1, 1)).cluster_centers_).astype('uint16')
    # digitalMeanValues.sort()
    # print(digitalMeanValues)
    #####


    try:
        ###### simplified k-means
        digitalMeanValues = getDigitalMeans(digital, refinementStepsCount=3, minDigitalSpacing=300)
    except ValueError:
        fitLineParameters = []
        fitLineParameters.append(np.array([0, 0]))
        fitLineParameters.append(np.array([0, 0]))
        digitalMeanValues = np.array([0, 0])
        analogFitError = np.array([float('inf'), float('inf')])
        (digitalStdDev_highGain, digitalStdDev_mediumGain) = np.array([float('inf'), float('inf')])
        return (fitLineParameters, digitalMeanValues, analogFitError, (digitalStdDev_highGain, digitalStdDev_mediumGain))

    threshold = np.mean(digitalMeanValues)

    digital_highGain = digital[digital < threshold]
    digital_mediumGain = digital[digital > threshold]

    cutoffPart = 0.02
    cutoffRank = int(np.round(cutoffPart * digital_highGain.shape[0]))
    digitalStdDev_highGain = np.std(np.sort(digital_highGain)[cutoffRank:-cutoffRank])
    cutoffRank = int(np.round(cutoffPart * digital_mediumGain.shape[0]))
    digitalStdDev_mediumGain = np.std(np.sort(digital_mediumGain)[cutoffRank:-cutoffRank])

    gainIndices = [np.array(np.nonzero(digital < threshold)[0]),
                   np.array(np.nonzero(digital >= threshold)[0])]

    shrinkFactor = 0.3
    shrinkedGainIntervalsInData = []
    for i in np.arange(0, len(gainIndices)):
        dataCount = gainIndices[i].size
        shrinkedGainIntervalsInData.append(gainIndices[i][np.array([np.floor(dataCount * shrinkFactor), np.floor(dataCount * (1 - shrinkFactor))]).astype(int)])

    # print(shrinkedGainIntervalsInData)

    fitLineParameters = []
    for i in np.arange(0, len(gainIndices)):
        if shrinkedGainIntervalsInData[i][1] - shrinkedGainIntervalsInData[i][0] < 5:  # not enough data for estimation
            fitLineParameters.append(np.array([0, 0]))
        else:
            fitLineParameters.append(np.polyfit(np.arange(shrinkedGainIntervalsInData[i][0], shrinkedGainIntervalsInData[i][1], dtype=np.float32),
                                                analog[np.arange(shrinkedGainIntervalsInData[i][0], shrinkedGainIntervalsInData[i][1], dtype=np.int)], 1))

    analogFitError = []
    for i in np.arange(len(fitLineParameters)):
        # print analog[shrinkedGainIntervalsInData[i][0]:shrinkedGainIntervalsInData[i][1]] - np.polyval(fitLineParameters[i], np.arange(shrinkedGainIntervalsInData[i][0], shrinkedGainIntervalsInData[i][1]))
        analogFitError.append(np.mean(np.abs(analog[shrinkedGainIntervalsInData[i][0]:shrinkedGainIntervalsInData[i][1]]
                                             - np.polyval(fitLineParameters[i],
                                                          np.arange(shrinkedGainIntervalsInData[i][0], shrinkedGainIntervalsInData[i][1])))))

    # plt.plot(analog)
    # plt.hold(True)
    # for i in np.arange(len(fitLineParameters)):
    #     plt.plot(gainIndices[i], np.polyval(fitLineParameters[i], gainIndices[i]), linewidth=1, color='g')
    #     plt.plot(np.arange(shrinkedGainIntervalsInData[i][0], shrinkedGainIntervalsInData[i][1]),
    #              np.polyval(fitLineParameters[i], np.arange(shrinkedGainIntervalsInData[i][0], shrinkedGainIntervalsInData[i][1])), linewidth=2, color='r')
    # print analogFitError

    return (fitLineParameters, digitalMeanValues, analogFitError, (digitalStdDev_highGain, digitalStdDev_mediumGain))


# simplified k-means
def getDigitalMeans(digital, refinementStepsCount=3, minDigitalSpacing=400):
    binWidth = np.round(minDigitalSpacing / 7)
    digitalHistogramEdges = np.arange(np.min(digital), np.max(digital), 100, dtype='int16')
    if digitalHistogramEdges.size == 0 or digitalHistogramEdges[-1] - digitalHistogramEdges[0] < minDigitalSpacing:
        raise ValueError
    (digitalHistogram, _) = np.histogram(digital, digitalHistogramEdges)

    digitalHistogramSmooth = convolve(digitalHistogram, np.ones((3,)), mode='same')

    mostFrequentValues = digitalHistogramEdges[np.argsort(digitalHistogramSmooth)[::-1]]
    means = np.array([mostFrequentValues[0], mostFrequentValues[-1]])
    for i in np.arange(1, mostFrequentValues.size):
        if means[0] - mostFrequentValues[i] >= minDigitalSpacing:
            means[1] = mostFrequentValues[i]
            break

    for _ in np.arange(refinementStepsCount):
        threshold = (means[0] + means[1]) / 2
        smallerBoolIndices = digital < threshold
        means[0] = np.mean(digital[smallerBoolIndices])
        means[1] = np.mean(digital[np.invert(smallerBoolIndices)])

    if means[1] - means[0] < minDigitalSpacing:
        raise ValueError

    return means
