import numpy as np
from scipy.signal import convolve
from scipy.signal import medfilt


# import matplotlib.pyplot as plt


def fit2DynamicScanSlopes(analog, digital):
    try:
        ###### simplified k-means
        digitalMeanValues = get2DigitalMeans(digital, refinementStepsCount=3, minDigitalSpacing=300)
    except ValueError:
        fitLineParameters = []
        fitLineParameters.append(np.array([0, 0]))
        fitLineParameters.append(np.array([0, 0]))
        digitalMeanValues = np.array([0, 0])
        analogFitStdDevs = np.array([float('inf'), float('inf')])
        (digitalStdDev_highGain, digitalStdDev_mediumGain) = np.array([float('inf'), float('inf')])
        return fitLineParameters, digitalMeanValues, analogFitStdDevs, (digitalStdDev_highGain, digitalStdDev_mediumGain)

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

    analogFitStdDevs = []
    for i in np.arange(len(fitLineParameters)):
        # print analog[shrinkedGainIntervalsInData[i][0]:shrinkedGainIntervalsInData[i][1]] - np.polyval(fitLineParameters[i], np.arange(shrinkedGainIntervalsInData[i][0], shrinkedGainIntervalsInData[i][1]))
        analogFitStdDevs.append(np.mean(np.abs(analog[shrinkedGainIntervalsInData[i][0]:shrinkedGainIntervalsInData[i][1]]
                                               - np.polyval(fitLineParameters[i],
                                                            np.arange(shrinkedGainIntervalsInData[i][0], shrinkedGainIntervalsInData[i][1])))))

    # plt.plot(analog)
    # plt.hold(True)
    # for i in np.arange(len(fitLineParameters)):
    #     plt.plot(gainIndices[i], np.polyval(fitLineParameters[i], gainIndices[i]), linewidth=1, color='g')
    #     plt.plot(np.arange(shrinkedGainIntervalsInData[i][0], shrinkedGainIntervalsInData[i][1]),
    #              np.polyval(fitLineParameters[i], np.arange(shrinkedGainIntervalsInData[i][0], shrinkedGainIntervalsInData[i][1])), linewidth=2, color='r')
    # print(analogFitStdDevs)

    return fitLineParameters, digitalMeanValues, analogFitStdDevs, (digitalStdDev_highGain, digitalStdDev_mediumGain)


# simplified k-means
def get2DigitalMeans(digital, refinementStepsCount=3, minDigitalSpacing=400):
    binWidth = 100
    digitalHistogramEdges = np.arange(np.min(digital), np.max(digital) + binWidth, binWidth, dtype='int16')
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


def fit3DynamicScanSlopes_precomputedDigitalMeans(analog, digital, digitalMeanValues):
    fitLineParameters = []
    fitLineParameters.append(np.array([0, 0]))
    fitLineParameters.append(np.array([0, 0]))
    fitLineParameters.append(np.array([0, 0]))
    analogFitStdDevs = np.array([float('inf'), float('inf'), float('inf')])

    thresholds = (np.mean(digitalMeanValues[0:2]), np.mean(digitalMeanValues[1:3]))

    gainIndices = [np.array(np.nonzero(digital < thresholds[0])[0]),
                   np.array(np.nonzero((thresholds[0] < digital) & (digital < thresholds[1]))[0]),
                   np.array(np.nonzero(thresholds[1] < digital)[0])]

    if gainIndices[0].size < 5 or gainIndices[1].size < 30 or gainIndices[2].size < 30:
        return fitLineParameters, analogFitStdDevs

    analogSaturationValue = np.median(analog[gainIndices[1][-6:-1]])
    gainIndices[2] = gainIndices[2][analog[gainIndices[2]] < analogSaturationValue]

    if gainIndices[2].size < 30:
        return fitLineParameters, analogFitStdDevs

    equalizedXAxis = np.hstack((np.arange(3, 203), np.arange(203, 200 + (analog.size - 200) * 10, 10))).astype('float32')

    linearFitDataShrinkFactor = 0.15
    shrinkedGainIndices = []
    digitalStdDevs = []
    for i in np.arange(3):
        dataCount = gainIndices[i].size

        cutoffRank = int(np.floor(linearFitDataShrinkFactor * dataCount))
        shrinkedGainIndices.append(gainIndices[i][cutoffRank:-cutoffRank])

        digitalStdDevs.append(np.std(digital[gainIndices[i]]))

    # # constant samples count for low gain
    # skipSamplesLowGain = 10
    # samplesCountLowGain = 200
    # lastLowSampleToTake = np.min((skipSamplesLowGain + samplesCountLowGain, gainIndices[2].size))
    # shrinkedGainIndices.append(gainIndices[2][skipSamplesLowGain:lastLowSampleToTake + 1])

    # print(shrinkedGainIndices)

    maxSpikeWidth = 2
    minSpkieHeight = 400
    # analog = removeSpikes(analog, maxSpikeWidth, minSpkieHeight)    # requested by Aschkan. Really needed?



    fitLineParameters = []
    analogFitStdDevs = []
    for i in np.arange(0, len(gainIndices)):
        if len(shrinkedGainIndices[i]) == 0 or shrinkedGainIndices[i][-1] - shrinkedGainIndices[i][0] < 5:  # not enough data for estimation
            fitLineParameters.append(np.array([0, 0]))
            analogFitStdDevs.append(float('inf'))
        else:
            x = equalizedXAxis[shrinkedGainIndices[i]]
            y = analog[shrinkedGainIndices[i]]
            fitLineParameters.append(np.polyfit(x, y, 1))
            analogFitStdDevs.append(np.std(analog[shrinkedGainIndices[i]] - np.polyval(fitLineParameters[i], shrinkedGainIndices[i])))

    # plt.hist(np.abs(analog[shrinkedGainIndices[i]] - np.polyval(fitLineParameters[i], shrinkedGainIndices[i])))

    # import matplotlib.pyplot as plt
    # plt.plot(equalizedXAxis, analog)
    # plt.hold(True)
    # for i in np.arange(len(fitLineParameters)):
    #     x = equalizedXAxis[shrinkedGainIndices[i]]
    #     x_full = equalizedXAxis[gainIndices[i]]
    #     plt.plot(x_full, np.polyval(fitLineParameters[i], x_full), linewidth=2, color='g')
    #     plt.plot(x, np.polyval(fitLineParameters[i], x), linewidth=2, color='r')
    # print(analogFitStdDevs)
    return fitLineParameters, analogFitStdDevs


def fit3DynamicScanSlopes(analog, digital):
    fitLineParameters = []
    fitLineParameters.append(np.array([0, 0]))
    fitLineParameters.append(np.array([0, 0]))
    fitLineParameters.append(np.array([0, 0]))
    digitalMeanValues = np.array([0, 0, 0])
    analogFitStdDevs = np.array([float('inf'), float('inf'), float('inf')])
    (digitalStdDev_highGain, digitalStdDev_mediumGain, digitalStdDev_lowGain) = np.array([float('inf'), float('inf'), float('inf')])

    try:
        #digitalMeanValues = get3DigitalMeans(digital, refinementStepsCount=3, minDigitalSpacing=800)
        digitalMeanValues = get3DigitalMeans_diffFilter_lowGainExtrapolated(digital)
        # print(digitalMeanValues)
    except ValueError:
         return fitLineParameters, digitalMeanValues, analogFitStdDevs, (digitalStdDev_highGain, digitalStdDev_mediumGain, digitalStdDev_lowGain)

    thresholds = (np.mean(digitalMeanValues[0:2]), np.mean(digitalMeanValues[1:3]))

    gainIndices = [np.array(np.nonzero(digital < thresholds[0])[0]),
                   np.array(np.nonzero((thresholds[0] < digital) & (digital < thresholds[1]))[0]),
                   np.array(np.nonzero(thresholds[1] < digital)[0])]

    if gainIndices[0].size < 5 or gainIndices[1].size < 30 or gainIndices[2].size < 30:
        return fitLineParameters, digitalMeanValues, analogFitStdDevs, (digitalStdDev_highGain, digitalStdDev_mediumGain, digitalStdDev_lowGain)

    analogSaturationValue = np.median(analog[gainIndices[1][-6:-1]])
    gainIndices[2] = gainIndices[2][analog[gainIndices[2]] < analogSaturationValue]

    if gainIndices[2].size < 30:
        return fitLineParameters, digitalMeanValues, analogFitStdDevs, (digitalStdDev_highGain, digitalStdDev_mediumGain, digitalStdDev_lowGain)

    equalizedXAxis = np.hstack((np.arange(3, 203), np.arange(203, 200 + (analog.size - 200) * 10, 10))).astype('float32')

    linearFitDataShrinkFactor = 0.15
    shrinkedGainIndices = []
    digitalStdDevs = []
    for i in np.arange(0, len(gainIndices)):
        dataCount = gainIndices[i].size

        cutoffRank = int(np.floor(linearFitDataShrinkFactor * dataCount))
        shrinkedGainIndices.append(gainIndices[i][cutoffRank:-cutoffRank])

        digitalStdDevs.append(np.std(digital[gainIndices[i]]))

    # print(shrinkedGainIndices)

    maxSpikeWidth = 2
    minSpkieHeight = 400
    # analog = removeSpikes(analog, maxSpikeWidth, minSpkieHeight)    # requested by Aschkan. Really needed?

    fitLineParameters = []
    analogFitStdDevs = []
    for i in np.arange(0, len(gainIndices)):
        if len(shrinkedGainIndices[i]) == 0 or shrinkedGainIndices[i][-1] - shrinkedGainIndices[i][0] < 5:  # not enough data for estimation
            fitLineParameters.append(np.array([0, 0]))
            analogFitStdDevs.append(float('inf'))
        else:
            x = equalizedXAxis[shrinkedGainIndices[i]]
            y = analog[shrinkedGainIndices[i]]
            fitLineParameters.append(np.polyfit(x, y, 1))
            analogFitStdDevs.append(np.std(analog[shrinkedGainIndices[i]] - np.polyval(fitLineParameters[i], shrinkedGainIndices[i])))

    # plt.hist(np.abs(analog[shrinkedGainIndices[i]] - np.polyval(fitLineParameters[i], shrinkedGainIndices[i])))

    # import matplotlib.pyplot as plt
    # plt.plot(equalizedXAxis, analog)
    # plt.hold(True)
    # for i in np.arange(len(fitLineParameters)):
    #     x = equalizedXAxis[shrinkedGainIndices[i]]
    #     x_full = equalizedXAxis[gainIndices[i]]
    #     plt.plot(x_full, np.polyval(fitLineParameters[i], x_full), linewidth=2, color='g')
    #     plt.plot(x, np.polyval(fitLineParameters[i], x), linewidth=2, color='r')
    # print(analogFitStdDevs)
    return fitLineParameters, digitalMeanValues, analogFitStdDevs, (digitalStdDevs[0], digitalStdDevs[1], digitalStdDevs[2])


# simplified k-means
def get3DigitalMeans(digital, refinementStepsCount=3, minDigitalSpacing=600):
    # binWidth = 100
    # digitalHistogramEdges = np.arange(np.min(digital), np.max(digital) + binWidth, binWidth, dtype='int16')
    # if digitalHistogramEdges.size == 0 or digitalHistogramEdges[-1] - digitalHistogramEdges[0] < 2 * minDigitalSpacing:
    #     raise ValueError
    # (digitalHistogram, _) = np.histogram(digital, digitalHistogramEdges)
    # # plt.plot(digitalHistogramEdges[0:-1], digitalHistogram)
    #
    # digitalHistogramSmooth = convolve(digitalHistogram, np.ones((3,)), mode='same')
    # minDigitalSpacing = minDigitalSpacing - 2 * binWidth
    # # plt.plot(digitalHistogramEdges[0:-1], digitalHistogramSmooth)
    #
    # mostFrequentValues = digitalHistogramEdges[np.argsort(digitalHistogramSmooth)[::-1]][0:np.count_nonzero(digitalHistogramSmooth)]
    # # print(mostFrequentValues)
    # means = np.array([mostFrequentValues[0], mostFrequentValues[-1], mostFrequentValues[-2]])
    # secondMeanFound = False
    # for i in np.arange(1, mostFrequentValues.size):
    #     if abs(means[0] - mostFrequentValues[i]) >= minDigitalSpacing:
    #         means[1] = mostFrequentValues[i]
    #         secondMeanIndex = i
    #         secondMeanFound = True
    #         break
    #
    # if not secondMeanFound or secondMeanIndex == mostFrequentValues.size - 1:
    #     raise ValueError
    #
    # thirdMeanFound = False
    # for i in np.arange(secondMeanIndex + 1, mostFrequentValues.size):
    #     if abs(means[0] - mostFrequentValues[i]) >= minDigitalSpacing and abs(means[1] - mostFrequentValues[i]) >= minDigitalSpacing:
    #         means[2] = mostFrequentValues[i]
    #         thirdMeanFound = True
    #         break
    #
    # if not thirdMeanFound:
    #     raise ValueError
    #
    # means.sort()

    min = np.median(digital[1:6])
    max = np.median(digital[-6:-1])
    means = np.array((min, (min + max) / 2, max))

    digital_perforated = np.hstack((digital[1:20], digital[21::5]))

    for _ in np.arange(refinementStepsCount):
        thresholds = ((means[0] + means[1]) / 2, (means[1] + means[2]) / 2)
        means[0] = np.mean(digital_perforated[digital_perforated < thresholds[0]])
        means[1] = np.mean(digital_perforated[(thresholds[0] < digital_perforated) & (digital_perforated < thresholds[1])])
        means[2] = np.mean(digital_perforated[thresholds[1] < digital_perforated])

    if means[1] - means[0] < minDigitalSpacing or means[2] - means[1] < minDigitalSpacing or np.isnan(means).any():
        raise ValueError

    return means


def get3DigitalMeans_diffFilter_lowGainExtrapolated(digital):
    derivative = np.diff(digital)
    derivative[0] = 0 #first sample usually bad
    switchingIndicesCandidates = np.where(derivative > 350)
    #switchingIndicesCandidates_sorted = switchingIndicesCandidates[0][np.argsort(derivative[switchingIndicesCandidates[0]])][::-1]
    switchingIndicesCandidates_sorted = np.sort(switchingIndicesCandidates[0])

    if switchingIndicesCandidates_sorted.size < 2:
        raise ValueError

    switchingIndices = np.zeros(2, dtype=int)
    switchingIndices[0] = switchingIndicesCandidates_sorted[0]

    minSamplesCountInMediumGain = 30
    for i in np.arange(1, switchingIndicesCandidates_sorted.size):
        if abs(switchingIndices[0] - switchingIndicesCandidates_sorted[i]) >= minSamplesCountInMediumGain:
            switchingIndices[1] = switchingIndicesCandidates_sorted[i]
            break

    if switchingIndices[0] <= 10 or switchingIndices[1] <= 10:
        raise ValueError

    switchingIndices.sort()

    means = np.zeros(3)
    means[0] = np.mean(digital[2:switchingIndices[0] - 3])
    means[1] = np.mean(digital[switchingIndices[0] + 3:switchingIndices[1] - 3])
    means[2] = means[1] + (means[1] - means[0])*0.8  # extrapolation

    return means


def removeSpikes(data, maxSpikeWidth, minSpkieHeight):
    medianFiltered = medfilt(data, 2 * maxSpikeWidth + 1)

    spikeBoolIndices = abs(medianFiltered - data) >= minSpkieHeight
    data[spikeBoolIndices] = medianFiltered[spikeBoolIndices]

    return data
