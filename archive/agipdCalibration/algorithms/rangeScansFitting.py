import numpy as np
from scipy.signal import convolve
from scipy.signal import medfilt


# import matplotlib.pyplot as plt

# clamped gain algorithm, not yet debugged
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

    # catch situations where not sufficient data is available
    if gainIndices[0].size < 5 or gainIndices[1].size < 30 or gainIndices[2].size < 30:
        return fitLineParameters, analogFitStdDevs

    # find saturation to get valid low gain samples
    analogSaturationValue = np.median(analog[gainIndices[1][-6:-1]])
    gainIndices[2] = gainIndices[2][analog[gainIndices[2]] < analogSaturationValue]

    # catch situations where not sufficient data is available
    if gainIndices[2].size < 30:
        return fitLineParameters, analogFitStdDevs

    # time axis is not homogeneous
    equalizedXAxis = np.hstack((np.arange(3, 203), np.arange(203, 200 + (analog.size - 200) * 10, 10))).astype('float32')

    linearFitDataShrinkFactor = 0.15
    shrinkedGainIndices = []
    digitalStdDevs = []
    for i in np.arange(3):
        dataCount = gainIndices[i].size

        cutoffRank = int(np.floor(linearFitDataShrinkFactor * dataCount))
        shrinkedGainIndices.append(gainIndices[i][cutoffRank:-cutoffRank])

        digitalStdDevs.append(np.std(digital[gainIndices[i]]))

    # print(shrinkedGainIndices)

    # remove spikes from analog values
    maxSpikeWidth = 2
    minSpkieHeight = 200
    analog = removeSpikes(analog, maxSpikeWidth, minSpkieHeight)

    # only center part of the data is used for fitting the line. Outer linearFitDataShrinkFactor - part is thrown away
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

    # # debug output
    # import matplotlib.pyplot as plt
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
    # Preset values
    fitLineParameters = [np.array([0, 0]), np.array([0, 0]), np.array([0, 0])]
    digitalMeanValues = np.array([0, 0, 0])
    analogFitStdDevs = np.array([float('inf'), float('inf'), float('inf')])
    (digitalStdDev_highGain, digitalStdDev_mediumGain, digitalStdDev_lowGain) = np.array([float('inf'), float('inf'), float('inf')])

    try:
        # remove spikes from digital values
        maxSpikeWidth = 2
        minSpkieHeight = 200
        digital = removeSpikes(digital, maxSpikeWidth, minSpkieHeight)

        # compute digital mean values
        digitalMeanValues = get3DigitalMeans_diffFilter_lowGainExtrapolated(digital)
    except ValueError:
        return fitLineParameters, digitalMeanValues, analogFitStdDevs, (digitalStdDev_highGain, digitalStdDev_mediumGain, digitalStdDev_lowGain)

    thresholds = (np.mean(digitalMeanValues[0:2]), np.mean(digitalMeanValues[1:3]))

    gainIndices = [np.array(np.nonzero(digital < thresholds[0])[0]),
                   np.array(np.nonzero((thresholds[0] < digital) & (digital < thresholds[1]))[0]),
                   np.array(np.nonzero(thresholds[1] < digital)[0])]

    # catch situations where not sufficient data is available
    if gainIndices[0].size < 7 or gainIndices[1].size < 30 or gainIndices[2].size < 30:
        return fitLineParameters, digitalMeanValues, analogFitStdDevs, (digitalStdDev_highGain, digitalStdDev_mediumGain, digitalStdDev_lowGain)

    # find saturation to get valid low gain samples
    analogSaturationValue = np.median(analog[gainIndices[1][-6:-1]])
    gainIndices[2] = gainIndices[2][analog[gainIndices[2]] < analogSaturationValue]

    # catch situations where not sufficient data is available
    if gainIndices[2].size < 30:
        return fitLineParameters, digitalMeanValues, analogFitStdDevs, (digitalStdDev_highGain, digitalStdDev_mediumGain, digitalStdDev_lowGain)

    # time axis is not homogeneous
    equalizedXAxis = np.hstack((np.arange(3, 203), np.arange(203, 200 + (analog.size - 200) * 10, 10))).astype('float32')

    # only center part of the data is used for fitting the line. Outer linearFitDataShrinkFactor - part is thrown away
    linearFitDataShrinkFactor = 0.15
    shrinkedGainIndices = []
    digitalStdDevs = []
    for i in np.arange(0, len(gainIndices)):
        dataCount = gainIndices[i].size

        cutoffRank = int(np.floor(linearFitDataShrinkFactor * dataCount))
        shrinkedGainIndices.append(gainIndices[i][cutoffRank:-cutoffRank])

        digitalStdDevs.append(np.std(digital[gainIndices[i]]))

    # print(shrinkedGainIndices)

    # remove spikes from analog values
    maxSpikeWidth = 2
    minSpkieHeight = 200
    analog = removeSpikes(analog, maxSpikeWidth, minSpkieHeight)

    # compute liniear fits by linear regression
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

    # # debug output
    # import matplotlib.pyplot as plt
    # plt.hist(np.abs(analog[shrinkedGainIndices[i]] - np.polyval(fitLineParameters[i], shrinkedGainIndices[i])))

    # # debug output
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


# compute digital means by finding the swictching indices (by finding that are maxima in the derivative)
def get3DigitalMeans_diffFilter_lowGainExtrapolated(digital):
    derivative = np.diff(digital)
    derivative[0] = 0  # first sample usually bad
    switchingIndicesCandidates = np.where(derivative > 350)
    switchingIndicesCandidates_sorted = np.sort(switchingIndicesCandidates[0])

    if switchingIndicesCandidates_sorted.size < 2:
        raise ValueError

    # compute switching indices that have a meaningful distance
    switchingIndices = np.zeros(2, dtype=int)
    switchingIndices[0] = switchingIndicesCandidates_sorted[0]

    minSamplesCountInMediumGain = 30
    for i in np.arange(1, switchingIndicesCandidates_sorted.size):
        if abs(switchingIndices[0] - switchingIndicesCandidates_sorted[i]) >= minSamplesCountInMediumGain:
            switchingIndices[1] = switchingIndicesCandidates_sorted[i]
            break

    # if not enough values for proper estimation of the mean
    if switchingIndices[0] <= 7:
        raise ValueError

    switchingIndices.sort()

    means = np.zeros(3)
    means[0] = np.mean(digital[2:switchingIndices[0] - 3])
    means[1] = np.mean(digital[switchingIndices[0] + 3:switchingIndices[1] - 3])
    means[2] = means[1] + (means[1] - means[0]) * 0.8  # extrapolation

    return means


# remove spikes by replacing them with the local median
def removeSpikes(data, maxSpikeWidth, minSpkieHeight):
    medianFiltered = medfilt(data, 2 * maxSpikeWidth + 1)

    spikeBoolIndices = abs(medianFiltered - data) >= minSpkieHeight
    data[spikeBoolIndices] = medianFiltered[spikeBoolIndices]

    return data
