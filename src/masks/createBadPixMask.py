"""
Create bad pixel/cell masks based on general quality factors

badCellMask: overall mask (logical OR of all sub-masks)

Note: when no pixels/cells are masked, plotting throws an exception!

"""

import h5py
import numpy as np
import os

import matplotlib.pyplot as plt
import pyqtgraph as pg


moduleID = 'M314'
moduleNumber = 'm7'
temperature = 'temperature_m15C'
itestc = 'itestc150'
tint = 'tint150ns'
element = 'Cu'
baseDir = "/gpfs/cfel/fsds/labs/agipd/calibration/processed/{}/{}".format(moduleID, temperature)

# Input files
# Dynamic Range Scan (gain - current source)
anaGainsFileName = "analogGains_{}_{}.h5".format(moduleID, moduleNumber)
analogGainsFileName = os.path.join(baseDir, 'drscs', itestc, anaGainsFileName)
digiGainsFileName = "digitalMeans_{}_{}.h5".format(moduleID, moduleNumber)
digitalMeansFileName = os.path.join(baseDir, 'drscs', itestc, digiGainsFileName)
# Darks
darkFileName = "darkOffset_{}_{}_{}.h5".format(moduleID, moduleNumber, tint)
darkOffsetFileName = os.path.join(baseDir, 'dark', darkFileName)
# X-ray
xrayFileName = "photonSpacing_{}_{}_xray_{}.h5".format(moduleID, moduleNumber, element)
photonSpacingFileName = os.path.join(baseDir, 'xray', xrayFileName)


# Output mask file
saveFileName = os.path.join(baseDir, 'mask_{}_{}_{}_{}_{}_{}.h5'.format(moduleID, moduleNumber, temperature, itestc, tint, element))

# Create output file
saveFile = h5py.File(saveFileName, "w", libver='latest')
dset_badCellMask = saveFile.create_dataset("badCellMask", shape=(352, 128, 512), dtype=bool)
subMasks = saveFile.create_group("subMasks")

# Get data from gain files
analogGainsFile = h5py.File(analogGainsFileName, 'r', libver='latest')
analogGains = analogGainsFile["/analogGains"][...]  # shape=(3, 352, 128, 512)
analogLineOffsets = analogGainsFile["/anlogLineOffsets"][...]  # shape=(3, 352, 128, 512)
analogFitStdDevs = analogGainsFile["/analogFitStdDevs"][...]  # shape=(3, 352, 128, 512)
analogGainsFile.close()

digitalMeansFile = h5py.File(digitalMeansFileName, 'r', libver='latest')
digitalMeans = digitalMeansFile["/digitalMeans"][...]  # shape=(352, 3, 128, 512)
digitalThresholds = digitalMeansFile["/digitalThresholds"][...]  # shape=(2, 352, 128, 512)
digitalStdDeviations = digitalMeansFile["/digitalStdDeviations"][...]  # shape=(352, 3, 128, 512)
digitalStdDeviations = digitalStdDeviations.transpose((1, 0, 2, 3))  # shape=(3, 352, 128, 512)
digitalSpacingsSafetyFactors = digitalMeansFile["/digitalSpacingsSafetyFactors"][...]  # shape=(352, 2, 128, 512)
digitalSpacingsSafetyFactors.transpose((1,0,2,3)) # shape=(2, 352, 128, 512)
digitalMeans = digitalMeans.transpose((1, 0, 2, 3))  # shape=(3, 352, 128, 512)
digitalMeansFile.close()

# Get data from dark file
darkOffsetFile = h5py.File(darkOffsetFileName, 'r', libver='latest')
darkOffset = darkOffsetFile["/darkOffset"][...]  # shape=(352, 128, 512)
darkStandardDeviation = darkOffsetFile["/darkStandardDeviation"][...]  # shape=(352, 128, 512)
darkOffsetFile.close()

# Get data from xray file
photonSpacingFile = h5py.File(photonSpacingFileName, 'r', libver='latest')
photonSpacing = photonSpacingFile["/photonSpacing"][...]  # shape=(128, 512)
photonSpacingQuality = photonSpacingFile["/quality"][...]  # shape=(128, 512)
photonSpacingFile.close()

########### Cut values #################################################
#if moduleNumber == 6:
darkOffsetRange = np.array([2000, 9000])
darkStandardDeviationRange = np.array([2, 40])
analogFitStdDevsRange = np.array([[0, 150], [0, 1250], [0, 550]])
analogGainsRange = np.array([[20, 400], [0.45, 10.6], [0.01, 2]])
analogLineOffsetsRange = np.array([[2950, 5250], [4300, 6200], [0, 0]])
digitalStdDeviationsRange = np.array([[0, 80], [0, 35], [0, 0]])
digitalSpacingsSafetyFactorsMin = np.array([9, 0])
digitalMeansRange = np.array([[5580, 7000], [6740, 8000], [0, 0]])
digitalThresholdsRange = np.array([[6000, 7600], [0, 0]])
photonSpacingRange = np.array([50, 200])
photonSpacingQualityMin = 0
########################################################################



##################### the rest can be left untouched ###################

badCellMask = np.zeros((352, 128, 512), dtype=bool)


########################################################################
failedFitMask = np.zeros((352, 128, 512), dtype=bool)
dset_failedFitMask = subMasks.create_dataset("failedFitMask", shape=(352, 128, 512), dtype=bool)

for tmp in (analogGains, analogLineOffsets, analogFitStdDevs, digitalMeans, digitalThresholds, digitalStdDeviations):
    for i in np.arange(tmp.shape[0]):
        failedFitMask = np.logical_or(failedFitMask, ~np.isfinite(tmp[i, ...]))
    tmp[~np.isfinite(tmp)] = 0  # to surpress "invalid value in comparison" warnings
for tmp in (darkOffset, darkStandardDeviation, photonSpacing, photonSpacingQuality):
    failedFitMask = np.logical_or(failedFitMask, ~np.isfinite(tmp))
    tmp[~np.isfinite(tmp)] = 0  # to surpress "invalid value in comparison" warnings

figure = plt.figure()
axes = figure.gca()
#figure.show()

dset_failedFitMask[...] = failedFitMask
badCellMask = np.logical_or(badCellMask, failedFitMask)
print('\n\n failed fit percentage of masked cells: ', 100 * failedFitMask.flatten().sum() / failedFitMask.size)
########################################################################


########################################################################
if not ('darkOffsetRange' in locals()):
    darkOffsetRange = np.zeros((2,))
    axes.clear()
    axes.hist(darkOffset[~badCellMask], bins='sqrt')
    figure.canvas.draw()
    darkOffsetRange[0] = float(input('minimum darkOffset = '))
    darkOffsetRange[1] = float(input('maximum darkOffset = '))

darkOffset = darkOffset.transpose(1, 2, 0)  # shape = (128, 512, 352)
darkOffset = darkOffset.reshape([128, 512, 11, 32])

darkOffsetMask = np.zeros((128, 512, 11, 32), dtype=bool)
dset_darkOffsetMask = subMasks.create_dataset("darkOffsetMask", shape=(352, 128, 512), dtype=bool)

darkOffsetMask[darkOffset < darkOffsetRange[0]] = True
darkOffsetMask[darkOffset > darkOffsetRange[1]] = True
for y in np.arange(128):
    for x in np.arange(512):
        lineMediansOffset = np.median(darkOffset[y, x, ...], axis=1)
        tmp = np.median(lineMediansOffset)
        if tmp > darkOffsetRange[1] or tmp < darkOffsetRange[0]:
            darkOffsetMask[y, x, ...] = True
        else:
            for line in np.arange(11):
                if lineMediansOffset[line] > darkOffsetRange[1] or lineMediansOffset[line] < darkOffsetRange[0]:
                    darkOffsetMask[y, x, line, :] = True

darkOffsetMask = darkOffsetMask.reshape((128, 512, 352)).transpose(2, 0, 1)
dset_darkOffsetMask[...] = darkOffsetMask
badCellMask = np.logical_or.reduce((badCellMask, darkOffsetMask))
print('\n\n dark offset percentage of masked cells: ', 100 * darkOffsetMask.flatten().sum() / darkOffsetMask.size)
########################################################################


########################################################################
if not ('darkStandardDeviationRange' in locals()):
    darkStandardDeviationRange = np.zeros((2,))
    axes.clear()
    axes.hist(darkStandardDeviation[~badCellMask], bins='sqrt')
    figure.canvas.draw()
    darkStandardDeviationRange[0] = float(input('minimum darkStandardDeviation = '))
    darkStandardDeviationRange[1] = float(input('maximum darkStandardDeviation = '))

darkStandardDeviation = darkStandardDeviation.transpose(1, 2, 0)  # shape = (128, 512, 352)
darkStandardDeviation = darkStandardDeviation.reshape([128, 512, 11, 32])
darkStdDevMask = np.zeros((128, 512, 11, 32), dtype=bool)
dset_darkStdDevMask = subMasks.create_dataset("darkStdDevMask", shape=(352, 128, 512), dtype=bool)

darkStdDevMask[darkStandardDeviation < darkStandardDeviationRange[0]] = True
darkStdDevMask[darkStandardDeviation > darkStandardDeviationRange[1]] = True
for y in np.arange(128):
    for x in np.arange(512):
        lineMediansStdDev = np.median(darkStandardDeviation[y, x, ...], axis=1)
        tmp = np.median(lineMediansStdDev)
        if tmp > darkStandardDeviationRange[1] or tmp < darkStandardDeviationRange[0]:
            darkStdDevMask[y, x, ...] = True
        else:
            for line in np.arange(11):
                if lineMediansStdDev[line] > darkStandardDeviationRange[1] or lineMediansStdDev[line] < darkStandardDeviationRange[0]:
                    darkStdDevMask[y, x, line, :] = True

darkStdDevMask = darkStdDevMask.reshape((128, 512, 352)).transpose(2, 0, 1)
dset_darkStdDevMask[...] = darkStdDevMask
badCellMask = np.logical_or.reduce((badCellMask, darkStdDevMask))
print('\n\n dark std percentage of masked cells: ', 100 * darkStdDevMask.flatten().sum() / darkStdDevMask.size)
########################################################################


########################################################################
if not ('analogFitStdDevsRange' in locals()):
    analogFitStdDevsRange = np.zeros((3, 2))
    axes.hist(analogFitStdDevs[0, ~badCellMask], bins='sqrt')
    figure.canvas.draw()
    analogFitStdDevsRange[0, 1] = float(input('maximum high gain analogFitStdDevs = '))
    axes.clear()
    axes.hist(analogFitStdDevs[1, ~badCellMask], bins='sqrt')
    figure.canvas.draw()
    analogFitStdDevsRange[1, 1] = float(input('maximum medium gain analogFitStdDevs = '))
    axes.clear()
    axes.hist(analogFitStdDevs[2, ~badCellMask], bins='sqrt')
    figure.canvas.draw()
    analogFitStdDevsRange[2, 1] = float(input('maximum low gain analogFitStdDevs = '))

anaFitStdDevsMask = np.zeros((3, 352, 128, 512), dtype=bool)
dset_anaFitStdDevMask = subMasks.create_dataset("anaFitStdDevMask", shape=(3, 352, 128, 512), dtype=bool)
anaFitStdDevsMask[
    0, ~np.all((analogFitStdDevsRange[0, 0] <= analogFitStdDevs[0, ...], analogFitStdDevs[0, ...] <= analogFitStdDevsRange[0, 1]), axis=0)] = True
anaFitStdDevsMask[
    1, ~np.all((analogFitStdDevsRange[1, 0] <= analogFitStdDevs[1, ...], analogFitStdDevs[1, ...] <= analogFitStdDevsRange[1, 1]), axis=0)] = True
anaFitStdDevsMask[
    2, ~np.all((analogFitStdDevsRange[2, 0] <= analogFitStdDevs[2, ...], analogFitStdDevs[2, ...] <= analogFitStdDevsRange[2, 1]), axis=0)] = True
dset_anaFitStdDevMask[...] = anaFitStdDevsMask
badCellMask = np.logical_or.reduce((badCellMask, anaFitStdDevsMask[0, ...], anaFitStdDevsMask[1, ...], anaFitStdDevsMask[2, ...]))
print('\n\n analog fit std dev percentage of masked cells: ', 100 * anaFitStdDevsMask.flatten().sum() / anaFitStdDevsMask.size)
########################################################################


########################################################################
if not ('analogGainsRange' in locals()):
    analogGainsRange = np.zeros((3, 2))
    axes.clear()
    axes.hist(analogGains[0, ~badCellMask], bins='sqrt')
    figure.canvas.draw()
    analogGainsRange[0, 0] = float(input('minimum high gain = '))
    analogGainsRange[0, 1] = float(input('maximum high gain = '))
    axes.clear()
    axes.hist(analogGains[1, ~badCellMask], bins='sqrt')
    figure.canvas.draw()
    analogGainsRange[1, 0] = float(input('minimum medium gain = '))
    analogGainsRange[1, 1] = float(input('maximum medium gain = '))
    axes.clear()
    axes.hist(analogGains[2, ~badCellMask], bins='sqrt')
    figure.canvas.draw()
    analogGainsRange[2, 0] = float(input('minimum low gain = '))
    analogGainsRange[2, 1] = float(input('maximum low gain = '))

anaGainsMask = np.zeros((3, 352, 128, 512), dtype=bool)
dset_anaGainsMask = subMasks.create_dataset("anaGainsMask", shape=(3, 352, 128, 512), dtype=bool)
anaGainsMask[0, ~np.all((analogGainsRange[0, 0] <= analogGains[0, ...], analogGains[0, ...] <= analogGainsRange[0, 1]), axis=0)] = True
anaGainsMask[1, ~np.all((analogGainsRange[1, 0] <= analogGains[1, ...], analogGains[1, ...] <= analogGainsRange[1, 1]), axis=0)] = True
anaGainsMask[2, ~np.all((analogGainsRange[2, 0] <= analogGains[2, ...], analogGains[2, ...] <= analogGainsRange[2, 1]), axis=0)] = True

dset_anaGainsMask[...] = anaGainsMask
badCellMask = np.logical_or.reduce((badCellMask, anaGainsMask[0, ...], anaGainsMask[1, ...], anaGainsMask[2, ...]))
print('\n\n gain range percentage of masked cells: ', 100 * anaGainsMask.flatten().sum() / anaGainsMask.size)
########################################################################


########################################################################
if not ('analogLineOffsetsRange' in locals()):
    analogLineOffsetsRange = np.zeros((3, 2))
    axes.clear()
    axes.hist(analogLineOffsets[0, ~badCellMask], bins='sqrt')
    figure.canvas.draw()
    analogLineOffsetsRange[0, 0] = float(input('minimum high gain lineOffset = '))
    analogLineOffsetsRange[0, 1] = float(input('maximum high gain lineOffset = '))
    axes.clear()
    axes.hist(analogLineOffsets[1, ~badCellMask], bins='sqrt')
    figure.canvas.draw()
    analogLineOffsetsRange[1, 0] = float(input('minimum medium gain lineOffset = '))
    analogLineOffsetsRange[1, 1] = float(input('maximum medium gain lineOffset = '))
    axes.clear()
    axes.hist(analogLineOffsets[2, ~badCellMask], bins='sqrt')
    figure.canvas.draw()
    analogLineOffsetsRange[2, 0] = float(input('minimum low gain lineOffset = '))
    analogLineOffsetsRange[2, 1] = float(input('maximum low gain lineOffset = '))

anaOffsetMask = np.zeros((3, 352, 128, 512), dtype=bool)
dset_anaOffsetMask = subMasks.create_dataset("anaOffsetMask", shape=(3, 352, 128, 512), dtype=bool)
anaOffsetMask[
    0, ~np.all((analogLineOffsetsRange[0, 0] <= analogLineOffsets[0, ...], analogLineOffsets[0, ...] <= analogLineOffsetsRange[0, 1]), axis=0)] = True
anaOffsetMask[
    1, ~np.all((analogLineOffsetsRange[1, 0] <= analogLineOffsets[1, ...], analogLineOffsets[1, ...] <= analogLineOffsetsRange[1, 1]), axis=0)] = True
anaOffsetMask[
    2, ~np.all((analogLineOffsetsRange[2, 0] <= analogLineOffsets[2, ...], analogLineOffsets[2, ...] <= analogLineOffsetsRange[2, 1]), axis=0)] = True

dset_anaOffsetMask[...] = anaOffsetMask
badCellMask = np.logical_or.reduce((badCellMask, anaOffsetMask[0, ...], anaOffsetMask[1, ...], anaOffsetMask[2, ...]))
print('\n\n gain offset percentage of masked cells: ', 100 * anaOffsetMask.flatten().sum() / anaOffsetMask.size)
########################################################################


########################################################################
if not ('digitalStdDeviationsRange' in locals()):
    digitalStdDeviationsRange = np.zeros((3, 2))
    axes.clear()
    axes.hist(digitalStdDeviations[0, ~badCellMask], bins='sqrt')
    figure.canvas.draw()
    digitalStdDeviationsRange[0, 1] = float(input('maximum high gain digitalStdDeviations = '))
    axes.clear()
    axes.hist(digitalStdDeviations[1, ~badCellMask], bins='sqrt')
    figure.canvas.draw()
    digitalStdDeviationsRange[1, 1] = float(input('maximum medium gain digitalStdDeviations = '))
    axes.clear()
    axes.hist(digitalStdDeviations[2, ~badCellMask], bins='sqrt')
    figure.canvas.draw()
    digitalStdDeviationsRange[2, 1] = float(input('maximum low gain digitalStdDeviations = '))

digStdDevMask = np.zeros((3, 352, 128, 512), dtype=bool)
dset_digStdDevMask = subMasks.create_dataset("digStdDevMask", shape=(3, 352, 128, 512), dtype=bool)
digStdDevMask[
    0, ~np.all((digitalStdDeviationsRange[0, 0] <= digitalStdDeviations[0, ...], digitalStdDeviations[0, ...] <= digitalStdDeviationsRange[0, 1]),
               axis=0)] = True
digStdDevMask[
    1, ~np.all((digitalStdDeviationsRange[1, 0] <= digitalStdDeviations[1, ...], digitalStdDeviations[1, ...] <= digitalStdDeviationsRange[1, 1]),
               axis=0)] = True
digStdDevMask[
    2, ~np.all((digitalStdDeviationsRange[2, 0] <= digitalStdDeviations[2, ...], digitalStdDeviations[2, ...] <= digitalStdDeviationsRange[2, 1]),
               axis=0)] = True

dset_digStdDevMask[...] = digStdDevMask
badCellMask = np.logical_or.reduce((badCellMask, digStdDevMask[0, ...], digStdDevMask[1, ...], digStdDevMask[2, ...]))
print('\n\n dig std percentage of masked cells: ', 100 * digStdDevMask.flatten().sum() / digStdDevMask.size)
########################################################################


########################################################################
if not ('digitalSpacingsSafetyFactorsMin' in locals()):
    digitalSpacingsSafetyFactorsMin = np.zeros((2,))
    axes.clear()
    axes.hist(digitalSpacingsSafetyFactors[0, ~badCellMask], bins='sqrt')
    figure.canvas.draw()
    digitalSpacingsSafetyFactorsMin[0] = float(input('minimum high-medium digitalSpacingsSafetyFactors = '))
    axes.clear()
    axes.hist(digitalSpacingsSafetyFactors[1, ~badCellMask], bins='sqrt')
    figure.canvas.draw()
    digitalSpacingsSafetyFactorsMin[1] = float(input('minimum medium-low digitalSpacingsSafetyFactors = '))

digSpacingMask = np.zeros((2, 352, 128, 512), dtype=bool)
dset_digSpacingMask = subMasks.create_dataset("digSpacingMask", shape=(2, 352, 128, 512), dtype=bool)
digSpacingMask[0, digitalSpacingsSafetyFactors[0, ...] < digitalSpacingsSafetyFactorsMin[0]] = True
digSpacingMask[1, digitalSpacingsSafetyFactors[1, ...] < digitalSpacingsSafetyFactorsMin[1]] = True

dset_digSpacingMask[...] = digSpacingMask
badCellMask = np.logical_or.reduce((badCellMask, digSpacingMask[0, ...], digSpacingMask[1, ...]))
print('\n\n ph. spacing safety factor percentage of masked cells: ', 100 * digSpacingMask.flatten().sum() / digSpacingMask.size)
########################################################################


########################################################################
if not ('digitalMeansRange' in locals()):
    digitalMeansRange = np.zeros((3, 2))
    axes.clear()
    axes.hist(digitalMeans[0, ~badCellMask], bins='sqrt')
    figure.canvas.draw()
    digitalMeansRange[0, 0] = float(input('minimum high gain digitalMeans = '))
    digitalMeansRange[0, 1] = float(input('maximum high gain digitalMeans = '))
    axes.clear()
    axes.hist(digitalMeans[1, ~badCellMask], bins='sqrt')
    figure.canvas.draw()
    digitalMeansRange[1, 0] = float(input('minimum medium gain digitalMeans = '))
    digitalMeansRange[1, 1] = float(input('maximum medium gain digitalMeans = '))
    axes.clear()
    axes.hist(digitalMeans[2, ~badCellMask], bins='sqrt')
    figure.canvas.draw()
    digitalMeansRange[2, 0] = float(input('minimum low gain digitalMeans = '))
    digitalMeansRange[2, 1] = float(input('maximum low gain digitalMeans = '))

digMeansMask = np.zeros((3, 352, 128, 512), dtype=bool)
dset_digMeansMask = subMasks.create_dataset("digMeansMask", shape=(3, 352, 128, 512), dtype=bool)
digMeansMask[0, ~np.all((digitalMeansRange[0, 0] <= digitalMeans[0, ...], digitalMeans[0, ...] <= digitalMeansRange[0, 1]), axis=0)] = True
digMeansMask[1, ~np.all((digitalMeansRange[1, 0] <= digitalMeans[1, ...], digitalMeans[1, ...] <= digitalMeansRange[1, 1]), axis=0)] = True
digMeansMask[2, ~np.all((digitalMeansRange[2, 0] <= digitalMeans[2, ...], digitalMeans[2, ...] <= digitalMeansRange[2, 1]), axis=0)] = True

dset_digMeansMask[...] = digMeansMask
badCellMask = np.logical_or.reduce((badCellMask, digMeansMask[0, ...], digMeansMask[1, ...], digMeansMask[2, ...]))
print('\n\n dig. means percentage of masked cells: ', 100 * digMeansMask.flatten().sum() / digMeansMask.size)
########################################################################


########################################################################
if not ('digitalThresholdsRange' in locals()):
    digitalThresholdsRange = np.zeros((2, 2))
    axes.clear()
    axes.hist(digitalThresholds[0, ~badCellMask], bins=2 ** 10)
    figure.canvas.draw()
    digitalThresholdsRange[0, 0] = float(input('minimum high-medium digitalThresholds = '))
    digitalThresholdsRange[0, 1] = float(input('maximum high-medium digitalThresholds = '))
    axes.clear()
    axes.hist(digitalThresholds[1, ~badCellMask], bins='sqrt')
    figure.canvas.draw()
    digitalThresholdsRange[1, 0] = float(input('minimum medium-low digitalThresholds = '))
    digitalThresholdsRange[1, 1] = float(input('maximum medium-low digitalThresholds = '))

digThreshMask = np.zeros((2, 352, 128, 512), dtype=bool)
dset_digThreshMask = subMasks.create_dataset("digThreshMask", shape=(2, 352, 128, 512), dtype=bool)
digThreshMask[
    0, ~np.all((digitalThresholdsRange[0, 0] <= digitalThresholds[0, ...], digitalThresholds[0, ...] <= digitalThresholdsRange[0, 1]), axis=0)] = True
digThreshMask[
    1, ~np.all((digitalThresholdsRange[1, 0] <= digitalThresholds[1, ...], digitalThresholds[1, ...] <= digitalThresholdsRange[1, 1]), axis=0)] = True

dset_digThreshMask[...] = digThreshMask
badCellMask = np.logical_or.reduce((badCellMask, digThreshMask[0, ...], digThreshMask[1, ...]))
print('\n\n dig thresh percentage of masked cells: ', 100 * digThreshMask.flatten().sum() / digThreshMask.size)
########################################################################


########################################################################
if not ('photonSpacingRange' in locals()):
    photonSpacingRange = np.zeros((2,))
    axes.clear()
    axes.hist(photonSpacing[~badAsicBordersMask[0, ...]], bins=2 ** 12)
    figure.canvas.draw()
    photonSpacingRange[0] = float(input('minimum photonSpacing = '))
    photonSpacingRange[1] = float(input('maximum photonSpacing = '))

photonSpacingMask = np.zeros((128, 512), dtype=bool)
dset_photonSpacingMask = subMasks.create_dataset("photonSpacingMask", shape=(128, 512), dtype=bool)
photonSpacingMask[~np.all((photonSpacingRange[0] <= photonSpacing, photonSpacing <= photonSpacingRange[1]), axis=0)] = True

dset_photonSpacingMask[...] = photonSpacingMask
badCellMask = np.logical_or(badCellMask, photonSpacingMask)
print('\n\n ph. spacing percentage of masked cells: ', 100 * photonSpacingMask.flatten().sum() / photonSpacingMask.size)
########################################################################


########################################################################
if not ('photonSpacingQualityMin' in locals()):
    photonSpacingQualityMin = 0
    axes.clear()
    axes.hist(photonSpacingQuality[~badAsicBordersMask[0, ...]], bins='sqrt')
    figure.canvas.draw()
    photonSpacingQualityMin = float(input('minimum photonSpacingQuality = '))

photonSpacingQualityMask = np.zeros((128, 512), dtype=bool)
dset_photonSpacingQualityMask = subMasks.create_dataset("photonSpacingQualityMask", shape=(128, 512), dtype=bool)
photonSpacingQualityMask[photonSpacingQuality <= photonSpacingQualityMin] = True

dset_photonSpacingQualityMask[...] = photonSpacingQualityMask
badCellMask = np.logical_or(badCellMask, photonSpacingQualityMask)
print('\n\n ph. spacing quality percentage of masked cells: ', 100 * photonSpacingQualityMask.flatten().sum() / photonSpacingQualityMask.size)
########################################################################


dset_badCellMask[...] = badCellMask
saveFile.flush()
saveFile.close()
print('badCellMask saved in ', saveFileName)

print('\n\n\nentered values:')
print('darkOffsetRange = \n', darkOffsetRange)
print('darkStandardDeviationRange = \n', darkStandardDeviationRange)
print('analogFitStdDevsRange = \n', analogFitStdDevsRange)
print('analogGainsRange = \n', analogGainsRange)
print('analogLineOffsetsRange = \n', analogLineOffsetsRange)
print('digitalStdDeviationsRange = \n', digitalStdDeviationsRange)
print('digitalSpacingsSafetyFactorsMin = \n', digitalSpacingsSafetyFactorsMin)
print('digitalMeansRange = \n', digitalMeansRange)
print('digitalThresholdsRange = \n', digitalThresholdsRange)
print('photonSpacingRange = \n', photonSpacingRange)
print('photonSpacingQualityMin = \n', photonSpacingQualityMin)

badCellMaskFile = h5py.File(saveFileName, "r", libver='latest')
badCellMask = badCellMaskFile['badCellMask'][...]
pg.image(badCellMask.transpose(0, 2, 1))

print('\n\n percentage of masked cells: ', 100 * badCellMask.flatten().sum() / badCellMask.size)
print('\n\n\npress enter to quit')
tmp = input()
