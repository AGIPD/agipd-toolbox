import h5py
import numpy as np

import matplotlib.pyplot as plt
import pyqtgraph as pg

analogGainsFileName = '/gpfs/cfel/cxi/scratch/user/gevorkov/agipdCalibration_workspace/analogGains_m3.h5'
digitalMeansFileName = '/gpfs/cfel/cxi/scratch/user/gevorkov/agipdCalibration_workspace/digitalMeans_m3.h5'
darkOffsetFileName = '/gpfs/cfel/cxi/scratch/user/gevorkov/agipdCalibration_workspace/lysozymeData_m3_darkcal_inSitu.h5'
photonSpacingFileName = '/gpfs/cfel/cxi/scratch/user/gevorkov/agipdCalibration_workspace/photonSpacing_m3_inSitu.h5'

saveFileName = '/gpfs/cfel/cxi/scratch/user/gevorkov/agipdCalibration_workspace/mask_m3.h5'

saveFile = h5py.File(saveFileName, "w", libver='latest')
dset_badCellMask = saveFile.create_dataset("badCellMask", shape=(352, 128, 512), dtype=bool)

analogGainsFile = h5py.File(analogGainsFileName, 'r', libver='latest')
digitalMeansFile = h5py.File(digitalMeansFileName, 'r', libver='latest')
darkOffsetFile = h5py.File(darkOffsetFileName, 'r', libver='latest')
photonSpacingFile = h5py.File(photonSpacingFileName, 'r', libver='latest')

analogGains = analogGainsFile["/analogGains"][...]  # shape=(3, 352, 128, 512)
analogLineOffsets = analogGainsFile["/anlogLineOffsets"][...]  # shape=(3, 352, 128, 512)
analogFitError = analogGainsFile["/analogFitError"][...]  # shape=(3, 352, 128, 512)

digitalMeans = digitalMeansFile["/digitalMeans"][...]  # shape=(352, 3, 128, 512)
digitalThresholds = digitalMeansFile["/digitalThresholds"][...]  # shape=(2, 352, 128, 512)
digitalStdDeviations = digitalMeansFile["/digitalStdDeviations"][...]  # shape=(352, 3, 128, 512)
digitalStdDeviations = digitalStdDeviations.transpose((1, 0, 2, 3))  # shape=(3, 352, 128, 512)
digitalMeans = digitalMeans.transpose((1, 0, 2, 3))  # shape=(3, 352, 128, 512)

darkOffset = darkOffsetFile["/darkOffset"][...]  # shape=(352, 128, 512)
# darkStandardDeviation = darkOffsetFile["/darkStandardDeviation"][...]  # shape=(352, 128, 512)

photonSpacing = photonSpacingFile["/photonSpacing"][...]  # shape=(128, 512)
# photonSpacingQuality = photonSpacingFile["/quality"][...]  # shape=(128, 512)

########### fix values
analogFitErrorRange = np.array([[0, 40], [0, 20], [0, 0]])
analogGainsRange = np.array([[25, 40], [0.8, 1.2], [0, 0]])
analogLineOffsetsRange = np.array([[2000, 5000], [4000, 6600], [0, 0]])
digitalStdDeviationsRange = np.array([[0, 160], [0, 80], [0, 0]])
digitalSpacingsSafetyFactorsMin = np.array([7, 0])
digitalMeansRange = np.array([[5000, 7000], [6000, 8000], [0, 0]])
digitalThresholdsRange = np.array([[6000, 7200], [0, 0]])
darkOffsetRange = np.array([2500, 5500])
# darkStandardDeviationRange = np.array([0, 40])
photonSpacingRange = np.array([100, 200])
# photonSpacingQualityMin = 0
###########

badCellMask = np.zeros((352, 128, 512), dtype=bool)

################# manual mask
manualMask = np.zeros((128, 512), dtype=bool)
manualMask[65:105, 200:260] = True
manualMask[80:105, 200:511] = True

manualMask[80:86, 22:43] = True
manualMask[75:100, 31:43] = True

manualMask[64:127, 127:195] = True

badCellMask = np.logical_or(manualMask, badCellMask)
###########


badAsicBordersMask = np.zeros((352, 128, 512), dtype=bool)
badAsicBordersMask[:, (0, 63, 64, 127), :] = True
for column in np.arange(8):
    badAsicBordersMask[:, :, (column * 64, column * 64 + 63)] = True

badCellMask = np.logical_or(badAsicBordersMask, badCellMask)
for tmp in (analogGains, analogLineOffsets, analogFitError, digitalMeans, digitalThresholds, digitalStdDeviations):
    for i in np.arange(tmp.shape[0]):
        badCellMask = np.logical_or(badCellMask, ~np.isfinite(tmp[i, ...]))
    tmp[~np.isfinite(tmp)] = 0  # to surpress "invalid value in comparison" warnings
# for tmp in (darkOffset, darkStandardDeviation, photonSpacing, photonSpacingQuality):
#     badCellMask = np.logical_or(badCellMask, ~np.isfinite(tmp))
#     tmp[~np.isfinite(tmp)] = 0  # to surpress "invalid value in comparison" warnings
# badPixelMask = np.logical_or.reduce((badAsicBordersMask[0, ...], ~np.isfinite(photonSpacing), ~np.isfinite(photonSpacingQuality)))
for tmp in (darkOffset, photonSpacing):  ########################################### only for in Situ! No quality measure available!
    badCellMask = np.logical_or(badCellMask, ~np.isfinite(tmp))
    tmp[~np.isfinite(tmp)] = 0  # to surpress "invalid value in comparison" warnings
badPixelMask = np.logical_or(badAsicBordersMask[0, ...], ~np.isfinite(photonSpacing))

figure = plt.figure()
axes = figure.gca()
figure.show()

if not ('analogFitErrorRange' in locals()):
    analogFitErrorRange = np.zeros((3, 2))
    axes.hist(analogFitError[0, ~badCellMask], bins='sqrt')
    figure.canvas.draw()
    analogFitErrorRange[0, 1] = float(input('maximum high gain analogFitError = '))
    axes.clear()
    axes.hist(analogFitError[1, ~badCellMask], bins='sqrt')
    figure.canvas.draw()
    analogFitErrorRange[1, 1] = float(input('maximum medium gain analogFitError = '))
    # axes.clear() axes.hist(analogFitError[2,~badCellMask])
    # plt.show(block=False) figure.canvas.draw()
    # analogFitErrorRange[2,1] = float(input('maximum low gain analogFitError = '))

badAnalogFitErrorMask = np.zeros((3, 352, 128, 512), dtype=bool)
badAnalogFitErrorMask[0, ~np.all((analogFitErrorRange[0, 0] <= analogFitError[0, ...], analogFitError[0, ...] <= analogFitErrorRange[0, 1]), axis=0)] = True
badAnalogFitErrorMask[1, ~np.all((analogFitErrorRange[1, 0] <= analogFitError[1, ...], analogFitError[1, ...] <= analogFitErrorRange[1, 1]), axis=0)] = True
# badAnalogFitErrorMask[2, ~np.all((analogFitErrorRange[2,0] <= analogFitError[2,...],analogFitError[2,...] <= analogFitErrorRange[2,1]),axis=0)] = True
badCellMask = np.logical_or.reduce((badCellMask, badAnalogFitErrorMask[0, ...], badAnalogFitErrorMask[1, ...], badAnalogFitErrorMask[2, ...]))

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
    # axes.clear()
    # axes.hist(analogGains[2, ~badCellMask], bins='sqrt')
    # figure.canvas.draw()
    # analogGainsRange[2,0] = float(input('minimum low gain = '))
    # analogGainsRange[2,1] = float(input('maximum low gain = '))

badAnalogGainsMask = np.zeros((3, 352, 128, 512), dtype=bool)
badAnalogGainsMask[0, ~np.all((analogGainsRange[0, 0] <= analogGains[0, ...], analogGains[0, ...] <= analogGainsRange[0, 1]), axis=0)] = True
badAnalogGainsMask[1, ~np.all((analogGainsRange[1, 0] <= analogGains[1, ...], analogGains[1, ...] <= analogGainsRange[1, 1]), axis=0)] = True
# badAnalogGainsMask[2, ~np.all((analogGainsRange[2,0] <= analogGains[2, ...],analogGains[2, ...] <= analogGainsRange[2,1]),axis=0)] = True
badCellMask = np.logical_or.reduce((badCellMask, badAnalogGainsMask[0, ...], badAnalogGainsMask[1, ...], badAnalogGainsMask[2, ...]))

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
    # axes.clear()
    # axes.hist(analogLineOffsets[2, ~badCellMask], bins='sqrt')
    # figure.canvas.draw()
    # analogLineOffsetsRange[2,0] = float(input('minimum low gain lineOffset = '))
    # analogLineOffsetsRange[2,1] = float(input('maximum low gain lineOffset = '))

badAnalogLineOffsetsMask = np.zeros((3, 352, 128, 512), dtype=bool)
badAnalogLineOffsetsMask[
    0, ~np.all((analogLineOffsetsRange[0, 0] <= analogLineOffsets[0, ...], analogLineOffsets[0, ...] <= analogLineOffsetsRange[0, 1]), axis=0)] = True
badAnalogLineOffsetsMask[
    1, ~np.all((analogLineOffsetsRange[1, 0] <= analogLineOffsets[1, ...], analogLineOffsets[1, ...] <= analogLineOffsetsRange[1, 1]), axis=0)] = True
# badAnalogLineOffsetsMask[2, ~np.all((analogLineOffsetsRange[2,0] <= analogLineOffsets[2, ...],analogLineOffsets[2, ...] <= analogLineOffsetsRange[2,1]),axis=0)] = True
badCellMask = np.logical_or.reduce((badCellMask, badAnalogLineOffsetsMask[0, ...], badAnalogLineOffsetsMask[1, ...], badAnalogLineOffsetsMask[2, ...]))

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
    # axes.clear()
    # axes.hist(digitalStdDeviations[2, ~badCellMask], bins='sqrt')
    # figure.canvas.draw()
    # digitalStdDeviationsRange[2,1] = float(input('maximum low gain digitalStdDeviations = '))

badDigitalStdDeviationsMask = np.zeros((3, 352, 128, 512), dtype=bool)
badDigitalStdDeviationsMask[
    0, ~np.all((digitalStdDeviationsRange[0, 0] <= digitalStdDeviations[0, ...], digitalStdDeviations[0, ...] <= digitalStdDeviationsRange[0, 1]),
               axis=0)] = True
badDigitalStdDeviationsMask[
    1, ~np.all((digitalStdDeviationsRange[1, 0] <= digitalStdDeviations[1, ...], digitalStdDeviations[1, ...] <= digitalStdDeviationsRange[1, 1]),
               axis=0)] = True
# badDigitalStdDeviationsMask[2, ~np.all((digitalStdDeviationsRange[2,0] <= digitalStdDeviations[2, ...],digitalStdDeviations[2, ...] <= digitalStdDeviationsRange[2,1]),axis=0)] = True
badCellMask = np.logical_or.reduce((badCellMask, badDigitalStdDeviationsMask[0, ...], badDigitalStdDeviationsMask[1, ...], badDigitalStdDeviationsMask[2, ...]))

# should be saved in the new generation of digitalMeans files
tmp = digitalStdDeviations == 0  #
# badCellMask = np.logical_or.reduce((badCellMask, tmp[0, ...], tmp[1, ...], tmp[2, ...]))
badCellMask = np.logical_or.reduce((badCellMask, tmp[0, ...], tmp[1, ...]))
digitalStdDeviations[digitalStdDeviations == 0] = 1  # to surpress division by 0 warnings
digitalSpacingsSafetyFactors = np.empty((2, 352, 128, 512), dtype='float32')
digitalSpacingsSafetyFactors[0, ...] = (digitalMeans[1, ...] - digitalMeans[0, ...]) / (digitalStdDeviations[1, ...] + digitalStdDeviations[0, ...])
digitalSpacingsSafetyFactors[1, ...] = (digitalMeans[2, ...] - digitalMeans[1, ...]) / (digitalStdDeviations[2, ...] + digitalStdDeviations[1, ...])

if not ('digitalSpacingsSafetyFactorsMin' in locals()):
    digitalSpacingsSafetyFactorsMin = np.zeros((2,))
    axes.clear()
    axes.hist(digitalSpacingsSafetyFactors[0, ~badCellMask], bins='sqrt')
    figure.canvas.draw()
    digitalSpacingsSafetyFactorsMin[0] = float(input('minimum high-medium digitalSpacingsSafetyFactors = '))
    # axes.clear()
    # axes.hist(digitalSpacingsSafetyFactors[1, ~badCellMask], bins='sqrt')
    # figure.canvas.draw()
    # digitalSpacingsSafetyFactorsMax[1] = float(input('minimum medium-low digitalSpacingsSafetyFactors = '))

badDigitalSpacingsSafetyFactorsMask = np.zeros((2, 352, 128, 512), dtype=bool)
badDigitalSpacingsSafetyFactorsMask[0, digitalSpacingsSafetyFactors[0, ...] < digitalSpacingsSafetyFactorsMin[0]] = True
# badDigitalSpacingsSafetyFactorsMask[1, digitalSpacingsSafetyFactors[1, ...]<digitalSpacingsSafetyFactorsMin[1] ] = True
badCellMask = np.logical_or.reduce((badCellMask, badDigitalSpacingsSafetyFactorsMask[0, ...], badDigitalSpacingsSafetyFactorsMask[1, ...]))

if not ('digitalMeansRange' in locals()):
    digitalMeansRange = np.zeros((3, 2))
    axes.clear()
    axes.hist(digitalMeans[0, ~badCellMask], bins=2 ** 10)
    figure.canvas.draw()
    digitalMeansRange[0, 0] = float(input('minimum high gain digitalMeans = '))
    digitalMeansRange[0, 1] = float(input('maximum high gain digitalMeans = '))
    axes.clear()
    axes.hist(digitalMeans[1, ~badCellMask], bins='sqrt')
    figure.canvas.draw()
    digitalMeansRange[1, 0] = float(input('minimum medium gain digitalMeans = '))
    digitalMeansRange[1, 1] = float(input('maximum medium gain digitalMeans = '))
    # axes.clear()
    # axes.hist(digitalMeans[2, ~badCellMask], bins='sqrt')
    # figure.canvas.draw()
    # digitalMeansRange[2,0] = float(input('minimum low gain digitalMeans = '))
    # digitalMeansRange[2,1] = float(input('maximum low gain digitalMeans = '))

badDigitalMeansMask = np.zeros((3, 352, 128, 512), dtype=bool)
badDigitalMeansMask[0, ~np.all((digitalMeansRange[0, 0] <= digitalMeans[0, ...], digitalMeans[0, ...] <= digitalMeansRange[0, 1]), axis=0)] = True
badDigitalMeansMask[1, ~np.all((digitalMeansRange[1, 0] <= digitalMeans[1, ...], digitalMeans[1, ...] <= digitalMeansRange[1, 1]), axis=0)] = True
# badDigitalMeansMask[2, ~np.all((digitalMeansRange[2,0] <= digitalMeans[2, ...] ,digitalMeans[2, ...]<= digitalMeansRange[2,1]),axis=0)] = True
badCellMask = np.logical_or.reduce((badCellMask, badDigitalMeansMask[0, ...], badDigitalMeansMask[1, ...], badDigitalMeansMask[2, ...]))

if not ('digitalThresholdsRange' in locals()):
    digitalThresholdsRange = np.zeros((2, 2))
    axes.clear()
    axes.hist(digitalThresholds[0, ~badCellMask], bins=2 ** 10)
    figure.canvas.draw()
    digitalThresholdsRange[0, 0] = float(input('minimum high-medium digitalThresholds = '))
    digitalThresholdsRange[0, 1] = float(input('maximum high-medium digitalThresholds = '))
    # axes.clear()
    # axes.hist(digitalThresholds[1, ~badCellMask], bins='sqrt')
    # figure.canvas.draw()
    # digitalThresholdsRange[1, 0] = float(input('minimum medium-low digitalThresholds = '))
    # digitalThresholdsRange[1, 1] = float(input('maximum medium-low digitalThresholds = '))

badDigitalThresholdsMask = np.zeros((2, 352, 128, 512), dtype=bool)
badDigitalThresholdsMask[
    0, ~np.all((digitalThresholdsRange[0, 0] <= digitalThresholds[0, ...], digitalThresholds[0, ...] <= digitalThresholdsRange[0, 1]), axis=0)] = True
# badDigitalThresholdsMask[
#     1, ~np.all((digitalThresholdsRange[1, 0] <= digitalThresholds[1, ...], digitalThresholds[1, ...] <= digitalThresholdsRange[1, 1]), axis=0)] = True
badCellMask = np.logical_or.reduce((badCellMask, badDigitalThresholdsMask[0, ...], badDigitalThresholdsMask[1, ...]))

if not ('darkOffsetRange' in locals()):
    darkOffsetRange = np.zeros((2,))
    axes.clear()
    axes.hist(darkOffset[~badCellMask], bins=2 ** 10)
    figure.canvas.draw()
    darkOffsetRange[0] = float(input('minimum darkOffset = '))
    darkOffsetRange[1] = float(input('maximum darkOffset = '))

badDarkOffsetMask = np.zeros((352, 128, 512), dtype=bool)
badDarkOffsetMask[~np.all((darkOffsetRange[0] <= darkOffset, darkOffset <= darkOffsetRange[1]), axis=0)] = True
badCellMask = np.logical_or.reduce((badCellMask, badDarkOffsetMask))

# if not ('darkStandardDeviationRange' in locals()):
#     darkStandardDeviationRange = np.zeros((2,))
#     axes.clear()
#     axes.hist(darkStandardDeviation[~badCellMask], bins='sqrt')
#     figure.canvas.draw()
#     darkStandardDeviationRange[ 1] = float(input('maximum darkStandardDeviation = '))
#
# badDarkStandardDeviationMask = np.zeros((352, 128, 512), dtype=bool)
# badDarkStandardDeviationMask[~np.all((darkOffsetRange[0] <= darkStandardDeviation, darkStandardDeviation <= darkOffsetRange[1]), axis=0)] = True
# badCellMask = np.logical_or.reduce((badCellMask, badDarkStandardDeviationMask))

if not ('photonSpacingRange' in locals()):
    photonSpacingRange = np.zeros((2,))
    axes.clear()
    axes.hist(photonSpacing[~badAsicBordersMask[0, ...]], bins=2 ** 12)
    figure.canvas.draw()
    photonSpacingRange[0] = float(input('minimum photonSpacing = '))
    photonSpacingRange[1] = float(input('maximum photonSpacing = '))

badPhotonSpacingMask = np.zeros((128, 512), dtype=bool)
badPhotonSpacingMask[~np.all((photonSpacingRange[0] <= photonSpacing, photonSpacing <= photonSpacingRange[1]), axis=0)] = True
badCellMask = np.logical_or(badCellMask, badPhotonSpacingMask)

# if not ('photonSpacingQualityRange' in locals()):
#     photonSpacingQualityMin = 0
#     axes.clear()
#     axes.hist(photonSpacingQuality[~badAsicBordersMask[0, ...]], bins='sqrt')
#     figure.canvas.draw()
#     photonSpacingQualityMin = float(input('minimum photonSpacingQuality = '))
#
# badPhotonSpacingQualityMask = np.zeros((128, 512), dtype=bool)
# badPhotonSpacingQualityMask[ photonSpacingQuality <= photonSpacingQualityMin] = True
# badCellMask = np.logical_or.reduce((badCellMask, badPhotonSpacingQualityMask))

dset_badCellMask[...] = badCellMask
saveFile.flush()
saveFile.close()
print('badCellMask saved in ', saveFileName)

print('\n\n\nentered values:')
print('analogFitErrorRange = \n', analogFitErrorRange)
print('analogGainsRange = \n', analogGainsRange)
print('analogLineOffsetsRange = \n', analogLineOffsetsRange)
print('digitalStdDeviationsRange = \n', digitalStdDeviationsRange)
print('digitalSpacingsSafetyFactorsMin = \n', digitalSpacingsSafetyFactorsMin)
print('digitalMeansRange = \n', digitalMeansRange)
print('digitalThresholdsRange = \n', digitalThresholdsRange)
print('darkOffsetRange = \n', darkOffsetRange)
# print('darkStandardDeviationRange = \n', darkStandardDeviationRange)
print('photonSpacingRange = \n', photonSpacingRange)
# print('photonSpacingQualityMin = \n', photonSpacingQualityMin)


badCellMaskFile = h5py.File(saveFileName, "r", libver='latest')
badCellMask = badCellMaskFile['badCellMask'][...]
pg.image(badCellMask.transpose(0, 2, 1))

print('\n\n\npress enter to quit')
tmp = input()
