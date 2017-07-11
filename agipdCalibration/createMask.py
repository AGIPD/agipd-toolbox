import h5py
import numpy as np

import matplotlib.pyplot as plt
import pyqtgraph as pg

# moduleNumber = 2
# analogGainsFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/analogGains_m' + str(moduleNumber) + '.h5'
# digitalMeansFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/digitalMeans_m' + str(moduleNumber) + '.h5'
# darkOffsetFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/darkOffset_m' + str(moduleNumber) + '.h5'
# photonSpacingFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/photonSpacing_m' + str(moduleNumber) + '.h5'
#
# saveFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/mask_m' + str(moduleNumber) + '.h5'

moduleNumber = 6
analogGainsFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/python_saved_workspace/analogGains_M303_m6_drscs_itestc80.h5'
digitalMeansFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/python_saved_workspace/digitalMeans_M303_m6_drscs_itestc80.h5'
darkOffsetFileName = '/gpfs/cfel/fsds/labs/processed/calibration_1.1/aschkan_stash/3_module_MC_imaging/dark/M303_m6_MC_3module_image_darkcal_replicated.h5'
photonSpacingFileName = '/gpfs/cfel/fsds/labs/processed/calibration_1.1/aschkan_stash/3_module_MC_imaging/xray/photonSpacing_m6_xray_Mo_00000.h5'

saveFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/python_saved_workspace/mask_m6.h5'


saveFile = h5py.File(saveFileName, "w", libver='latest')
dset_badCellMask = saveFile.create_dataset("badCellMask", shape=(352, 128, 512), dtype=bool)

analogGainsFile = h5py.File(analogGainsFileName, 'r', libver='latest')
digitalMeansFile = h5py.File(digitalMeansFileName, 'r', libver='latest')
darkOffsetFile = h5py.File(darkOffsetFileName, 'r', libver='latest')
photonSpacingFile = h5py.File(photonSpacingFileName, 'r', libver='latest')

analogGains = analogGainsFile["/analogGains"][...]  # shape=(3, 352, 128, 512)
analogLineOffsets = analogGainsFile["/anlogLineOffsets"][...]  # shape=(3, 352, 128, 512)
analogFitStdDevs = analogGainsFile["/analogFitStdDevs"][...]  # shape=(3, 352, 128, 512)

digitalMeans = digitalMeansFile["/digitalMeans"][...]  # shape=(352, 3, 128, 512)
digitalThresholds = digitalMeansFile["/digitalThresholds"][...]  # shape=(2, 352, 128, 512)
digitalStdDeviations = digitalMeansFile["/digitalStdDeviations"][...]  # shape=(352, 3, 128, 512)
digitalStdDeviations = digitalStdDeviations.transpose((1, 0, 2, 3))  # shape=(3, 352, 128, 512)
digitalSpacingsSafetyFactors = digitalMeansFile["/digitalSpacingsSafetyFactors"][...]  # shape=(352, 2, 128, 512)
digitalSpacingsSafetyFactors.transpose((1,0,2,3)) # shape=(2, 352, 128, 512)
digitalMeans = digitalMeans.transpose((1, 0, 2, 3))  # shape=(3, 352, 128, 512)

darkOffset = darkOffsetFile["/darkOffset"][...]  # shape=(352, 128, 512)
darkStandardDeviation = darkOffsetFile["/darkStandardDeviation"][...]  # shape=(352, 128, 512)

photonSpacing = photonSpacingFile["/photonSpacing"][...]  # shape=(128, 512)
photonSpacingQuality = photonSpacingFile["/quality"][...]  # shape=(128, 512)

########### fix values - edit, if you want to rerun script multiple times, without always reentering the same values
if moduleNumber == 6:
    darkOffsetRange = np.array([4000, 9000])
    darkStandardDeviationRange = np.array([2, 40])
    analogFitStdDevsRange = np.array([[0, 150], [0, 1250], [0, 550]])
    analogGainsRange = np.array([[20, 400], [0.45, 10.6], [0.01, 2]])
    analogLineOffsetsRange = np.array([[2950, 5250], [4300, 6200], [0, 0]])
    digitalStdDeviationsRange = np.array([[0, 80], [0, 35], [0, 0]])
    digitalSpacingsSafetyFactorsMin = np.array([9, 0])
    digitalMeansRange = np.array([[5580, 7000], [6740, 8000], [0, 0]])
    digitalThresholdsRange = np.array([[6000, 7600], [0, 0]])
    photonSpacingRange = np.array([120, 200])
    photonSpacingQualityMin = 0
###########

badCellMask = np.zeros((352, 128, 512), dtype=bool)

################# manual mask - please edit (e.g. beamstop)
manualMask = np.zeros((128, 512), dtype=bool)

# if moduleNumber == 3:
#     manualMask[65:105, 200:260] = True
#     manualMask[80:105, 200:511] = True
#
#     manualMask[80:86, 22:43] = True
#     manualMask[75:100, 31:43] = True
#
#     manualMask[64:127, 127:195] = True

badCellMask = np.logical_or(manualMask, badCellMask)
###########

######### systematic mask - please edit
# systematicMask = np.zeros((128, 512, 11, 32), dtype=bool)
# systematicMask[..., 27:29] = True  # 32-column tips
# systematicMask = systematicMask.reshape((128, 512, 352)).transpose(2, 0, 1)
# badCellMask = np.logical_or(systematicMask, badCellMask)
#########

######### bad asic borders mask - please edit
badAsicBordersMask = np.zeros((352, 128, 512), dtype=bool)
# badAsicBordersMask[:, (0, 63, 64, 127), :] = True
# for column in np.arange(8):
#     badAsicBordersMask[:, :, (column * 64, column * 64 + 63)] = True
badCellMask = np.logical_or(badAsicBordersMask, badCellMask)
#########

########################################### the rest can be left untouched

for tmp in (analogGains, analogLineOffsets, analogFitStdDevs, digitalMeans, digitalThresholds, digitalStdDeviations):
    for i in np.arange(tmp.shape[0]):
        badCellMask = np.logical_or(badCellMask, ~np.isfinite(tmp[i, ...]))
    tmp[~np.isfinite(tmp)] = 0  # to surpress "invalid value in comparison" warnings
for tmp in (darkOffset, darkStandardDeviation, photonSpacing, photonSpacingQuality):
    badCellMask = np.logical_or(badCellMask, ~np.isfinite(tmp))
    tmp[~np.isfinite(tmp)] = 0  # to surpress "invalid value in comparison" warnings

figure = plt.figure()
axes = figure.gca()
figure.show()

if not ('darkOffsetRange' in locals()):
    darkOffsetRange = np.zeros((2,))
    axes.clear()
    axes.hist(darkOffset[~badCellMask], bins='sqrt')
    figure.canvas.draw()
    darkOffsetRange[0] = float(input('minimum darkOffset = '))
    darkOffsetRange[1] = float(input('maximum darkOffset = '))

darkOffset = darkOffset.transpose(1, 2, 0)  # shape = (128, 512, 352)
darkOffset = darkOffset.reshape([128, 512, 11, 32])
badOffsetPixelMask = np.zeros((128, 512, 11, 32), dtype=bool)
badOffsetPixelMask[darkOffset < darkOffsetRange[0]] = True
badOffsetPixelMask[darkOffset > darkOffsetRange[1]] = True
for y in np.arange(128):
    for x in np.arange(512):
        lineMediansOffset = np.median(darkOffset[y, x, ...], axis=1)
        tmp = np.median(lineMediansOffset)
        if tmp > darkOffsetRange[1] or tmp < darkOffsetRange[0]:
            badOffsetPixelMask[y, x, ...] = True
        else:
            for line in np.arange(11):
                if lineMediansOffset[line] > darkOffsetRange[1] or lineMediansOffset[line] < darkOffsetRange[0]:
                    badOffsetPixelMask[y, x, line, :] = True

badDarkOffsetMask = badOffsetPixelMask.reshape((128, 512, 352)).transpose(2, 0, 1)
badCellMask = np.logical_or.reduce((badCellMask, badDarkOffsetMask))

if not ('darkStandardDeviationRange' in locals()):
    darkStandardDeviationRange = np.zeros((2,))
    axes.clear()
    axes.hist(darkStandardDeviation[~badCellMask], bins='sqrt')
    figure.canvas.draw()
    darkStandardDeviationRange[0] = float(input('minimum darkStandardDeviation = '))
    darkStandardDeviationRange[1] = float(input('maximum darkStandardDeviation = '))

darkStandardDeviation = darkStandardDeviation.transpose(1, 2, 0)  # shape = (128, 512, 352)
darkStandardDeviation = darkStandardDeviation.reshape([128, 512, 11, 32])
badStdDevPixelMask = np.zeros((128, 512, 11, 32), dtype=bool)
badStdDevPixelMask[darkStandardDeviation < darkStandardDeviationRange[0]] = True
badStdDevPixelMask[darkStandardDeviation > darkStandardDeviationRange[1]] = True
for y in np.arange(128):
    for x in np.arange(512):
        lineMediansStdDev = np.median(darkStandardDeviation[y, x, ...], axis=1)
        tmp = np.median(lineMediansStdDev)
        if tmp > darkStandardDeviationRange[1] or tmp < darkStandardDeviationRange[0]:
            badStdDevPixelMask[y, x, ...] = True
        else:
            for line in np.arange(11):
                if lineMediansStdDev[line] > darkStandardDeviationRange[1] or lineMediansStdDev[line] < darkStandardDeviationRange[0]:
                    badStdDevPixelMask[y, x, line, :] = True

badDarkStandardDeviationMask = badStdDevPixelMask.reshape((128, 512, 352)).transpose(2, 0, 1)
badCellMask = np.logical_or.reduce((badCellMask, badDarkStandardDeviationMask))

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

badanalogFitStdDevsMask = np.zeros((3, 352, 128, 512), dtype=bool)
badanalogFitStdDevsMask[
    0, ~np.all((analogFitStdDevsRange[0, 0] <= analogFitStdDevs[0, ...], analogFitStdDevs[0, ...] <= analogFitStdDevsRange[0, 1]), axis=0)] = True
badanalogFitStdDevsMask[
    1, ~np.all((analogFitStdDevsRange[1, 0] <= analogFitStdDevs[1, ...], analogFitStdDevs[1, ...] <= analogFitStdDevsRange[1, 1]), axis=0)] = True
badanalogFitStdDevsMask[
    2, ~np.all((analogFitStdDevsRange[2, 0] <= analogFitStdDevs[2, ...], analogFitStdDevs[2, ...] <= analogFitStdDevsRange[2, 1]), axis=0)] = True
badCellMask = np.logical_or.reduce((badCellMask, badanalogFitStdDevsMask[0, ...], badanalogFitStdDevsMask[1, ...], badanalogFitStdDevsMask[2, ...]))

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

badAnalogGainsMask = np.zeros((3, 352, 128, 512), dtype=bool)
badAnalogGainsMask[0, ~np.all((analogGainsRange[0, 0] <= analogGains[0, ...], analogGains[0, ...] <= analogGainsRange[0, 1]), axis=0)] = True
badAnalogGainsMask[1, ~np.all((analogGainsRange[1, 0] <= analogGains[1, ...], analogGains[1, ...] <= analogGainsRange[1, 1]), axis=0)] = True
badAnalogGainsMask[2, ~np.all((analogGainsRange[2, 0] <= analogGains[2, ...], analogGains[2, ...] <= analogGainsRange[2, 1]), axis=0)] = True
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
    axes.clear()
    axes.hist(analogLineOffsets[2, ~badCellMask], bins='sqrt')
    figure.canvas.draw()
    analogLineOffsetsRange[2, 0] = float(input('minimum low gain lineOffset = '))
    analogLineOffsetsRange[2, 1] = float(input('maximum low gain lineOffset = '))

badAnalogLineOffsetsMask = np.zeros((3, 352, 128, 512), dtype=bool)
badAnalogLineOffsetsMask[
    0, ~np.all((analogLineOffsetsRange[0, 0] <= analogLineOffsets[0, ...], analogLineOffsets[0, ...] <= analogLineOffsetsRange[0, 1]), axis=0)] = True
badAnalogLineOffsetsMask[
    1, ~np.all((analogLineOffsetsRange[1, 0] <= analogLineOffsets[1, ...], analogLineOffsets[1, ...] <= analogLineOffsetsRange[1, 1]), axis=0)] = True
badAnalogLineOffsetsMask[
    2, ~np.all((analogLineOffsetsRange[2, 0] <= analogLineOffsets[2, ...], analogLineOffsets[2, ...] <= analogLineOffsetsRange[2, 1]), axis=0)] = True
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
    axes.clear()
    axes.hist(digitalStdDeviations[2, ~badCellMask], bins='sqrt')
    figure.canvas.draw()
    digitalStdDeviationsRange[2, 1] = float(input('maximum low gain digitalStdDeviations = '))

badDigitalStdDeviationsMask = np.zeros((3, 352, 128, 512), dtype=bool)
badDigitalStdDeviationsMask[
    0, ~np.all((digitalStdDeviationsRange[0, 0] <= digitalStdDeviations[0, ...], digitalStdDeviations[0, ...] <= digitalStdDeviationsRange[0, 1]),
               axis=0)] = True
badDigitalStdDeviationsMask[
    1, ~np.all((digitalStdDeviationsRange[1, 0] <= digitalStdDeviations[1, ...], digitalStdDeviations[1, ...] <= digitalStdDeviationsRange[1, 1]),
               axis=0)] = True
badDigitalStdDeviationsMask[
    2, ~np.all((digitalStdDeviationsRange[2, 0] <= digitalStdDeviations[2, ...], digitalStdDeviations[2, ...] <= digitalStdDeviationsRange[2, 1]),
               axis=0)] = True
badCellMask = np.logical_or.reduce((badCellMask, badDigitalStdDeviationsMask[0, ...], badDigitalStdDeviationsMask[1, ...], badDigitalStdDeviationsMask[2, ...]))

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

badDigitalSpacingsSafetyFactorsMask = np.zeros((2, 352, 128, 512), dtype=bool)
badDigitalSpacingsSafetyFactorsMask[0, digitalSpacingsSafetyFactors[0, ...] < digitalSpacingsSafetyFactorsMin[0]] = True
badDigitalSpacingsSafetyFactorsMask[1, digitalSpacingsSafetyFactors[1, ...] < digitalSpacingsSafetyFactorsMin[1]] = True
badCellMask = np.logical_or.reduce((badCellMask, badDigitalSpacingsSafetyFactorsMask[0, ...], badDigitalSpacingsSafetyFactorsMask[1, ...]))

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

badDigitalMeansMask = np.zeros((3, 352, 128, 512), dtype=bool)
badDigitalMeansMask[0, ~np.all((digitalMeansRange[0, 0] <= digitalMeans[0, ...], digitalMeans[0, ...] <= digitalMeansRange[0, 1]), axis=0)] = True
badDigitalMeansMask[1, ~np.all((digitalMeansRange[1, 0] <= digitalMeans[1, ...], digitalMeans[1, ...] <= digitalMeansRange[1, 1]), axis=0)] = True
badDigitalMeansMask[2, ~np.all((digitalMeansRange[2, 0] <= digitalMeans[2, ...], digitalMeans[2, ...] <= digitalMeansRange[2, 1]), axis=0)] = True
badCellMask = np.logical_or.reduce((badCellMask, badDigitalMeansMask[0, ...], badDigitalMeansMask[1, ...], badDigitalMeansMask[2, ...]))

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

badDigitalThresholdsMask = np.zeros((2, 352, 128, 512), dtype=bool)
badDigitalThresholdsMask[
    0, ~np.all((digitalThresholdsRange[0, 0] <= digitalThresholds[0, ...], digitalThresholds[0, ...] <= digitalThresholdsRange[0, 1]), axis=0)] = True
badDigitalThresholdsMask[
    1, ~np.all((digitalThresholdsRange[1, 0] <= digitalThresholds[1, ...], digitalThresholds[1, ...] <= digitalThresholdsRange[1, 1]), axis=0)] = True
badCellMask = np.logical_or.reduce((badCellMask, badDigitalThresholdsMask[0, ...], badDigitalThresholdsMask[1, ...]))

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

if not ('photonSpacingQualityRange' in locals()):
    photonSpacingQualityMin = 0
    axes.clear()
    axes.hist(photonSpacingQuality[~badAsicBordersMask[0, ...]], bins='sqrt')
    figure.canvas.draw()
    photonSpacingQualityMin = float(input('minimum photonSpacingQuality = '))

badPhotonSpacingQualityMask = np.zeros((128, 512), dtype=bool)
badPhotonSpacingQualityMask[photonSpacingQuality <= photonSpacingQualityMin] = True
badCellMask = np.logical_or.reduce((badCellMask, badPhotonSpacingQualityMask))

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
