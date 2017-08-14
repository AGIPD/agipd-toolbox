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


module_id = 'M314'
module_number = 'm7'
temperature = 'temperature_m15C'
itestc = 'itestc150'
tint = 'tint150ns'
element = 'Cu'
base_dir = "/gpfs/cfel/fsds/labs/agipd/calibration/processed/{}/{}".format(module_id, temperature)

# Input files
# Dynamic Range Scan (gain - current source)
gains_filename = "analogGains_{}_{}.h5".format(module_id, module_number)
gains_filepath = os.path.join(base_dir, 'drscs', itestc, gains_filename)
gainbits_filename = "digitalMeans_{}_{}.h5".format(module_id, module_number)
gainbits_filepath = os.path.join(base_dir, 'drscs', itestc, gainbits_filename)
# Darks
dark_filename = "darkOffset_{}_{}_{}.h5".format(module_id, module_number, tint)
dark_filepath = os.path.join(base_dir, 'dark', dark_filename)
# X-ray
xray_filename = "photonSpacing_{}_{}_xray_{}.h5".format(module_id, module_number, element)
xray_filepath = os.path.join(base_dir, 'xray', xray_filename)


# Output mask file
save_filename = os.path.join(base_dir, 'cal_output/agipd_mask_{}_{}_{}_{}_{}_{}.h5'.format(module_id, module_number, temperature, itestc, tint, element))

# Create output file
f_out = h5py.File(save_filename, "w", libver='latest')
dset_combined_mask = f_out.create_dataset("combined_mask", shape=(352, 128, 512), dtype=bool)
masks = f_out.create_group("masks")

# Get data from gain files
gains_file = h5py.File(gains_filepath, 'r', libver='latest')
gains = gains_file["/analogGains"][...]  # shape=(3, 352, 128, 512)
gain_offsets = gains_file["/anlogLineOffsets"][...]  # shape=(3, 352, 128, 512)
gain_stds = gains_file["/analogFitStdDevs"][...]  # shape=(3, 352, 128, 512)
gains_file.close()

gainbits_file = h5py.File(gainbits_filepath, 'r', libver='latest')
gainbits = gainbits_file["/digitalMeans"][...]  # shape=(352, 3, 128, 512)
gainbit_thresholds = gainbits_file["/digitalThresholds"][...]  # shape=(2, 352, 128, 512)
gainbit_stds = gainbits_file["/digitalStdDeviations"][...]  # shape=(352, 3, 128, 512)
gainbit_stds = gainbit_stds.transpose((1, 0, 2, 3))  # shape=(3, 352, 128, 512)
gainbit_spacing_safety = gainbits_file["/digitalSpacingsSafetyFactors"][...]  # shape=(352, 2, 128, 512)
gainbit_spacing_safety.transpose((1,0,2,3)) # shape=(2, 352, 128, 512)
gainbits = gainbits.transpose((1, 0, 2, 3))  # shape=(3, 352, 128, 512)
gainbits_file.close()

# Get data from dark file
dark_file = h5py.File(dark_filepath, 'r', libver='latest')
dark_offset = dark_file["/darkOffset"][...]  # shape=(352, 128, 512)
noise = dark_file["/darkStandardDeviation"][...]  # shape=(352, 128, 512)
dark_file.close()

# Get data from xray file
xray_file = h5py.File(xray_filepath, 'r', libver='latest')
photon_spacing = xray_file["/photonSpacing"][...]  # shape=(128, 512)
photon_spacing_quality = xray_file["/quality"][...]  # shape=(128, 512)
xray_file.close()

########### Cut values #################################################
dark_offset_range = np.array([3, 3]) # in sigmas from mean
noise_range = np.array([3, 3]) # in sigmas from mean
#analogFitStdDevsRange = np.array([[0, 150], [0, 1250], [0, 550]])
gains_range = np.array([[3, 3], [3, 3], [3, 3]]) # in sigmas from mean
#analogLineOffsetsRange = np.array([[2950, 5250], [4300, 6200], [0, 0]])
#digitalStdDeviationsRange = np.array([[0, 80], [0, 35], [0, 0]])
gainbit_stds_range = np.array([[3, 3], [3, 3], [3, 3]]) # in sigmas from mean
#digitalSpacingsSafetyFactorsMin = np.array([9, 0])
#digitalMeansRange = np.array([[5580, 7000], [6740, 8000], [0, 0]])
#digitalThresholdsRange = np.array([[6000, 7600], [0, 0]])
photon_spacing_range = np.array([3, 3]) # in sigmas from mean
#photonSpacingQualityMin = 0
########################################################################



##################### the rest can be left untouched ###################

combined_mask = np.zeros((352, 128, 512), dtype=bool)


########################################################################
failed_fit_mask = np.zeros((352, 128, 512), dtype=bool)
dset_failed_fit_mask = masks.create_dataset("failed_fit", shape=(352, 128, 512), dtype=bool)

for tmp in (gains, gain_offsets, gain_stds, gainbits, gainbit_thresholds, gainbit_stds):
    for i in np.arange(tmp.shape[0]):
        failed_fit_mask = np.logical_or(failed_fit_mask, ~np.isfinite(tmp[i, ...]))
    tmp[~np.isfinite(tmp)] = 0  # to surpress "invalid value in comparison" warnings
for tmp in (dark_offset, noise, photon_spacing, photon_spacing_quality):
    failed_fit_mask = np.logical_or(failed_fit_mask, ~np.isfinite(tmp))
    tmp[~np.isfinite(tmp)] = 0  # to surpress "invalid value in comparison" warnings

figure = plt.figure()
axes = figure.gca()
#figure.show()

dset_failed_fit_mask[...] = failed_fit_mask
combined_mask = np.logical_or(combined_mask, failed_fit_mask)
print('\n\n failed fit percentage of masked cells: ', 100 * failed_fit_mask.flatten().sum() / failed_fit_mask.size)
########################################################################


########################################################################
if not ('dark_offset_range' in locals()):
    dark_offset_range = np.zeros((2,))
    axes.clear()
    axes.hist(dark_offset[~combined_mask], bins='sqrt')
    figure.canvas.draw()
    dark_offset_range[0] = float(input('minimum dark_offset = '))
    dark_offset_range[1] = float(input('maximum dark_offset = '))

dark_offset_mask = np.zeros((352, 128, 512), dtype=bool)
dset_dark_offset_mask = masks.create_dataset("dark_offset", shape=(352, 128, 512), dtype=bool)

dark_offset_mean = np.mean(dark_offset[~combined_mask])
dark_offset_sigma = np.std(dark_offset[~combined_mask])

dark_offset_mask[dark_offset < dark_offset_mean - (dark_offset_range[0] * dark_offset_sigma)] = True
dark_offset_mask[dark_offset > dark_offset_mean + (dark_offset_range[1] * dark_offset_sigma)] = True

dset_dark_offset_mask[...] = dark_offset_mask
combined_mask = np.logical_or.reduce((combined_mask, dark_offset_mask))
print('\n\n dark offset percentage of masked cells: ', 100 * dark_offset_mask.flatten().sum() / dark_offset_mask.size)
########################################################################


########################################################################
if not ('noise_range' in locals()):
    noise_range = np.zeros((2,))
    axes.clear()
    axes.hist(noise[~combined_mask], bins='sqrt')
    figure.canvas.draw()
    noise_range[0] = float(input('minimum noise = '))
    noise_range[1] = float(input('maximum noise = '))

noise_mask = np.zeros((352, 128, 512), dtype=bool)
dset_noise_mask = masks.create_dataset("noise", shape=(352, 128, 512), dtype=bool)

noise_mean = np.mean(noise[~combined_mask])
noise_sigma = np.std(noise[~combined_mask])

noise_mask[noise < noise_mean - (noise_range[0] * noise_sigma)] = True
noise_mask[noise > noise_mean + (noise_range[1] * noise_sigma)] = True

dset_noise_mask[...] = noise_mask
combined_mask = np.logical_or.reduce((combined_mask, noise_mask))
print('\n\n noise percentage of masked cells: ', 100 * noise_mask.flatten().sum() / noise_mask.size)
########################################################################

"""
########################################################################
if not ('analogFitStdDevsRange' in locals()):
    analogFitStdDevsRange = np.zeros((3, 2))
    axes.hist(analogFitStdDevs[0, ~combined_mask], bins='sqrt')
    figure.canvas.draw()
    analogFitStdDevsRange[0, 1] = float(input('maximum high gain analogFitStdDevs = '))
    axes.clear()
    axes.hist(analogFitStdDevs[1, ~combined_mask], bins='sqrt')
    figure.canvas.draw()
    analogFitStdDevsRange[1, 1] = float(input('maximum medium gain analogFitStdDevs = '))
    axes.clear()
    axes.hist(analogFitStdDevs[2, ~combined_mask], bins='sqrt')
    figure.canvas.draw()
    analogFitStdDevsRange[2, 1] = float(input('maximum low gain analogFitStdDevs = '))

anaFitStdDevsMask = np.zeros((3, 352, 128, 512), dtype=bool)
dset_anaFitStdDevMask = masks.create_dataset("anaFitStdDevMask", shape=(3, 352, 128, 512), dtype=bool)
anaFitStdDevsMask[
    0, ~np.all((analogFitStdDevsRange[0, 0] <= analogFitStdDevs[0, ...], analogFitStdDevs[0, ...] <= analogFitStdDevsRange[0, 1]), axis=0)] = True
anaFitStdDevsMask[
    1, ~np.all((analogFitStdDevsRange[1, 0] <= analogFitStdDevs[1, ...], analogFitStdDevs[1, ...] <= analogFitStdDevsRange[1, 1]), axis=0)] = True
anaFitStdDevsMask[
    2, ~np.all((analogFitStdDevsRange[2, 0] <= analogFitStdDevs[2, ...], analogFitStdDevs[2, ...] <= analogFitStdDevsRange[2, 1]), axis=0)] = True
dset_anaFitStdDevMask[...] = anaFitStdDevsMask
combined_mask = np.logical_or.reduce((combined_mask, anaFitStdDevsMask[0, ...], anaFitStdDevsMask[1, ...], anaFitStdDevsMask[2, ...]))
print('\n\n analog fit std dev percentage of masked cells: ', 100 * anaFitStdDevsMask.flatten().sum() / anaFitStdDevsMask.size)
########################################################################
"""

########################################################################
if not ('gains_range' in locals()):
    gains_range = np.zeros((3, 2))
    axes.clear()
    axes.hist(gains[0, ~combined_mask], bins='sqrt')
    figure.canvas.draw()
    gains_range[0, 0] = float(input('minimum high gain = '))
    gains_range[0, 1] = float(input('maximum high gain = '))
    axes.clear()
    axes.hist(gains[1, ~combined_mask], bins='sqrt')
    figure.canvas.draw()
    gains_range[1, 0] = float(input('minimum medium gain = '))
    gains_range[1, 1] = float(input('maximum medium gain = '))
    axes.clear()
    axes.hist(gains[2, ~combined_mask], bins='sqrt')
    figure.canvas.draw()
    gains_range[2, 0] = float(input('minimum low gain = '))
    gains_range[2, 1] = float(input('maximum low gain = '))

gains_mask = np.zeros((3, 352, 128, 512), dtype=bool)
dset_gains_mask = masks.create_dataset("gain", shape=(3, 352, 128, 512), dtype=bool)

gains_hi_mean = np.mean(gains[0, ~combined_mask])
gains_med_mean = np.mean(gains[1, ~combined_mask])
gains_low_mean = np.mean(gains[2, ~combined_mask])

gains_hi_sigma = np.std(gains[0, ~combined_mask])
gains_med_sigma = np.std(gains[1, ~combined_mask])
gains_low_sigma = np.std(gains[2, ~combined_mask])

gains_mask[0, ( gains[0, ...] < (gains_hi_mean - (gains_range[0, 0] * gains_hi_sigma)) )] = True
gains_mask[0, ( gains[0, ...] > (gains_hi_mean + (gains_range[0, 1] * gains_hi_sigma)) )] = True
gains_mask[1, ( gains[1, ...] < (gains_med_mean - (gains_range[1, 0] * gains_med_sigma)) )] = True 
gains_mask[1, ( gains[1, ...] > (gains_med_mean + (gains_range[1, 1] * gains_med_sigma)) )] = True
gains_mask[2, ( gains[2, ...] < (gains_low_mean - (gains_range[2, 0] * gains_low_sigma)) )] = True
gains_mask[2, ( gains[2, ...] > (gains_low_mean + (gains_range[2, 1] * gains_low_sigma)) )] = True

dset_gains_mask[...] = gains_mask
combined_mask = np.logical_or.reduce((combined_mask, gains_mask[0, ...], gains_mask[1, ...], gains_mask[2, ...]))
print('\n\n gain range percentage of masked cells: ', 100 * gains_mask.flatten().sum() / gains_mask.size)
########################################################################

"""
########################################################################
if not ('analogLineOffsetsRange' in locals()):
    analogLineOffsetsRange = np.zeros((3, 2))
    axes.clear()
    axes.hist(analogLineOffsets[0, ~combined_mask], bins='sqrt')
    figure.canvas.draw()
    analogLineOffsetsRange[0, 0] = float(input('minimum high gain lineOffset = '))
    analogLineOffsetsRange[0, 1] = float(input('maximum high gain lineOffset = '))
    axes.clear()
    axes.hist(analogLineOffsets[1, ~combined_mask], bins='sqrt')
    figure.canvas.draw()
    analogLineOffsetsRange[1, 0] = float(input('minimum medium gain lineOffset = '))
    analogLineOffsetsRange[1, 1] = float(input('maximum medium gain lineOffset = '))
    axes.clear()
    axes.hist(analogLineOffsets[2, ~combined_mask], bins='sqrt')
    figure.canvas.draw()
    analogLineOffsetsRange[2, 0] = float(input('minimum low gain lineOffset = '))
    analogLineOffsetsRange[2, 1] = float(input('maximum low gain lineOffset = '))

anaOffsetMask = np.zeros((3, 352, 128, 512), dtype=bool)
dset_anaOffsetMask = masks.create_dataset("anaOffsetMask", shape=(3, 352, 128, 512), dtype=bool)
anaOffsetMask[
    0, ~np.all((analogLineOffsetsRange[0, 0] <= analogLineOffsets[0, ...], analogLineOffsets[0, ...] <= analogLineOffsetsRange[0, 1]), axis=0)] = True
anaOffsetMask[
    1, ~np.all((analogLineOffsetsRange[1, 0] <= analogLineOffsets[1, ...], analogLineOffsets[1, ...] <= analogLineOffsetsRange[1, 1]), axis=0)] = True
anaOffsetMask[
    2, ~np.all((analogLineOffsetsRange[2, 0] <= analogLineOffsets[2, ...], analogLineOffsets[2, ...] <= analogLineOffsetsRange[2, 1]), axis=0)] = True

dset_anaOffsetMask[...] = anaOffsetMask
combined_mask = np.logical_or.reduce((combined_mask, anaOffsetMask[0, ...], anaOffsetMask[1, ...], anaOffsetMask[2, ...]))
print('\n\n gain offset percentage of masked cells: ', 100 * anaOffsetMask.flatten().sum() / anaOffsetMask.size)
########################################################################
"""

########################################################################
if not ('gainbit_stds_range' in locals()):
    gainbit_stds_range = np.zeros((3, 2))
    axes.clear()
    axes.hist(gainbit_stds[0, ~combined_mask], bins='sqrt')
    figure.canvas.draw()
    gainbit_stds_range[0, 1] = float(input('maximum high gain gainbit_stds = '))
    axes.clear()
    axes.hist(gainbit_stds[1, ~combined_mask], bins='sqrt')
    figure.canvas.draw()
    gainbit_stds_range[1, 1] = float(input('maximum medium gain gainbit_stds = '))
    axes.clear()
    axes.hist(gainbit_stds[2, ~combined_mask], bins='sqrt')
    figure.canvas.draw()
    gainbit_stds_range[2, 1] = float(input('maximum low gain gainbit_stds = '))

gainbit_stds_mask = np.zeros((3, 352, 128, 512), dtype=bool)
dset_gainbit_stds_mask = masks.create_dataset("gainbit", shape=(3, 352, 128, 512), dtype=bool)

gainbit_stds_hi_mean = np.mean(gainbit_stds[0, ~combined_mask])
gainbit_stds_med_mean = np.mean(gainbit_stds[1, ~combined_mask])
gainbit_stds_low_mean = np.mean(gainbit_stds[2, ~combined_mask])

gainbit_stds_hi_sigma = np.std(gainbit_stds[0, ~combined_mask])
gainbit_stds_med_sigma = np.std(gainbit_stds[1, ~combined_mask])
gainbit_stds_low_sigma = np.std(gainbit_stds[2, ~combined_mask])

gainbit_stds_mask[0, ( gainbit_stds[0, ...] < (gainbit_stds_hi_mean - (gainbit_stds_range[0, 0] * gainbit_stds_hi_sigma)) )] = True
gainbit_stds_mask[0, ( gainbit_stds[0, ...] > (gainbit_stds_hi_mean + (gainbit_stds_range[0, 1] * gainbit_stds_hi_sigma)) )] = True
gainbit_stds_mask[1, ( gainbit_stds[1, ...] < (gainbit_stds_med_mean - (gainbit_stds_range[1, 0] * gainbit_stds_med_sigma)) )] = True
gainbit_stds_mask[1, ( gainbit_stds[1, ...] > (gainbit_stds_med_mean + (gainbit_stds_range[1, 1] * gainbit_stds_med_sigma)) )] = True
gainbit_stds_mask[2, ( gainbit_stds[2, ...] < (gainbit_stds_low_mean - (gainbit_stds_range[2, 0] * gainbit_stds_low_sigma)) )] = True
gainbit_stds_mask[2, ( gainbit_stds[2, ...] > (gainbit_stds_low_mean + (gainbit_stds_range[2, 1] * gainbit_stds_low_sigma)) )] = True

dset_gainbit_stds_mask[...] = gainbit_stds_mask
combined_mask = np.logical_or.reduce((combined_mask, gainbit_stds_mask[0, ...], gainbit_stds_mask[1, ...], gainbit_stds_mask[2, ...]))
print('\n\n dig std percentage of masked cells: ', 100 * gainbit_stds_mask.flatten().sum() / gainbit_stds_mask.size)
########################################################################

"""
########################################################################
if not ('digitalSpacingsSafetyFactorsMin' in locals()):
    digitalSpacingsSafetyFactorsMin = np.zeros((2,))
    axes.clear()
    axes.hist(digitalSpacingsSafetyFactors[0, ~combined_mask], bins='sqrt')
    figure.canvas.draw()
    digitalSpacingsSafetyFactorsMin[0] = float(input('minimum high-medium digitalSpacingsSafetyFactors = '))
    axes.clear()
    axes.hist(digitalSpacingsSafetyFactors[1, ~combined_mask], bins='sqrt')
    figure.canvas.draw()
    digitalSpacingsSafetyFactorsMin[1] = float(input('minimum medium-low digitalSpacingsSafetyFactors = '))

digSpacingMask = np.zeros((2, 352, 128, 512), dtype=bool)
dset_digSpacingMask = masks.create_dataset("digSpacingMask", shape=(2, 352, 128, 512), dtype=bool)
digSpacingMask[0, digitalSpacingsSafetyFactors[0, ...] < digitalSpacingsSafetyFactorsMin[0]] = True
digSpacingMask[1, digitalSpacingsSafetyFactors[1, ...] < digitalSpacingsSafetyFactorsMin[1]] = True

dset_digSpacingMask[...] = digSpacingMask
combined_mask = np.logical_or.reduce((combined_mask, digSpacingMask[0, ...], digSpacingMask[1, ...]))
print('\n\n ph. spacing safety factor percentage of masked cells: ', 100 * digSpacingMask.flatten().sum() / digSpacingMask.size)
########################################################################
"""
"""
########################################################################
if not ('digitalMeansRange' in locals()):
    digitalMeansRange = np.zeros((3, 2))
    axes.clear()
    axes.hist(digitalMeans[0, ~combined_mask], bins='sqrt')
    figure.canvas.draw()
    digitalMeansRange[0, 0] = float(input('minimum high gain digitalMeans = '))
    digitalMeansRange[0, 1] = float(input('maximum high gain digitalMeans = '))
    axes.clear()
    axes.hist(digitalMeans[1, ~combined_mask], bins='sqrt')
    figure.canvas.draw()
    digitalMeansRange[1, 0] = float(input('minimum medium gain digitalMeans = '))
    digitalMeansRange[1, 1] = float(input('maximum medium gain digitalMeans = '))
    axes.clear()
    axes.hist(digitalMeans[2, ~combined_mask], bins='sqrt')
    figure.canvas.draw()
    digitalMeansRange[2, 0] = float(input('minimum low gain digitalMeans = '))
    digitalMeansRange[2, 1] = float(input('maximum low gain digitalMeans = '))

digMeansMask = np.zeros((3, 352, 128, 512), dtype=bool)
dset_digMeansMask = masks.create_dataset("digMeansMask", shape=(3, 352, 128, 512), dtype=bool)
digMeansMask[0, ~np.all((digitalMeansRange[0, 0] <= digitalMeans[0, ...], digitalMeans[0, ...] <= digitalMeansRange[0, 1]), axis=0)] = True
digMeansMask[1, ~np.all((digitalMeansRange[1, 0] <= digitalMeans[1, ...], digitalMeans[1, ...] <= digitalMeansRange[1, 1]), axis=0)] = True
digMeansMask[2, ~np.all((digitalMeansRange[2, 0] <= digitalMeans[2, ...], digitalMeans[2, ...] <= digitalMeansRange[2, 1]), axis=0)] = True

dset_digMeansMask[...] = digMeansMask
combined_mask = np.logical_or.reduce((combined_mask, digMeansMask[0, ...], digMeansMask[1, ...], digMeansMask[2, ...]))
print('\n\n dig. means percentage of masked cells: ', 100 * digMeansMask.flatten().sum() / digMeansMask.size)
########################################################################
"""
"""
########################################################################
if not ('digitalThresholdsRange' in locals()):
    digitalThresholdsRange = np.zeros((2, 2))
    axes.clear()
    axes.hist(digitalThresholds[0, ~combined_mask], bins=2 ** 10)
    figure.canvas.draw()
    digitalThresholdsRange[0, 0] = float(input('minimum high-medium digitalThresholds = '))
    digitalThresholdsRange[0, 1] = float(input('maximum high-medium digitalThresholds = '))
    axes.clear()
    axes.hist(digitalThresholds[1, ~combined_mask], bins='sqrt')
    figure.canvas.draw()
    digitalThresholdsRange[1, 0] = float(input('minimum medium-low digitalThresholds = '))
    digitalThresholdsRange[1, 1] = float(input('maximum medium-low digitalThresholds = '))

digThreshMask = np.zeros((2, 352, 128, 512), dtype=bool)
dset_digThreshMask = masks.create_dataset("digThreshMask", shape=(2, 352, 128, 512), dtype=bool)
digThreshMask[
    0, ~np.all((digitalThresholdsRange[0, 0] <= digitalThresholds[0, ...], digitalThresholds[0, ...] <= digitalThresholdsRange[0, 1]), axis=0)] = True
digThreshMask[
    1, ~np.all((digitalThresholdsRange[1, 0] <= digitalThresholds[1, ...], digitalThresholds[1, ...] <= digitalThresholdsRange[1, 1]), axis=0)] = True

dset_digThreshMask[...] = digThreshMask
combined_mask = np.logical_or.reduce((combined_mask, digThreshMask[0, ...], digThreshMask[1, ...]))
print('\n\n dig thresh percentage of masked cells: ', 100 * digThreshMask.flatten().sum() / digThreshMask.size)
########################################################################
"""

########################################################################
if not ('photon_spacing_range' in locals()):
    photon_spacing_range = np.zeros((2,))
    axes.clear()
    #axes.hist(photon_spacing[~badAsicBordersMask[0, ...]], bins=2 ** 12)
    axes.hist(photon_spacing[~combined_mask], bins='sqrt')
    figure.canvas.draw()
    photon_spacing_range[0] = float(input('minimum photon_spacing = '))
    photon_spacing_range[1] = float(input('maximum photon_spacing = '))

photon_spacing_mask = np.zeros((128, 512), dtype=bool)
dset_photon_spacing_mask = masks.create_dataset("photon_spacing", shape=(128, 512), dtype=bool)

photon_spacing_mean = np.mean(photon_spacing[~combined_mask[175,::]])
photon_spacing_std = np.std(photon_spacing[~combined_mask[175,::]])

photon_spacing_mask[( photon_spacing < (photon_spacing_mean - (photon_spacing_range[0] * photon_spacing_std)) )] = True
photon_spacing_mask[( photon_spacing > (photon_spacing_mean + (photon_spacing_range[1] * photon_spacing_std)) )] = True

dset_photon_spacing_mask[...] = photon_spacing_mask
combined_mask = np.logical_or(combined_mask, photon_spacing_mask)
print('\n\n ph. spacing percentage of masked cells: ', 100 * photon_spacing_mask.flatten().sum() / photon_spacing_mask.size)
########################################################################

"""
########################################################################
if not ('photonSpacingQualityMin' in locals()):
    photonSpacingQualityMin = 0
    axes.clear()
    axes.hist(photonSpacingQuality[~badAsicBordersMask[0, ...]], bins='sqrt')
    figure.canvas.draw()
    photonSpacingQualityMin = float(input('minimum photonSpacingQuality = '))

photonSpacingQualityMask = np.zeros((128, 512), dtype=bool)
dset_photonSpacingQualityMask = masks.create_dataset("photonSpacingQualityMask", shape=(128, 512), dtype=bool)
photonSpacingQualityMask[photonSpacingQuality <= photonSpacingQualityMin] = True

dset_photonSpacingQualityMask[...] = photonSpacingQualityMask
combined_mask = np.logical_or(combined_mask, photonSpacingQualityMask)
print('\n\n ph. spacing quality percentage of masked cells: ', 100 * photonSpacingQualityMask.flatten().sum() / photonSpacingQualityMask.size)
########################################################################
"""

dset_combined_mask[...] = combined_mask
f_out.flush()
f_out.close()
print('All masks saved in ', save_filename)

print('\n\n\nentered values:')
print('dark_offset (sigmas from mean) = \n', dark_offset_range)
print('noise = \n', noise_range)
#print('analogFitStdDevsRange = \n', analogFitStdDevsRange)
print('gains_range = \n', gains_range)
#print('analogLineOffsetsRange = \n', analogLineOffsetsRange)
print('gainbit_stds_range = \n', gainbit_stds_range)
#print('digitalSpacingsSafetyFactorsMin = \n', digitalSpacingsSafetyFactorsMin)
#print('digitalMeansRange = \n', digitalMeansRange)
#print('digitalThresholdsRange = \n', digitalThresholdsRange)
print('photon_spacing_range = \n', photon_spacing_range)
#print('photonSpacingQualityMin = \n', photonSpacingQualityMin)

combined_mask_file = h5py.File(save_filename, "r", libver='latest')
combined_mask = combined_mask_file['combined_mask'][...]
#pg.image(combined_mask.transpose(0, 2, 1))

print('\n\n percentage of masked cells: ', 100 * combined_mask.flatten().sum() / combined_mask.size)
print('\n\n\npress enter to quit')
tmp = input()
