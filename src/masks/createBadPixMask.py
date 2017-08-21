"""
Create bad pixel/cell masks based on general quality factors

badCellMask: overall mask (logical OR of all sub-masks)

Note: when no pixels/cells are masked, plotting throws an exception!

"""

import h5py
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pyqtgraph as pg
from string import Template
# need to tell python where to look for helpers.py
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
#print(BASE_PATH)
SRC_PATH = os.path.join(BASE_PATH, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
# now we can import functions from files in src directory
from helpers import create_dir


def create_mask(quantity, sig_range=[3, 3]):
    """
    Creates a mask with the same shape as quantity, masking cells where
    the value of quantity is outside of the acceptable range.  The acceptable
    range is defined as the median +/- (sig_range * std), where the range can
    be nonsymmetric

    Default range is +/- 3 * sigma

    """

    mask = np.zeros(quantity.shape, dtype=bool)
    
    median = np.median(quantity)
    sigma = np.std(quantity)

    mask[quantity < median - (sig_range[0] * sigma)] = True
    mask[quantity > median + (sig_range[1] * sigma)] = True

    return mask



if __name__ == "__main__":
    
    ## Input section ###############################################################

    module_id = 'M305'
    module_number = 'm8'
    temperature = 'temperature_m15C'
    itestc = 'merged'
    tint = 'tint150ns'
    element = 'Cu'
    counter = '00000'
    asic_mapping = [[16, 15, 14, 13, 12, 11, 10, 9],
                    [1,   2,  3,  4,  5,  6,  7, 8]]
    base_dir = "/gpfs/cfel/fsds/labs/agipd/calibration/processed/{}/{}".format(module_id, temperature)
    
    ## Input file names ##
    # Dynamic Range Scan (gain - current source)
    # TODO: currently only for merged!!
    gain_input_path = os.path.join(base_dir, "drscs", itestc)
    gain_input_template = Template("${p}/${m}_drscs_asic${a}_merged.h5").safe_substitute(p=gain_input_path, m=module_id)
    gain_input_template = Template(gain_input_template)
    
    # Darks
    dark_filename = "darkOffset_{}_{}_{}.h5".format(module_id, module_number, tint)
    dark_filepath = os.path.join(base_dir, 'dark', dark_filename)
    
    # X-ray
    xray_filename = "photonSpacing_{}_{}_xray_{}_{}.h5".format(module_id, module_number, element, counter)
    xray_filepath = os.path.join(base_dir, 'xray', xray_filename)
    

    # Output dirs and mask file
    output_dir = os.path.join(base_dir, "cal_output")
    plot_dir = os.path.join(output_dir, "plots")
    save_filename = os.path.join(output_dir, 'agipd_mask_{}_{}_{}_{}_{}_{}.h5'.format(module_id, module_number, temperature, itestc, tint, element))
    create_dir(output_dir)
    create_dir(plot_dir)
    f_out = h5py.File(save_filename, "w", libver='latest')
    dset_combined_mask = f_out.create_dataset("combined_mask", shape=(352, 128, 512), dtype=bool)
    masks = f_out.create_group("masks")


    ## Cut values - range of sigmas from median ##
    dark_offset_range = np.array([3, 3])
    noise_range = np.array([3, 3])
    gains_range = np.array([[3, 3], [3, 3], [3, 3]])
    gainbit_stds_range = np.array([[3, 3], [3, 3], [3, 3]])
    photon_spacing_range = np.array([3, 3])


    ########################################################################
    ##################### the rest can be left untouched ###################
    ########################################################################

    ## Read in data ##

    # Get data from gain files
    # each asic in separate file, needs to be assembled into full module
    gains_upper = []
    gains_lower = []
    gains_tmp = []
    gain_offsets_upper = []
    gain_offsets_lower = []
    gain_offsets_tmp = []
    error_code_upper = []
    error_code_lower = []
    error_code_tmp = []
    warning_code_upper = []
    warning_code_lower = []
    warning_code_tmp = []

    # Upper row
    asic_row = asic_mapping[0]
    for asic in asic_row:
        gain_input_file = gain_input_template.substitute(a=str(asic).zfill(2))
        f = h5py.File(gain_input_file, "r")
        gains_tmp = f["/slope/mean"][...] # shape: 3 x 64 x 64 x 352
        gain_offsets_tmp = f["/offset/mean"][...] # shape: 3 x 64 x 64 x 352
        error_code_tmp = f["/error_code"][...] # shape: 64 x 64 x 352
        warning_code_tmp = f["/warning_code"][...] # shape: 64 x 64 x 352
        if asic == asic_row[0]:
            gains_upper = gains_tmp
            gain_offsets_upper = gain_offsets_tmp
            error_code_upper = error_code_tmp
            warning_code_upper = warning_code_tmp
        else:
            gains_upper = np.concatenate((gains_upper, gains_tmp), axis=2)
            gain_offsets_upper = np.concatenate((gain_offsets_upper, gain_offsets_tmp), axis=2)
            error_code_upper = np.concatenate((error_code_upper, error_code_tmp), axis=1)
            warning_code_upper = np.concatenate((warning_code_upper, warning_code_tmp), axis=1)
        f.close()

    # lower row
    asic_row = asic_mapping[1]
    for asic in asic_row:
        gain_input_file = gain_input_template.substitute(a=str(asic).zfill(2))
        f = h5py.File(gain_input_file, "r")
        gains_tmp = f["/slope/mean"][...] # shape: 3 x 64 x 64 x 352
        gain_offsets_tmp = f["/offset/mean"][...] # shape: 3 x 64 x 64 x 352
        error_code_tmp = f["/error_code"][...] # shape: 64 x 64 x 352
        warning_code_tmp = f["/warning_code"][...] # shape: 64 x 64 x 352
        if asic == asic_row[0]:
            gains_lower = gains_tmp
            gain_offsets_lower = gain_offsets_tmp
            error_code_lower = error_code_tmp
            warning_code_lower = warning_code_tmp
        else:
            gains_lower = np.concatenate((gains_lower, gains_tmp), axis=2)
            gain_offsets_lower = np.concatenate((gain_offsets_lower, gain_offsets_tmp), axis=2)
            error_code_lower = np.concatenate((error_code_lower, error_code_tmp), axis=1)
            warning_code_lower = np.concatenate((warning_code_lower, warning_code_tmp), axis=1)
        f.close()

    # combine upper and lower rows into full module
    gains = np.concatenate((gains_upper, gains_lower), axis=1) # shape: 3 x 128 x 512 x 352
    gain_offsets = np.concatenate((gain_offsets_upper, gain_offsets_lower), axis=1) # shape: 3 x 128 x 512 x 352
    error_code = np.concatenate((error_code_upper, error_code_lower), axis=0) # shape: 128 x 512 x 352
    warning_code = np.concatenate((warning_code_upper, warning_code_lower), axis=0) # shape: 128 x 512 x 352
    
    # transpose to be in same order as dark and xray - for now at least
    gains = gains.transpose((0, 3, 1, 2)) # shape: 3 x 352 x 128 x 512
    gain_offsets = gain_offsets.transpose((0, 3, 1, 2)) # shape: 3 x 352 x 128 x 512
    error_code = error_code.transpose((2, 0, 1)) # 352 x 128 x 512
    warning_code = warning_code.transpose((2, 0, 1)) # 352 x 128 x 512

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


    ## Create masks ##

    # Initialize combined mask
    combined_mask = np.zeros((352, 128, 512), dtype=bool)
    mask_list = []
    ## Mask cells where fit of any quantity failed ##
    failed_fit_mask = np.zeros((352, 128, 512), dtype=bool)
    dset_failed_fit_mask = masks.create_dataset("failed_fit", shape=(352, 128, 512), dtype=bool)
    
    # for dark and xray, value is set to infinity when fit failed
    for tmp in (dark_offset, noise, photon_spacing, photon_spacing_quality):
        failed_fit_mask = np.logical_or(failed_fit_mask, ~np.isfinite(tmp))
        tmp[~np.isfinite(tmp)] = 0  # to surpress "invalid value in comparison" warnings

    # for gain info we have error codes - for now any error or warning code count as failed
    for tmp in (error_code, warning_code):
        failed_fit_mask = np.logical_or(failed_fit_mask, np.where(tmp != 0, True, False))

    dset_failed_fit_mask[...] = failed_fit_mask
    mask_list.append(failed_fit_mask)
    print('\nfailed fit % of cells masked: ', 100 * failed_fit_mask.flatten().sum() / failed_fit_mask.size)
    plt.matshow(failed_fit_mask)
    plt.savefig("{}/{}.png".format(plotdir, failed_fit))

    ## Mask cells where value outside acceptable range ##
    
    # Pedestal (dark offset)
    dset_dark_offset_mask = masks.create_dataset("dark_offset", shape=dark_offset.shape, dtype=bool)
    dark_offset_mask = create_mask(dark_offset, dark_offset_range)
    dset_dark_offset_mask[...] = dark_offset_mask
    mask_list.append(dark_offset_mask)
    print('dark_offset % of cells masked: ', 100 * dark_offset_mask.flatten().sum() / dark_offset_mask.size)
    # Noise (dark width)
    dset_noise_mask = masks.create_dataset("noise", shape=noise.shape, dtype=bool)
    noise_mask = create_mask(noise, noise_range)
    dset_noise_mask[...] = noise_mask
    mask_list.append(noise_mask)
    print('noise % of cells masked: ', 100 * noise_mask.flatten().sum() / noise_mask.size)
    # Gains (slope of drs, 3 stages)
    dset_gains_mask = masks.create_dataset("gain", shape=(3, 352, 128, 512), dtype=bool)
    gains_hi_mask = create_mask(gains[0, ...], gains_range[0])
    gains_med_mask = create_mask(gains[1, ...], gains_range[1])
    gains_low_mask = create_mask(gains[2, ...], gains_range[2])
    dset_gains_mask[0, ...] = gains_hi_mask
    dset_gains_mask[1, ...] = gains_med_mask
    dset_gains_mask[2, ...] = gains_low_mask
    mask_list.extend((gains_hi_mask, gains_med_mask, gains_low_mask))
    print('high gain range % of cells masked: ', 100 * gains_hi_mask.flatten().sum() / gains_hi_mask.size)
    print('medium gain range % of cells masked: ', 100 * gains_med_mask.flatten().sum() / gains_med_mask.size)
    print('low gain range % of cells masked: ', 100 * gains_low_mask.flatten().sum() / gains_low_mask.size)
    # Photon spacing (xray fluorescence gain)
    dset_photon_spacing_mask = masks.create_dataset("photon_spacing", shape=photon_spacing.shape, dtype=bool)
    photon_spacing_mask = create_mask(photon_spacing, photon_spacing_range)
    dset_photon_spacing_mask[...] = photon_spacing_mask
    mask_list.append(photon_spacing_mask)
    print('photon_spacing % of cells masked: ', 100 * photon_spacing_mask.flatten().sum() / photon_spacing_mask.size)


    # combine all generated masks
    for m in mask_list:
        combined_mask = np.logical_or(combined_mask, m)

    dset_combined_mask[...] = combined_mask

    f_out.flush()
    f_out.close()
    print('\nAll masks saved in ', save_filename)
    
    #combined_mask_file = h5py.File(save_filename, "r", libver='latest')
    #combined_mask = combined_mask_file['combined_mask'][...]
    #pg.image(combined_mask.transpose(0, 2, 1))

    print('\n\nTotal percentage of masked cells: ', 100 * combined_mask.flatten().sum() / combined_mask.size, '\n')
    #print('\n\n\npress enter to quit')
    #tmp = input()
