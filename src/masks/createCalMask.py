"""
Create bad pixel/cell masks based on calibration constants
- inherits from CreateMasks

combined_mask: overall mask (logical OR of all sub-masks)

"""

import h5py
import numpy as np
import os
import sys
import argparse
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
from createMaskBase import CreateMasks, read_data


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_dir",
                        type=str,
                        default="/gpfs/cfel/fsds/labs/agipd/calibration/processed",
                        help="Directory to get data from")
    parser.add_argument("--outfile_name",
                        type=str,
                        default="mask.h5",
                        help="Filename for output mask file")
    parser.add_argument("--module",
                        type=str,
                        required=True,
                        help="Module ID and position, e.g M310_m3")
    parser.add_argument("--temperature",
                        type=str,
                        required=True,
                        help="temperature to gather, e.g. temperature_30C")
    parser.add_argument("--current",
                        type=str,
                        required=True,
                        help="Current to use (e.g. itestc20) or merged")
    parser.add_argument("--tint",
                        type=str,
                        required=True,
                        help="Integration time for dark, e.g. tint150ns")
    parser.add_argument("--element",
                        type=str,
                        required=True,
                        help="Element used for xray, e.g. Cu or Mo")

    args = parser.parse_args()

    return args


def create_cal_mask(quantity, sig_range=[3, 3]):
    """
    Creates a mask from calibration data with the same shape as quantity, 
    masking cells where the value of quantity is outside of the acceptable range.  
    The acceptable range is defined as the median +/- (sig_range * std), 
    where the range can be nonsymmetric
    
    Default range is +/- 3 * sigma
    
    """
    
    mask = np.zeros(quantity.shape, dtype=bool)
    
    median = np.median(quantity)
    sigma = np.std(quantity)

    mask[quantity < median - (sig_range[0] * sigma)] = True
    mask[quantity > median + (sig_range[1] * sigma)] = True
    
    return mask


class CreateCalMasks(CreateMasks):
    
    def __init__(self, base_dir, module, temperature, outfile_name, current, tint, element, 
                 dark_offset_range=[3, 3], noise_range=[3, 3], gains_range=[[3, 3], [3, 3], [3, 3]],
                 photon_spacing_range=[3, 3]):
        
        CreateMasks.__init__(self, base_dir, module, temperature, outfile_name)

        self.current = current
        self.tint = tint
        self.element = element

        ## Input file names ##
        # Dynamic Range Scan (gain - current source)
        # TODO: currently only for merged!!
        self.gain_input_path = os.path.join(self.base_dir, "drscs", self.current)
        self.gain_input_template = Template("${p}/${m}_drscs_asic${a}_merged.h5").safe_substitute(p=self.gain_input_path, m=self.module_id)
        self.gain_input_template = Template(self.gain_input_template)
        
        # Darks
        self.dark_filename = "darkOffset_{}_{}.h5".format(self.module, self.tint)
        self.dark_filepath = os.path.join(self.base_dir, 'dark', self.dark_filename)
        
        # X-ray
        self.xray_filename = "photonSpacing_{}_xray_{}.h5".format(self.module, self.element)
        self.xray_filepath = os.path.join(self.base_dir, 'xray', self.xray_filename)

        # Sigma ranges
        self.dark_offset_range = dark_offset_range
        self.noise_range = noise_range
        self.gains_range = gains_range
        self.photon_spacing_range = photon_spacing_range
        


    def run(self):

        # Get data
        self.get_data()

        ## Mask cells where fit of any quantity failed ##
        failed_fit_mask = np.zeros((352, 128, 512), dtype=bool)
        dset_failed_fit_mask = self.masks.create_dataset("failed_fit", shape=(352, 128, 512), dtype=bool)
        
        # for dark and xray, value is set to infinity when fit failed
        for tmp in (dark_offset, noise, photon_spacing, photon_spacing_quality):
            failed_fit_mask = np.logical_or(failed_fit_mask, ~np.isfinite(tmp))
            tmp[~np.isfinite(tmp)] = 0  # to surpress "invalid value in comparison" warnings
            
        # for gain info we have error codes - for now any error or warning code count as failed
        for tmp in (error_code, warning_code):
            failed_fit_mask = np.logical_or(failed_fit_mask, np.where(tmp != 0, True, False))
            
        dset_failed_fit_mask[...] = failed_fit_mask
        self.mask_list.append(failed_fit_mask)
        print('\nfailed fit % of cells masked: ', 100 * failed_fit_mask.flatten().sum() / failed_fit_mask.size)
        plt.matshow(failed_fit_mask)
        plt.savefig("{}/{}.png".format(plotdir, failed_fit))
                
        ## Mask cells where value outside acceptable range ##
                
        # Pedestal (dark offset)
        dset_dark_offset_mask = self.masks.create_dataset("dark_offset", shape=dark_offset.shape, dtype=bool)
        dark_offset_mask = create_mask(dark_offset, dark_offset_range)
        dset_dark_offset_mask[...] = dark_offset_mask
        self.mask_list.append(dark_offset_mask)
        print('dark_offset % of cells masked: ', 100 * dark_offset_mask.flatten().sum() / dark_offset_mask.size)

        # Noise (dark width)
        dset_noise_mask = self.masks.create_dataset("noise", shape=noise.shape, dtype=bool)
        noise_mask = create_mask(noise, noise_range)
        dset_noise_mask[...] = noise_mask
        self.mask_list.append(noise_mask)
        print('noise % of cells masked: ', 100 * noise_mask.flatten().sum() / noise_mask.size)

        # Gains (slope of drs, 3 stages)
        dset_gains_mask = self.masks.create_dataset("gain", shape=(3, 352, 128, 512), dtype=bool)
        gains_hi_mask = create_mask(gains[0, ...], gains_range[0])
        gains_med_mask = create_mask(gains[1, ...], gains_range[1])
        gains_low_mask = create_mask(gains[2, ...], gains_range[2])
        dset_gains_mask[0, ...] = gains_hi_mask
        dset_gains_mask[1, ...] = gains_med_mask
        dset_gains_mask[2, ...] = gains_low_mask
        self.mask_list.extend((gains_hi_mask, gains_med_mask, gains_low_mask))
        print('high gain range % of cells masked: ', 100 * gains_hi_mask.flatten().sum() / gains_hi_mask.size)
        print('medium gain range % of cells masked: ', 100 * gains_med_mask.flatten().sum() / gains_med_mask.size)
        print('low gain range % of cells masked: ', 100 * gains_low_mask.flatten().sum() / gains_low_mask.size)

        # Photon spacing (xray fluorescence gain)
        dset_photon_spacing_mask = self.masks.create_dataset("photon_spacing", shape=photon_spacing.shape, dtype=bool)
        photon_spacing_mask = create_mask(photon_spacing, photon_spacing_range)
        dset_photon_spacing_mask[...] = photon_spacing_mask
        self.mask_list.append(photon_spacing_mask)
        print('photon_spacing % of cells masked: ', 100 * photon_spacing_mask.flatten().sum() / photon_spacing_mask.size)


        # combine all generated masks
        self.combine_masks()

        f_out.flush()
        f_out.close()
        print('\nAll masks saved in ', self.outfile_name)
        

    def get_data(self):
        
        gains, gain_offsets, error_code, warning_code = self.get_gain_data()
        dark_dset_list = ("darkOffset", "darkStandardDeviation")
        dark_offset, noise = read_data(self.dark_file, dark_dset_list)
        xray_dset_list = ("/photonSpacing")
        photon_spacing = read_data(self.xray_file, xray_dset_list)


    def get_gain_data(self):
        
        dset_list = ("/slope/mean", "/offset/mean", "/error_code", "/warning_code")

        gains_top, gain_offsets_top, error_code_top, warning_code_top = self.create_row(self.asic_mapping[0], dset_list)
        gains_bot, gain_offsets_bot, error_code_bot, warning_code_bot = self.create_row(self.asic_mapping[1], dset_list)

        # combine upper and lower rows into full module
        gains = np.concatenate((gains_top, gains_bot), axis=1) # shape: 3 x 128 x 512 x 352
        gain_offsets = np.concatenate((gain_offsets_top, gain_offsets_bot), axis=1) # shape: 3 x 128 x 512 x 352
        error_code = np.concatenate((error_code_top, error_code_bot), axis=0) # shape: 128 x 512 x 352
        warning_code = np.concatenate((warning_code_top, warning_code_bot), axis=0) # shape: 128 x 512 x 352
        
        # transpose to be in same order as dark and xray - for now at least
        gains = gains.transpose((0, 3, 1, 2)) # shape: 3 x 352 x 128 x 512
        gain_offsets = gain_offsets.transpose((0, 3, 1, 2)) # shape: 3 x 352 x 128 x 512
        error_code = error_code.transpose((2, 0, 1)) # 352 x 128 x 512
        warning_code = warning_code.transpose((2, 0, 1)) # 352 x 128 x 512

        return gains, gain_offsets, error_code, warning_code


    def create_row(self, asic_list, dset_list):

        for asic in asic_list:
            gain_input_file = self.gain_input_template.substitute(a=str(asic).zfill(2))
            dsets_gain = read_data(gain_input_file, dset_list)
            gains_tmp = dsets_gain[0]
            gain_offsets_tmp = dsets_gain[1]
            error_code_tmp = dsets_gain[2]
            warning_code_tmp = dsets_gain[3]

            if asic == asic_row[0]:
                gains_row = gains_tmp
                gain_offsets_row = gain_offsets_tmp
                error_code_row = error_code_tmp
                warning_code_row = warning_code_tmp
            else:
                gains_row = np.concatenate((gains_upper, gains_tmp), axis=2)
                gain_offsets_row = np.concatenate((gain_offsets_upper, gain_offsets_tmp), axis=2)
                error_code_row = np.concatenate((error_code_upper, error_code_tmp), axis=1)
                warning_code_row = np.concatenate((warning_code_upper, warning_code_tmp), axis=1)
            f.close()

        return gains_row, gain_offsets_row, error_code_row, warning_code_row






if __name__ == "__main__":

    args = get_arguments()

    base_dir = args.base_dir
    outfile_name = args.outfile_name
    module = args.module
    temperature = args.temperature
    current = args.current
    tint = args.tint
    element = args.element

    print("Configured parameter:")
    print("base_dir: ", base_dir)
    print("outfile_name: ", outfile_name)
    print("module: ", module)
    print("temperature: ", temperature)
    print("current: ", current)
    print("tint: ", tint)
    print("element: ", element)


    obj = CreateCalMasks(base_dir, module, temperature, outfile_name, current, tint, element)

    obj.run()




"""

    obj = CreateColormaps(input_template, output_template, module, current,
                          plot_dir, n_processes, gain_name, matrix_type,
                          mem_cell_list, individual_plots)

    obj.run()
"""
