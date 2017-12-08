from __future__ import print_function

import h5py
import time
import numpy as np
import os
import sys
from string import Template
import utils


class Combine():
    def __init__(self, cs_input_template, xray_input_fname, output_fname,
                 probe_type="Cu", n_memcells=352, use_xfel_format=False,
                 module=None):

        self.offset_path = "/slope/mean"
        self.slope_path = "/slope/mean"
        self.thresholds_path = "/thresholds"
        self.xray_slope_path = "/photonSpacing"

        self.cs_input_template = cs_input_template
        self.xray_input_fname = xray_input_fname

        self.output_fname = output_fname
#        utils.check_file_exists(self.output_fname)

        self.use_xfel_format = use_xfel_format
        if self.use_xfel_format:
            if module is None:
                sys.exit("To convert to xfel format the module number in XFEL "
                         "numbering has to be specified")
        self.module = int(module)

        self.n_memcells = n_memcells
        self.module_h = 128  # in pixels
        self.module_l = 512  # in pixels
        self.asic_size = 64  # in pixels

        self.xray_mem_cell = 175
        self.single_cell = None
        self.element_energy = {
            "Cu": 8,
            "Mo": 17.4
        }
        self.probe_type = probe_type

        # how the asics are located on the module
        self.asic_mapping = [[16, 15, 14, 13, 12, 11, 10, 9],
                             [1, 2, 3, 4, 5, 6, 7, 8]]
        self.n_asics = 16

        #                       [rows, columns]
        self.asics_per_module = [len(self.asic_mapping),
                                 len(self.asic_mapping[0])]
        self.index_map = range(self.asics_per_module[0] *
                               self.asics_per_module[1])

        self.rev_mapped_asic = [[] for i in np.arange(self.n_asics)]
        self.get_asic_borders()
        print("mapped_asic", self.rev_mapped_asic)

        self.a_cs_offset = [[] for i in np.arange(self.n_asics)]
        self.a_cs_slope = [[] for i in np.arange(self.n_asics)]
        self.a_thresholds = [[] for i in np.arange(self.n_asics)]

        self.cs_offset = None
        self.cs_slope = None
        self.xray_slope = None

        self.result = {}
        self.result["gain"] = np.zeros((3,
                                        self.n_memcells,
                                        self.module_h,
                                        self.module_l))
        self.result["offset_diff"] = np.zeros((2,
                                               self.n_memcells,
                                               self.module_h,
                                               self.module_l))
        self.result["gain_ratio"] = np.zeros((2,
                                              self.n_memcells,
                                              self.module_h,
                                              self.module_l))
        self.result["thresholds"] = None

    def get_asic_borders(self):
        for asic in np.arange(self.n_asics):
            # read in data from xray on a per asic basis

            self.rev_mapped_asic[asic] = self.reverse_asic_mapping(asic)
#           self.row_border[asic], self.col_border[asic] = (
#               self.determine_asic_border(mapped_asic))

    # from a given asic number calculate the number used in code
    def calculate_mapped_asic(self, asic):
        for row_i in np.arange(len(self.asic_mapping)):
            try:
                col_i = self.asic_mapping[row_i].index(asic)
                return self.index_map[row_i * self.asics_per_module[1] + col_i]
            except:
                pass

    # map asic number used in code back to asic number in definition
    # (see self.asic_mapping)
    def reverse_asic_mapping(self, asic):
        return np.hstack(self.asic_mapping)[asic]

    def determine_asic_border(self, mapped_asic):
        #       ____ ____ ____ ____ ____ ____ ____ ____
        # 0x64 |    |    |    |    |    |    |    |    |
        #      |  0 |  1 | 2  | 3  |  4 |  5 | 6  | 7  |
        # 1x64 |____|____|____|____|____|____|____|____|
        #      |  8 |  9 | 10 | 11 | 12 | 13 | 14 | 15 |
        # 2x64 |____|____|____|____|____|____|____|____|
        #      0*64 1x64 2x64 3x64 4x64 5x64 6x64 7x64 8x64

        row_progress = int(mapped_asic / self.asics_per_module[1])
        col_progress = int(mapped_asic % self.asics_per_module[1])
        print("row_progress: {}".format(row_progress))
        print("col_progress: {}".format(col_progress))

        a_row_start = row_progress * self.asic_size
        a_row_stop = (row_progress + 1) * self.asic_size
        a_col_start = col_progress * self.asic_size
        a_col_stop = (col_progress + 1) * self.asic_size

        print("asic_size {}".format(self.asic_size))
        print("a_col_start: {}".format(a_row_start))
        print("a_row_stop: {}".format(a_row_stop))
        print("a_row_start: {}".format(a_col_start))
        print("a_row_stop: {}".format(a_col_stop))

        return [a_row_start, a_row_stop], [a_col_start, a_col_stop]

    def run(self):

        self.load_cs_data()
        self.concatenate_to_module()
#        print(self.result["thresholds"])

        self.load_xray_data()

        self.calc_ratios()

        self.calc_gains()
#        print(self.result["gain"])

        self.calc_offset_diffs()
#        print(self.result["offset_diff"])

        if self.use_xfel_format:

            keys_to_convert = ["gain", "offset_diff", "thresholds"]

            data_to_convert = []
            for key in keys_to_convert:
                data_to_convert.append(self.result[key])

            converted_data, _ = utils.convert_to_xfel_format(self.module,
                                                             data_to_convert,
                                                             [])

            for i, key in enumerate(keys_to_convert):
                self.result[key] = converted_data[i]

        self.write_data()

    def load_cs_data(self):
        for asic in np.arange(self.n_asics):
            filled_up_asic = str(self.rev_mapped_asic[asic]).zfill(2)
            input_fname = (self.cs_input_template.substitute(a=filled_up_asic))
            source_file = h5py.File(input_fname, "r")

            try:
                self.a_cs_offset[asic] = source_file[self.offset_path][()]
                self.a_cs_slope[asic] = source_file[self.slope_path][()]
                self.a_thresholds[asic] = source_file[self.thresholds_path][()]
            finally:
                source_file.close()

    def concatenate_to_module(self):
        # upper row
        asic_row = self.asic_mapping[0]

        # index goes from 0 to 15
        cs_offset_upper = self.a_cs_offset[asic_row[0] - 1]
        cs_slope_upper = self.a_cs_slope[asic_row[0] - 1]
        thresholds_upper = self.a_thresholds[asic_row[0] - 1]

        for asic in asic_row[1:]:
            cs_offset_upper = np.concatenate((cs_offset_upper,
                                             self.a_cs_offset[asic - 1]),
                                             axis=2)
            cs_slope_upper = np.concatenate((cs_slope_upper,
                                            self.a_cs_slope[asic - 1]),
                                            axis=2)
            thresholds_upper = np.concatenate((thresholds_upper,
                                              self.a_thresholds[asic - 1]),
                                              axis=2)

        # lower row
        asic_row = self.asic_mapping[1]

        cs_offset_lower = self.a_cs_offset[asic_row[0] - 1]
        cs_slope_lower = self.a_cs_slope[asic_row[0] - 1]
        thresholds_lower = self.a_thresholds[asic_row[0] - 1]

        for asic in asic_row[1:]:
            cs_offset_lower = np.concatenate((cs_offset_lower,
                                             self.a_cs_offset[asic - 1]),
                                             axis=2)
            cs_slope_lower = np.concatenate((cs_slope_lower,
                                            self.a_cs_slope[asic - 1]),
                                            axis=2)
            thresholds_lower = np.concatenate((thresholds_lower,
                                              self.a_thresholds[asic - 1]),
                                              axis=2)

        # combine them
        self.cs_offset = np.concatenate((cs_offset_upper,
                                         cs_offset_lower),
                                        axis=1)
        self.cs_slope = np.concatenate((cs_slope_upper,
                                        cs_slope_lower),
                                       axis=1)
        self.result["thresholds"] = (
            np.concatenate((thresholds_upper, thresholds_lower),
                           axis=1)[..., :self.n_memcells])
        # now shape = (3, 352, 128, 512)
        self.result["thresholds"] = np.rollaxis(self.result["thresholds"],
                                                -1, 1)

    def load_xray_data(self):
        source_file = h5py.File(self.xray_input_fname, "r")

        try:
            # currently there is only data for mem_cell 175 in the files
            self.xray_slope = source_file[self.xray_slope_path][()]
        finally:
            source_file.close()

        # determine whether one or all memcells used in xray
        if len(self.xray_slope.shape) == 2:
            self.single_cell = True
        elif len(self.xray_slope.shape) == 3:
            if self.xray_slope.shape[2] == 1:
                self.single_cell = True
        else:
            print("Unknown xray_slope dataset shape!")
            sys.exit(1)

    def calc_gains(self):
        # convert xray slope from ADU to ADU/keV
        self.xray_slope = (
            self.xray_slope / self.element_energy[self.probe_type])

        # move memory cells index to front, makes calc easier
        if len(self.xray_slope.shape) > 2:
            # now shape = (1 or 352, 128, 512)
            self.xray_slope = np.rollaxis(self.xray_slope, -1, 0)
        print(self.xray_slope.shape)

        if self.single_cell:
            # xray_gain_h175 / cs_gain_h175
            cs_slope = self.cs_slope[0, self.xray_mem_cell, :, :]
            # set all failed pixels to 1
            cs_slope[np.where(cs_slope == 0)] = 1
            factor = np.divide(self.xray_slope, cs_slope)

            # xray_gain_h = cs_gain_h * xray_gain_h175 / cs_gain_h175
            # np.newaxis is required to match shape, because we use same factor
            # for all memcells
#            slope = self.cs_slope[..., :self.n_memcells]
            f = factor[np.newaxis, :, :]
            self.result["gain"][0, ...] = np.multiply(cs_slope[0, ...], f)
            # xray_gain_m = cs_gain_m * xray_gain_h175 / cs_gain_h175
            self.result["gain"][1, ...] = np.multiply(cs_slope[1, ...], f)
            # xray_gain_l = cs_gain_l * xray_gain_h175 / cs_gain_h175
            self.result["gain"][2, ...] = np.multiply(cs_slope[2, ...], f)

        else:
            # xray gain info for all memory cells
            self.result["gain"][0, ...] = self.xray_slope
            # gain_m = (cs_m / cs_h) * xray
            self.result["gain"][1, ...] = np.multiply(self.cs_mh_ratio,
                                                      self.xray_slope)
            # gain_l = (cs_l / cs_h) * xray
            self.result["gain"][2, ...] = np.multiply(self.cs_lh_ratio,
                                                      self.xray_slope)

        # move memory cells index back to original position
#        self.cs_slope = np.rollaxis(self.cs_slope, 1, 4)
#        self.result["gain"] = np.rollaxis(self.result["gain"], 1, 4)

    def calc_ratios(self):
        # now shape = (3, 352, 128, 512)
        self.cs_slope = np.rollaxis(self.cs_slope, -1, 1)

        s = self.cs_slope[:, :self.n_memcells, ...]

        mask = (s[0] == 0)

        self.cs_mh_ratio = np.zeros(s[0].shape)
        self.cs_mh_ratio[mask] = 0
        # cs_gain_m / cs_gain_h
        self.cs_mh_ratio[~mask] = np.divide(s[1, ...][~mask], s[0, ...][~mask])

        self.cs_lh_ratio = np.zeros(s[0].shape)
        self.cs_lh_ratio[mask] = 0
        # cs_gain_l / cs_gain_h
        self.cs_lh_ratio[~mask] = np.divide(s[2, ...][~mask], s[0, ...][~mask])

#        self.cs_mh_ratio = np.divide(s[1, ...], s[0, ...])
#        self.cs_lh_ratio = np.divide(s[2, ...], s[0, ...])

        self.result["gain_ratio"][0, ...] = self.cs_mh_ratio
        self.result["gain_ratio"][1, ...] = self.cs_lh_ratio

    def calc_offset_diffs(self):

        offset_diff = self.result["offset_diff"]
        cs_offset = self.cs_offset[..., :self.n_memcells]

        # now shape = (3, 352, 128, 512)
        cs_offset = np.rollaxis(cs_offset, -1, 1)

        # offset_medium - offset_high
        offset_diff[0, ...] = cs_offset[1, ...] - cs_offset[0, ...]
        # offset_low - offset_high
        offset_diff[1, ...] = cs_offset[2, ...] - cs_offset[0, ...]

    def write_data(self):
        output_file = h5py.File(self.output_fname, "w", libver="latest")

        try:
            print("\nStart saving data")
            t = time.time()

            for key in self.result:
                if type(self.result[key]) != dict:
                    output_file.create_dataset(
                        "/{}".format(key),
                        data=self.result[key])
                else:
                    for subkey in self.result[key]:
                        if type(self.result[key][subkey]) != dict:
                            output_file.create_dataset(
                                "/{}/{}".format(key, subkey),
                                data=self.result[key][subkey])
                        else:
                            for gain in ["high", "medium", "low"]:
                                output_file.create_dataset(
                                    "/{}/{}/{}".format(key, subkey, gain),
                                    data=self.result[key][subkey][gain])

            output_file.flush()
            print("took time: {}".format(time.time() - t))
        finally:
            output_file.close()


if __name__ == "__main__":
    from datetime import date

    base_dir = "/gpfs/cfel/fsds/labs/agipd/calibration/processed/"
    output_dir = "/gpfs/exfel/exp/SPB/201701/p002012/scratch/user/kuhnm/tests"
    module = "M215"
    module_pos = "m6"
    temperature = "temperature_m15C"
    probe_type = "Mo"
#    single_cell = True
    n_memcells = 30
#    n_memcells = 352

    module_mapping = {
        "M305": 0,
        "M315": 1,
        "M314": 2,
        "M310": 3,
        "M234": 4,
        "M309": 5,
        "M300": 6,
        "M308": 7,
        "M316": 12,
        "M215": 13,
        "M317": 14,
        "M318": 15,
        "M301": 8,
        "M306": 9,
        "M307": 10,
        "M313": 11,
    }
    use_xfel_format = True

    today = str(date.today())

    cs_input_path = os.path.join(base_dir,
                                 module,
                                 temperature,
                                 "drscs")
    # substitute all except current and asic
    cs_input_template = (
        Template("${p}/merged/${m}_drscs_asic${a}_merged.h5")
        .safe_substitute(p=cs_input_path, m=module)
    )
    # make a template out of this string to let Combine set current and asic
    cs_input_template = Template(cs_input_template)

    xray_input_file_name = ("photonSpacing_{}_{}_xray_{}.h5"
                            .format(module, module_pos, probe_type))
    xray_input_fname = os.path.join(base_dir,
                                    module,
                                    temperature,
                                    "xray",
                                    xray_input_file_name)
#    output_fname = os.path.join(base_dir,
#                                module,
#                                temperature,
#                                "cal_output",
#                                ("{}_{}_combined_calibration_constants.h5"
#                                 .format(module, temperature)))
    output_fname = os.path.join(output_dir,
                                ("gain_AGIPD{}_xfel_{}.h5"
                                 .format(module_mapping[module], today)))

    obj = Combine(cs_input_template, xray_input_fname, output_fname,
                  probe_type, n_memcells, use_xfel_format,
                  module_mapping[module])
    obj.run()
