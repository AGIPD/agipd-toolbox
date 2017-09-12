from __future__ import print_function

import h5py
import time
import numpy as np
import os
from string import Template
from process import check_file_exists


class Combine():
    def __init__(self, cs_input_template, xray_input_fname, output_fname):

        self.offset_path = "/slope/mean"
        self.slope_path = "/slope/mean"
        self.thresholds_path = "/thresholds"
        self.xray_slope_path = "/photonSpacing"

        self.cs_input_template = cs_input_template
        self.xray_input_fname = xray_input_fname

        self.output_fname = output_fname
        check_file_exists(self.output_fname)

        self.mem_cells = 352
        self.module_h = 128  # in pixels
        self.module_l = 512  # in pixels
        self.asic_size = 64  # in pixels

        self.xray_mem_cell = 175
        self.element_energy = 8  # Cu

        # how the asics are located on the module
        self.asic_mapping = [[16, 15, 14, 13, 12, 11, 10, 9],
                             [1,   2,  3,  4,  5,  6,  7, 8]]
        self.n_asics = 16

        self.current = ["itestc150", "itestc150", "itestc150", "itestc150",
                        "itestc150", "itestc150", "itestc150", "itestc150",
                        "itestc150", "itestc150", "itestc150", "itestc150",
                        "itestc150", "itestc150", "itestc150", "itestc150"]

        #                       [rows, columns]
        self.asics_per_module = [len(self.asic_mapping),
                                 len(self.asic_mapping[0])]
        self.index_map = range(self.asics_per_module[0] *
                               self.asics_per_module[1])

        self.rev_mapped_asic = [[] for i in np.arange(self.n_asics)]
#        self.mapped_asic = [[] for i in np.arange(self.n_asics)]
#        self.row_border = [[] for i in np.arange(self.n_asics)]
#        self.col_border = [[] for i in np.arange(self.n_asics)]
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
                                        self.module_h,
                                        self.module_l,
                                        self.mem_cells))
        self.result["offset"] = np.zeros((2,
                                          self.module_h,
                                          self.module_l,
                                          self.mem_cells))
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
        #print(self.result["thresholds"])

        self.load_xray_data()

        self.calc_gains()
        print(self.result["gains"])

        self.calc_offsets()
        print(self.result["offsets"])

        # self.write_data()

    def load_cs_data(self):
        for asic in np.arange(self.n_asics):
            input_fname = (self.cs_input_template
                           .substitute(c=self.current[asic],
                                       a=str(self.rev_mapped_asic[asic]).zfill(2)))
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

        cs_offset_upper = self.a_cs_offset[asic_row[0] - 1] # index goes from 0 to 15
        cs_slope_upper = self.a_cs_slope[asic_row[0] - 1]
        thresholds_upper = self.a_thresholds[asic_row[0] - 1]

        for asic in asic_row[1:]:            
            cs_offset_upper = np.concatenate((cs_offset_upper,
                                             self.a_cs_offset[asic - 1]),
                                             axis=1)
            cs_slope_upper = np.concatenate((cs_slope_upper,
                                            self.a_cs_slope[asic - 1]),
                                            axis=1)
            thresholds_upper = np.concatenate((thresholds_upper,
                                              self.a_thresholds[asic - 1]),
                                              axis=1)

        # lower row
        asic_row = self.asic_mapping[1]

        cs_offset_lower = self.a_cs_offset[asic_row[0] - 1]
        cs_slope_lower = self.a_cs_slope[asic_row[0] - 1]
        thresholds_lower = self.a_thresholds[asic_row[0] - 1]

        for asic in asic_row[1:]:
            cs_offset_lower = np.concatenate((cs_offset_lower,
                                             self.a_cs_offset[asic - 1]),
                                             axis=1)
            cs_slope_lower = np.concatenate((cs_slope_lower,
                                            self.a_cs_slope[asic - 1]),
                                            axis=1)
            thresholds_lower = np.concatenate((thresholds_lower,
                                              self.a_thresholds[asic - 1]),
                                              axis=1)

        # combine them
        self.cs_offset = np.concatenate((cs_offset_upper,
                                        cs_offset_lower),
                                        axis=2)
        self.cs_slope = np.concatenate((cs_slope_upper,
                                       cs_slope_lower),
                                       axis=2)
        #print(self.cs_slope.shape)
        self.result["thresholds"] = np.concatenate((thresholds_upper,
                                                   thresholds_lower),
                                                   axis=2)

    def load_xray_data(self):
        source_file = h5py.File(self.xray_input_fname, "r")

        try:
            # currently there is only data for mem_cell 175 in the files
            self.xray_slope = source_file[self.xray_slope_path][()]
        finally:
            source_file.close()

    def calc_gains(self):
        # convert xray slope from ADU to ADU/keV
        self.xray_slope = self.xray_slope / self.element_energy
        #print(self.xray_slope.shape)
        self.xray_slope = np.swapaxes(self.xray_slope, 0, 1) #reorder to match cs_slope
        #print(self.xray_slope.shape)
        #print(self.cs_slope[0, :, :, self.xray_mem_cell].shape)

        # xray_gain_h175 / cs_gain_h175
        factor = np.divide(self.xray_slope,
                           self.cs_slope[0, :, :, self.xray_mem_cell])
        print(factor.shape)
        # xray_gain_h = cs_gain_h * xray_gain_h175 / cs_gain_h175
        self.result["gain"][0, ...] = self.cs_slope[0, ...] * factor
        # xray_gain_m = cs_gain_m * xray_gain_h175 / cs_gain_h175
        self.result["gain"][1, ...] = self.cs_slope[1, ...] * factor
        # xray_gain_l = cs_gain_l * xray_gain_h175 / cs_gain_h175
        self.result["gain"][2, ...] = self.cs_slope[2, ...] * factor

    def calc_offset_diffs(self):

        offset = self.result["offset"]
        # offset_medium - offset_high
        offset[0, ...] = self.cs_offset[1, ...] - self.cs_offset[0, ...]
        # offset_low - offset_high
        offset[1, ...] = self.cs_offset[2, ...] - self.cs_offset[0, ...]

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
    base_path = "/gpfs/cfel/fsds/labs/agipd/calibration/processed/"
    module = "M314"
    temperature = "temperature_m15C"

    cs_input_path = os.path.join(base_path,
                                 module,
                                 temperature,
                                 "drscs")
    # substitute all except current and asic
    cs_input_template = (
        Template("${p}/${c}/process/${m}_drscs_${c}_asic${a}_processed.h5")
        .safe_substitute(p=cs_input_path, m=module)
    )
    # make a template out of this string to let Combine set current and asic
    cs_input_template = Template(cs_input_template)

    xray_input_fname = os.path.join(base_path,
                                    module,
                                    temperature,
                                    "xray",
                                    "photonSpacing_M314_m7_xray_Cu.h5")
    output_fname = os.path.join(base_path,
                                module,
                                temperature,
                                "cal_output",
                                ("{}_{}_combined_calibration_constants.h5"
                                 .format(module, temperature)))

    obj = Combine(cs_input_template, xray_input_fname, output_fname)
    obj.run()
