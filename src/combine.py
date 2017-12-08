from __future__ import print_function

import h5py
import time
import numpy as np
import os
import sys
import utils


class Combine():
    def __init__(self, cs_fname, xray_fname, output_fname=None,
                 probe_type="Cu", use_xfel_format=False, module=None):

        self.offset_path = "/offset"
        self.slope_path = "/slope"
        self.threshold_path = "/threshold"

        self.use_old_format = True
#        self.use_old_format = False

        # old cs format
        if self.use_old_format:
            self.offset_path = "/offset/mean"
            self.slope_path = "/slope/mean"
            self.threshold_path = "/thresholds"

        self.xray_slope_path = "/photonSpacing"

        self.cs_fname = cs_fname
        self.xray_fname = xray_fname

        self.output_fname = output_fname
#        if self.output_fname is not None:
#            utils.check_file_exists(self.output_fname)

        self.use_xfel_format = use_xfel_format
        if self.use_xfel_format:
            if module is None:
                sys.exit("To convert to xfel format the channel number in "
                         "XFEL numbering has to be specified")
        self.module = int(module)

        if self.use_old_format:
            (self.n_gains,
             self.n_rows,
             self.n_cols,
             self.n_memcells) = self.get_dimensions()

            self.n_rows = 128  # in pixels
            self.n_cols = 512  # in pixels
        else:
            (self.n_gains,
             self.n_memcells,
             self.n_rows,
             self.n_cols) = self.get_dimensions()
#        self.n_rows = 128  # in pixels
#        self.n_cols = 512  # in pixels
        self.asic_size = 64  # in pixels
#        self.n_gains = 2
#        self.n_gains = 3

        print("n_gains", self.n_gains)
        print("n_memcells", self.n_memcells)
        print("n_rows", self.n_rows)
        print("n_cols", self.n_cols)

        self.xray_mem_cell = 175
        self.single_cell = False
        self.element_energy = {
            "Cu": 8,
            "Mo": 17.4
        }
        self.probe_type = probe_type

        # how the asics are located on the module
        self.asic_mapping = utils.get_asic_order()
        if self.use_old_format:
            self.n_asics = 16
        else:
            self.n_asics = 1  # no asic splitting

        #                       n_rows * n_cols]
        self.index_map = range(len(self.asic_mapping) *
                               len(self.asic_mapping[0]))

        self.rev_mapped_asic = [[] for i in np.arange(self.n_asics)]
        self.get_asic_borders()
        print("mapped_asic", self.rev_mapped_asic)

        self.ci_offset = None
        self.ci_slope = None
        self.xray_slope = None

        self.result = {}
        self.result["gain"] = np.zeros((self.n_gains,
                                        self.n_memcells,
                                        self.n_rows,
                                        self.n_cols))
        self.result["offset_diff"] = np.zeros((self.n_gains - 1,
                                               self.n_memcells,
                                               self.n_rows,
                                               self.n_cols))
        self.result["gain_ratio"] = np.zeros((self.n_gains - 1,
                                              self.n_memcells,
                                              self.n_rows,
                                              self.n_cols))
        self.result["threshold"] = None

    def get_dimensions(self):
        input_fname = self.cs_fname.format(1)
        f = h5py.File(input_fname, "r")

        try:
            s = f[self.offset_path].shape
        finally:
            f.close()

        if self.use_xfel_format:
            dims = utils.convert_to_xfel_format(self.module, s)
        else:
            dims = utils.convert_to_agipd_format(self.module, s)

        # has shape (n_gains, n_memcells, n_rows, n_cols)
        dims = s

        return dims

    def get_asic_borders(self):
        for asic in np.arange(self.n_asics):
            # read in data from xray on a per asic basis

            self.rev_mapped_asic[asic] = self.reverse_asic_mapping(asic)

    # map asic number used in code back to asic number in definition
    # (see self.asic_mapping)
    def reverse_asic_mapping(self, asic):
        return np.hstack(self.asic_mapping)[asic]

    def run(self):

        if self.use_old_format:
            self.load_cs_data()
        else:
            self.load_ci_data()

        self.load_xray_data()

        self.calc_ratios()

        self.calc_gains()
#        print(self.result["gain"])

        self.calc_offset_diffs()
#        print(self.result["offset_diff"])

        # converts nested dictionary into flat one
        # e.g. {"a": {"n":1, "m":2}} -> {"a/n":1, "a/m":2}
        self.result = utils.flatten(self.result)

        if self.output_fname:
            self.write_data()

    def load_cs_data(self):
        a_ci_offset = [[] for i in np.arange(self.n_asics)]
        a_ci_slope = [[] for i in np.arange(self.n_asics)]
        a_threshold = [[] for i in np.arange(self.n_asics)]

        for asic in np.arange(self.n_asics):
            input_fname = self.cs_fname.format(self.rev_mapped_asic[asic])
            f = h5py.File(input_fname, "r")

            try:
                offset = f[self.offset_path][()]
                slope = f[self.slope_path][()]
                threshold = f[self.threshold_path][()]
            finally:
                f.close()

            if utils.is_xfel_format(offset.shape):
                offset = utils.convert_to_xfel_format(self.module, offset)
                slope = utils.convert_to_xfel_format(self.module, slope)
                threshold = utils.convert_to_xfel_format(self.module,
                                                         threshold)

            a_ci_offset[asic] = offset
            a_ci_slope[asic] = slope
            a_threshold[asic] = threshold

        self.ci_offset = utils.concatenate_to_module(a_ci_offset)
        self.ci_slope = utils.concatenate_to_module(a_ci_slope)
        threshold = utils.concatenate_to_module(a_threshold)

        self.result["threshold"] = threshold[..., :self.n_memcells]
        # now shape = (3, 352, 128, 512)
        self.result["threshold"] = np.rollaxis(self.result["threshold"],
                                               -1, 1)
#        print(self.result["threshold"])

    def load_ci_data(self):
        a_offset = [[] for i in np.arange(self.n_asics)]
        a_slope = [[] for i in np.arange(self.n_asics)]
        a_threshold = [[] for i in np.arange(self.n_asics)]

        for asic in np.arange(self.n_asics):
            input_fname = self.cs_fname.format(self.rev_mapped_asic[asic])
            f = h5py.File(input_fname, "r")

            try:
                offset = f[self.offset_path][()]
                slope = f[self.slope_path][()]
                threshold = f[self.threshold_path][()]
            finally:
                f.close()

            if utils.is_xfel_format(offset.shape):
                offset = utils.convert_to_xfel_format(self.module, offset)
                slope = utils.convert_to_xfel_format(self.module, slope)
                threshold = utils.convert_to_xfel_format(self.module,
                                                         threshold)

            a_offset[asic] = offset
            a_slope[asic] = slope
            a_threshold[asic] = threshold

        self.ci_offset = utils.concatenate_to_module(a_offset)
        self.ci_slope = utils.concatenate_to_module(a_slope)
        threshold = utils.concatenate_to_module(a_threshold)
        print("self.ci_slope", self.ci_slope.shape)

        self.result["threshold"] = threshold
#        print(self.result["threshold"])

    def load_xray_data(self):
        f = h5py.File(self.xray_fname, "r")

        try:
            # currently there is only data for mem_cell 175 in the files
            slope = f[self.xray_slope_path][()]
        finally:
            f.close()

        # determine whether one or all memcells used in xray
        if (len(slope.shape) == 2 or
                len(slope.shape) == 3 and slope.shape[2] == 1):
            self.single_cell = True
        else:
            print("Unknown xray_slope dataset shape!")
            sys.exit(1)

        # move memory cells index to front, makes calc easier
        if len(slope.shape) > 2:
            # now shape = (1 or 352, 128, 512)
            slope = np.rollaxis(slope, -1, 0)
        print(slope.shape)

        self.xray_slope = slope

    def calc_gains(self):
        # convert xray slope from ADU to ADU/keV
        self.xray_slope = (
            self.xray_slope / self.element_energy[self.probe_type])

        if self.single_cell:
            # xray_gain_h175 / cs_gain_h175
            ci_slope = self.ci_slope[0, self.xray_mem_cell, :, :]
            # set all failed pixels to 1
            ci_slope[np.where(ci_slope == 0)] = 1
            factor = np.divide(self.xray_slope, ci_slope)

            # xray_gain_h = cs_gain_h * xray_gain_h175 / cs_gain_h175
            # np.newaxis is required to match shape, because we use same factor
            # for all memcells
#            slope = self.ci_slope[..., :self.n_memcells]
            f = factor[np.newaxis, :, :]
            for i in range(self.n_gains):
                # xray_gain_<m|l> = cs_gain_<m|l> * xray_gain_h175 / cs_gain_h175  # noqa E501
                self.result["gain"][i, ...] = np.multiply(ci_slope[0, ...], f)

        else:
            # xray gain info for all memory cells
            self.result["gain"][0, ...] = self.xray_slope
            for i in range(self.n_gains - 1):
                # gain_<m|l> = (cs_<m|l> / cs_h) * xray
                self.result["gain"][i + 1, ...] = np.multiply(self.ratios[i],
                                                              self.xray_slope)

        # move memory cells index back to original position
#        self.ci_slope = np.rollaxis(self.ci_slope, 1, 4)
#        self.result["gain"] = np.rollaxis(self.result["gain"], 1, 4)

    def calc_ratios(self):
        if self.use_old_format:
            # now shape = (3, 352, 128, 512)
            self.ci_slope = np.rollaxis(self.ci_slope, -1, 1)

        s = self.ci_slope[:, :self.n_memcells, ...]

        print("s.shape", s.shape)
        print("self.result", self.result["gain_ratio"].shape)

        mask = (s[0] == 0)

        self.ratios = [np.zeros(s[0].shape) for i in range(self.n_gains - 1)]

        for i, r in enumerate(self.ratios):
            # gain_<m|l> / gain_<h>
            r[~mask] = np.divide(s[i + 1, ...][~mask], s[0, ...][~mask])

            self.result["gain_ratio"][i, ...] = r

    def calc_offset_diffs(self):

        offset_diff = self.result["offset_diff"]
        ci_offset = self.ci_offset[..., :self.n_memcells]

        # now shape = (3, 352, 128, 512)
        ci_offset = np.rollaxis(ci_offset, -1, 1)

        for i in range(self.n_gains - 1):
            # offset_medium - offset_high resp. offset_low - offset_high
            offset_diff[i, ...] = ci_offset[i + 1, ...] - ci_offset[0, ...]

    def write_data(self):
        output_file = h5py.File(self.output_fname, "w", libver="latest")

        try:
            print("\nStart saving data")
            t = time.time()

            for key in self.result:
                output_file.create_dataset(key,
                                           data=self.result[key])

            output_file.flush()
            print("took time: {}".format(time.time() - t))
        finally:
            output_file.close()

    def get(self):
        return self.result


if __name__ == "__main__":
    base_dir = "/gpfs/cfel/fsds/labs/agipd/calibration/processed/"
    pc_base_dir = "/gpfs/exfel/exp/SPB/201730/p900009/scratch/user/kuhnm/pcdrs"
#    output_dir = "/gpfs/exfel/exp/SPB/201701/p002012/scratch/user/kuhnm/tests"
    output_dir = "/gpfs/exfel/exp/SPB/201730/p900009/scratch/user/kuhnm/tmp"
    module = "M215"
    module_pos = "m6"
    temperature = "temperature_m15C"
    probe_type = "Mo"

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

    channel = module_mapping[module]

    cs_dir = os.path.join(base_dir,
                          module,
                          temperature,
                          "drscs",
                          "merged")
    cs_filename = "{}_drscs_asic".format(module) + "{:02d}_merged.h5"

    cs_fname = os.path.join(cs_dir, cs_filename)

    pc_fname = os.path.join(pc_base_dir,
                            "pcdrs_AGIPD{:02d}_xfel.h5".format(channel))

    xray_dir = os.path.join(base_dir,
                            module,
                            temperature,
                            "xray")
    xray_filename = ("photonSpacing_{}_{}_xray_{}.h5"
                     .format(module, module_pos, probe_type))
    xray_fname = os.path.join(base_dir,
                              module,
                              temperature,
                              "xray",
                              xray_filename)
#    output_fname = os.path.join(base_dir,
#                                module,
#                                temperature,
#                                "cal_output",
#                                ("{}_{}_combined_calibration_constants.h5"
#                                 .format(module, temperature)))
    output_fname = os.path.join(output_dir,
                                "gain_AGIPD{:02d}_xfel.h5".format(channel))
#    output_fname = None

    charge_inject_fname = cs_fname
#    charge_inject_fname = pc_fname
    obj = Combine(charge_inject_fname, xray_fname, output_fname,
                  probe_type, use_xfel_format, module_mapping[module])
    obj.run()

#    result = obj.get()
#    for key in result:
#        print(key)
