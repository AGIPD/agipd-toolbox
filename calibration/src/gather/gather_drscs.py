import numpy as np
import os
import sys

from gather_base import GatherBase

try:
    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
except:
    CURRENT_DIR = os.path.dirname(os.path.realpath('__file__'))

BASE_PATH = os.path.dirname(os.path.dirname(CURRENT_DIR))
SRC_PATH = os.path.join(BASE_PATH, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import utils  # noqa E402


class GatherDrscs(GatherBase):
    def __init__(self, **kwargs):

        self.n_runs = 4

        super().__init__(**kwargs)

    def set_pos_indices(self, run_idx, asic):

        pos_idxs = []

        if self.asic is not None:
            # column position at top rows
            if self.a_row_start == 0:
                start = (self.n_runs - 1) - run_idx
            else:
                # the asics of the lower rows are upside down
                start = run_idx

            pos_idx_rows = slice(None)
            pos_idx_cols = np.arange(start, self.n_cols, self.n_runs)

            pos_idxs.append([pos_idx_rows, pos_idx_cols])

        else:
            rows_top = slice(0, self.asic_size)
            start = (self.n_runs - 1) - run_idx
            cols_top = np.arange(start, self.n_cols, self.n_runs)

            pos_idxs.append([rows_top, cols_top])

            # the asics of the lower rows are upside down
            rows_bottom = slice(self.asic_size, self.n_rows_total)
            cols_bottom = np.arange(run_idx, self.n_cols, self.n_runs)

            pos_idxs.append([rows_bottom, cols_bottom])
            # returns:
            # [[all, [3, 7, 11, 15, ...]     # top half of the module
            #  [all, [0, 4, 8, 12, ...]]     # bottom half of the module

        return pos_idxs


if __name__ == "__main__":
    import multiprocessing

#    use_xfel_format = True
    use_xfel_format = False

    if use_xfel_format:
        base_path = "/gpfs/exfel/exp/SPB/201730/p900009"
        run_list = [[488, 489, 490, 491, 492, 493, 494, 495]]

        subdir = "scratch/user/kuhnm/tmp"

        number_of_runs = 1
        channels_per_run = 1
        for runs in run_list:
            process_list = []
            for j in range(number_of_runs):
                for i in range(channels_per_run):
                    channel = j * channels_per_run + i
                    in_file_name = ("RAW-R{run_number:04d}-" +
                                    "AGIPD{:02d}".format(channel) +
                                    "-S{part:05d}.h5")
                    in_fname = os.path.join(base_path,
                                            "raw",
                                            "r{run_number:04d}",
                                            in_file_name)

                    run_subdir = "r" + "-r".join(str(r).zfill(4) for r in runs)
                    out_dir = os.path.join(base_path,
                                           subdir,
                                           run_subdir,
                                           "gather")
                    utils.create_dir(out_dir)

                    out_file_name = ("{}-AGIPD{:02d}-gathered.h5"
                                     .format(run_subdir.upper(), channel))
                    out_fname = os.path.join(out_dir,
                                             out_file_name)

                    p = multiprocessing.Process(target=GatherDrscs,
                                                args=(in_fname,
                                                      out_fname,
                                                      runs,
                                                      False,  # max_part
                                                      True,  # split_asics
                                                      use_xfel_format))
                    p.start()
                    process_list.append(p)

                for p in process_list:
                    p.join()

    else:

        in_base_path = "/gpfs/cfel/fsds/labs/agipd/calibration"
        # with frame loss
#        in_subdir = "raw/317-308-215-318-313/temperature_m15C/dark"
#        module = "M317_m2"
#        runs = ["00001"]

        # no frame loss
        current = "itestc150"
        in_subdir = ("raw/315-304-309-314-316-306-307/temperature_m25C/drscs/"
                     "{}".format(current))
        module = "M304_m2"
        runs = ["col15_00019", "col26_00020", "col37_00021", "col48_00022"]
#        asic = None  # asic (None means full module)
        asic = 1

        max_part = False
        out_base_path = "/gpfs/exfel/exp/SPB/201730/p900009/scratch/user/kuhnm"
        out_subdir = "tmp"
        measurement = "drscs"
        meas_spec = {
            "dark": "tint150ns",
            "drscs": current
        }

        in_file_name = ("{}_{}_{}_"
                        .format(module,
                                measurement,
                                meas_spec[measurement])
                        + "{run_number}_part{part:05d}.nxs")
        in_fname = os.path.join(in_base_path,
                                in_subdir,
                                in_file_name)

        out_dir = os.path.join(out_base_path,
                               out_subdir,
                               "gather")
        utils.create_dir(out_dir)
        if asic is None:
            out_file_name = ("{}_{}_{}.h5"
                             .format(module.split("_")[0],
                                     measurement,
                                     meas_spec[measurement]))
        else:
            out_file_name = ("{}_{}_{}_asic{:02d}.h5"
                             .format(module.split("_")[0],
                                     measurement,
                                     meas_spec[measurement],
                                     asic))
        out_fname = os.path.join(out_dir, out_file_name)

        GatherDrscs(in_fname=in_fname,
                    out_fname=out_fname,
                    runs=runs,
                    max_part=max_part,
                    asic=asic,
                    use_xfel_format=use_xfel_format)
