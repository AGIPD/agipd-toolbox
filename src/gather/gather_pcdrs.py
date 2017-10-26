import numpy as np
import os
import sys

from gather_base import AgipdGatherBase

try:
    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
except:
    CURRENT_DIR = os.path.dirname(os.path.realpath('__file__'))

BASE_PATH = os.path.dirname(os.path.dirname(CURRENT_DIR))
SRC_PATH = os.path.join(BASE_PATH, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import utils  # noqa E402


class AgipdGatherPcdrs(AgipdGatherBase):
    def __init__(self, input_fname, output_fname, runs, max_part=False,
                 split_asics=True, use_xfel_format=False, backing_store=True):

        self.runs = runs
        self.n_runs = 8

        self.cellId_path = None

        super().__init__(input_fname, output_fname, runs, max_part,
                         split_asics, use_xfel_format, backing_store)

    def define_needed_data_paths(self):
        if self.use_xfel_format:
            self.cellId_path = os.path.join(base_path, "image/cellId")

    def set_pos_indices(self, run_idx):

        start = self.n_runs - 1 - run_idx
        stop = self.n_rows // 2
        idx_upper = np.arange(start, stop, self.n_runs)
#        print("idx_upper", idx_upper)

        start = self.n_rows // 2 + run_idx
        stop = self.n_rows
        idx_lower = np.arange(start, stop, self.n_runs)
#        print("idx_lower", idx_lower)

        pos_idx_rows = np.concatenate((idx_upper, idx_lower))
        pos_idx_cols = slice(None)

#        print("pos_idx_rows", pos_idx_rows)
#        print("pos_idx_cols", pos_idx_cols)

        return pos_idx_rows, pos_idx_cols


if __name__ == "__main__":
    import multiprocessing

    module_mapping = {
        "M305": "00",
    }

    use_xfel_format = True
#    use_xfel_format = False

    if use_xfel_format:
        base_path = "/gpfs/exfel/exp/SPB/201730/p900009"
#        run_list = [["0488"]]
#        run_list = [[488, 489]]
        run_list = [[488, 489, 490, 491, 492, 493, 494, 495]]

        subdir = "scratch/user/kuhnm"

        number_of_runs = 1
        modules_per_run = 1
        for runs in run_list:
            process_list = []
            for j in range(number_of_runs):
                for i in range(modules_per_run):
                    module = str(j * modules_per_run + i).zfill(2)
                    input_file_name = ("RAW-R{run_number:04d}-" +
                                       "AGIPD{}".format(module) +
                                       "-S{part:05d}.h5")
                    input_fname = os.path.join(base_path,
                                               "raw",
                                               "r{run_number:04d}",
                                               input_file_name)

                    run_subdir = "r" + "-r".join(str(r).zfill(2) for r in runs)
                    output_dir = os.path.join(base_path,
                                              subdir,
                                              run_subdir,
                                              "gather")
                    utils.create_dir(output_dir)

                    output_file_name = ("{}-AGIPD{}-gathered.h5"
                                        .format(run_subdir.upper(), module))
                    output_fname = os.path.join(output_dir,
                                                output_file_name)

                    p = multiprocessing.Process(target=AgipdGatherPcdrs,
                                                args=(input_fname,
                                                      output_fname,
                                                      runs,
                                                      False,  # max_part
                                                      True,  # split_asics
                                                      use_xfel_format))
                    p.start()
                    process_list.append(p)

                for p in process_list:
                    p.join()
