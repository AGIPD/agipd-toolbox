import sys
import numpy as np
import os

from process_base import AgipdProcessBase


class AgipdProcessDark(AgipdProcessBase):
    def __init__(self, input_fname, output_fname, runs, use_xfel_format=False):

        self.n_offsets = None

        super().__init__(input_fname, output_fname, runs, use_xfel_format)

    def initiate(self):
        self.n_offsets = len(self.runs)

        self.shapes = {
            "offset": (self.n_offsets,
                       self.n_memcells,
                       self.n_rows,
                       self.n_cols),
            "threshold": (self.n_offsets - 1,
                          self.n_memcells,
                          self.n_rows,
                          self.n_cols)
        }

        self.result = {
            "offset": {
                "data": np.empty(self.shapes["offset"]),
                "path": "offset",
                "type": np.int16
            },
            "gainlevel_mean": {
                "data": np.empty(self.shapes["offset"]),
                "path": "gainlevel_mean",
                "type": np.int16
            },
            "stddev": {
                "data": np.empty(self.shapes["offset"]),
                "path": "stddev",
                "type": np.int16
            },
            "threshold": {
                "data": np.empty(self.shapes["threshold"]),
                "path": "threshold",
                "type": np.float
            }
        }

    def calculate(self):
        for i, run_number in enumerate(self.runs):
            input_fname = self.input_fname.format(run_number=run_number)

            print("Start loading data from", input_fname)
            analog, digital = self.load_data(input_fname)
            print("Loading done")

            print("Start computing means and standard deviations")
            self.result["offset"]["data"][i, ...] = np.mean(analog, axis=0)
            self.result["gainlevel_mean"]["data"][i, ...] = np.mean(digital,
                                                                    axis=0)

            s = self.result["stddev"]["data"][i, ...]
            for cell in np.arange(self.n_memcells):
                s[cell, ...] = np.std(analog[:, cell, :, :].astype("float"),
                                      axis=0)
            print("Done computing means and standard deviations")

        t = self.result["threshold"]["data"]
        md = self.result["gainlevel_mean"]["data"]
        for i in range(self.n_offsets - 1):
            t[i, ...] = (md[i, ...] + md[i + 1, ...]) // 2

if __name__ == "__main__":
    import multiprocessing
    from datetime import date

    try:
        CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
    except:
        CURRENT_DIR = os.path.dirname(os.path.realpath('__file__'))

    BASE_PATH = os.path.dirname(os.path.dirname(CURRENT_DIR))
    SRC_PATH = os.path.join(BASE_PATH, "src")
    print("SRC_PATH", SRC_PATH)

    if SRC_PATH not in sys.path:
        sys.path.insert(0, SRC_PATH)

    import utils

    input_base_dir = "/gpfs/exfel/exp/SPB/201730/p900009/scratch/user/kuhnm"
    output_base_dir = input_base_dir
    run_list = ["0428", "0429", "0430"]

#    use_xfel_format = False
    use_xfel_format = True

    today = str(date.today())

    number_of_runs = 1
    modules_per_run = 1
#    number_of_runs = 2
#    modules_per_run = 16//number_of_runs
    process_list = []
    for j in range(number_of_runs):
        for i in range(modules_per_run):
            channel = str(j * modules_per_run + i).zfill(2)
            print("channel", channel)

            input_file_name = ("R{run_number}-" +
                               "AGIPD{}-gathered.h5".format(channel))
            input_fname = os.path.join(input_base_dir,
                                       "r{run_number}",
                                       "gather",
                                       input_file_name)

            output_dir = os.path.join(output_base_dir, "dark")
            utils.create_dir(output_dir)

            if use_xfel_format:
                fname = "dark_AGIPD{}_xfel_{}.h5".format(channel, today)
            else:
                fname = "dark_AGIPD{}_agipd_{}.h5".format(channel, today)

            output_fname = os.path.join(output_dir, fname)

            p = multiprocessing.Process(target=AgipdProcessDark,
                                        args=(input_fname,
                                              output_fname,
                                              run_list,
                                              use_xfel_format))
            p.start()
            process_list.append(p)

        for p in process_list:
            p.join()
