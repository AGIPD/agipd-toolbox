import sys
import numpy as np
import os

from process_base import AgipdProcessBase


class AgipdProcessDark(AgipdProcessBase):
    def __init__(self, in_fname, out_fname, runs, use_xfel_format=False):

        self.n_offsets = None

        super().__init__(in_fname, out_fname, runs, use_xfel_format)

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
            in_fname = self.in_fname.format(run_number=run_number)

            print("Start loading data from", in_fname)
            analog, digital = self.load_data(in_fname)
            print("Loading done")

            m_analog, m_digital = self.mask_out_problems(analog=analog,
                                                         digital=digital)

            print("Start computing means and standard deviations")
            offset = np.mean(m_analog, axis=0).astype(np.int)
            gainlevel_mean = np.mean(m_digital, axis=0).astype(np.int)

            self.result["offset"]["data"][i, ...] = offset
            self.result["gainlevel_mean"]["data"][i, ...] = gainlevel_mean

            s = self.result["stddev"]["data"][i, ...]
            for cell in np.arange(self.n_memcells):
                s[cell, ...] = m_analog[:, cell, :, :].std(axis=0)
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

    in_base_dir = "/gpfs/exfel/exp/SPB/201730/p900009/scratch/user/kuhnm"
    out_base_dir = in_base_dir
    run_list = [428, 429, 430]

#    use_xfel_format = False
    use_xfel_format = True

    today = str(date.today())

    number_of_runs = 1
    channels_per_run = 1
#    number_of_runs = 2
#    channels_per_run = 16//number_of_runs
    process_list = []
    for j in range(number_of_runs):
        for i in range(channels_per_run):
            channel = j * channels_per_run + i
            print("channel", channel)

            in_file_name = ("R{run_number:04d}-" +
                            "AGIPD{:02d}-gathered.h5".format(channel))
            in_fname = os.path.join(in_base_dir,
                                    "r{run_number:04d}",
                                    "gather",
                                    in_file_name)

            out_dir = os.path.join(out_base_dir, "dark")
            utils.create_dir(out_dir)

            if use_xfel_format:
                fname = "dark_AGIPD{:02d}_xfel_{}.h5".format(channel, today)
            else:
                fname = "dark_AGIPD{:02d}_agipd_{}.h5".format(channel, today)

            out_fname = os.path.join(out_dir, fname)

            print("in_fname=", in_fname)
            print("out_fname", out_fname)
            print("runs", run_list)
            print("use_xfel_format=", use_xfel_format)
            p = multiprocessing.Process(target=AgipdProcessDark,
                                        args=(in_fname,
                                              out_fname,
                                              run_list,
                                              use_xfel_format))
            p.start()
            process_list.append(p)

        for p in process_list:
            p.join()
