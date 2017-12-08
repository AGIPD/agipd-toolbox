import sys
import numpy as np
import os

from process_base import AgipdProcessBase


class AgipdProcessPcdrs(AgipdProcessBase):
    def __init__(self, in_fname, out_fname, runs, use_xfel_format=False):

        self.fit_interval = None
        self.n_offsets = 2

        super().__init__(in_fname, out_fname, runs, use_xfel_format)

    def calculate_(self):
        analog, digital = self.load_data(self.in_fname)
        mc = 0
        ypix = 0
        xpix = 0

        print()
        try:
            x = np.arange(*self.fit_interval[0])
            y = analog[slice(*self.fit_interval[0]),
                       mc, ypix, xpix].astype(np.float)
            res = self.fit_linear(x, y)

            print("slope", res[0][0])
            print("offset", res[0][1])

        except:
            print("memcell, xpix, ypix", mc, ypix, xpix)
            print("analog.shape", analog.shape)
            raise
        print()

    def initiate(self):
        self.n_ypixs = self.n_rows
        self.n_xpixs = self.n_cols
        # n_memcells is set in init of base class thus has to be overwritten
        # here
        # reason: in run 488, ... the data shows 67 memory cells being written
        # although only 64 contain actual usefull data
        # TODO what happens if data is processed for all 67?
        #self.n_memcells = 74
        self.n_memcells = 32
        print("n_memcell={}, n_ypixs={}, n_xpixs={}"
              .format(self.n_memcells, self.n_xpixs, self.n_ypixs))

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
            "slope": {
                "data": np.empty(self.shapes["offset"]),
                "path": "slope",
                "type": np.float
            },
            "threshold": {
                "data": np.empty(self.shapes["threshold"]),
                "path": "threshold",
                "type": np.float
            }
        }

    def determine_fit_intervals(self):
        self.fit_interval = [[42,122], [402,552]]
#        self.fit_interval = [[40,120], [400,550]]

    def calculate(self):
        analog, digital = self.load_data(self.in_fname)

        self.determine_fit_intervals()

        print("Start fitting")
        for i in range(self.n_offsets):
            for mc in range(self.n_memcells):
                print("gain stage {}, memcell {}".format(i, mc))
                for ypix in range(self.n_ypixs):
                    for xpix in range(self.n_xpixs):
                        try:
                            x = np.arange(*self.fit_interval[i])
                            y = analog[slice(*self.fit_interval[i]),
                                       mc, ypix, xpix].astype(np.float)
                            res = self.fit_linear(x, y)

                            gain_mean = np.mean(
                                digital[slice(*self.fit_interval[i]),
                                        mc, ypix, xpix])

                            result = self.result
                            idx = (i, mc, ypix, xpix)
                            result["slope"]["data"][idx] = res[0][0]
                            result["offset"]["data"][idx] = res[0][1]
                            result["gainlevel_mean"]["data"][idx] = gain_mean
                        except:
                            print("memcell, xpix, ypix", mc, ypix, xpix)
                            print("analog.shape", analog.shape)
                            print("res", res)
                            raise

        print("Calculate threshold")
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
    run_list = ["r0488-r0489-r0490-r0491-r0492-r0493-r0494-r0495"]
    run_type = "pcdrs"

    use_xfel_format = False
#    use_xfel_format = True

    today = str(date.today())

    number_of_runs = 1
    channels_per_run = 1
#    number_of_runs = 2
#    channeld_per_run = 16//number_of_runs
    process_list = []
    for j in range(number_of_runs):
        for i in range(channels_per_run):
            channel = j * channels_per_run + i
            print("channel", channel)

            in_file_name = (run_list[0].upper() +
                            "-AGIPD{:02d}-gathered.h5".format(channel))
            in_fname = os.path.join(in_base_dir,
                                    run_list[0],
                                    "gather",
                                    in_file_name)

            out_dir = os.path.join(out_base_dir, run_type)
            utils.create_dir(out_dir)

            if use_xfel_format:
                fname = ("{}_AGIPD{:02d}_xfel_{}.h5"
                         .format(run_type, channel, today))
            else:
                fname = ("{}_AGIPD{:02d}_agipd_{}.h5"
                         .format(run_type, channel, today))

            out_fname = os.path.join(out_dir, fname)

            p = multiprocessing.Process(target=AgipdProcessPcdrs,
                                        args=(in_fname,
                                              out_fname,
                                              run_list,
                                              use_xfel_format))
            p.start()
            process_list.append(p)

        for p in process_list:
            p.join()
