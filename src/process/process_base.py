import h5py
import sys
import numpy as np
import time
import os

try:
    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
except:
    CURRENT_DIR = os.path.dirname(os.path.realpath('__file__'))

BASE_PATH = os.path.dirname(os.path.dirname(CURRENT_DIR))
SRC_PATH = os.path.join(BASE_PATH, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import utils  # noqa E402


class AgipdProcessBase():
    def __init__(self, input_fname, output_fname, runs, use_xfel_format=False):

        self.input_fname = input_fname
        self.output_fname = output_fname
        self.use_xfel_format = use_xfel_format

        self.runs = runs

        # TODO extract n_cols and n_rows from raw_shape
        self.n_rows = 128
        self.n_cols = 512

#        n_xpixs = 128
#        n_ypixs = 512

        f = None
        try:
            input_fname = self.input_fname.format(run_number=self.runs[0])
            f = h5py.File(input_fname, "r")
            self.n_memcells = f["analog"].shape[1]
        finally:
            if f is not None:
                f.close()

        self.shapes = {}
        self.result = {}

        input_fname = self.input_fname.format(run_number=self.runs[0])
        self.channel = int(input_fname.rsplit("/", 1)[1].split("AGIPD")[1][:2])
        self.in_wing2 = utils.located_in_wing2(self.channel)

        print("\n\n\nStart process_dark")
        print("input_fname = ", self.input_fname)
        print("output_fname = ", self.output_fname, "\n")

        self.run()

    def load_data(self, input_fname):
        f = h5py.File(input_fname, "r")
        analog = f["analog"][()]
        digital = f["digital"][()]
        f.close()

        return analog, digital

    def initiate(self):
        pass

    def run(self):

        total_time = time.time()

        self.initiate()

        self.calculate()

        if self.use_xfel_format:
            self.convert_to_xfel_format()

        self.write_data()

        print('ProcessDark took time:  ', time.time() - total_time, '\n\n')

    def calculate(self):
        pass

    def fit_linear(self, x, y):
        # find out if the col was effected by frame loss
        lost_frames = np.where(y == 0)
        y[lost_frames] = np.NAN

        # remove the ones with frameloss
        missing = np.isnan(y)
        y = y[~missing]
        x = x[~missing]

        number_of_points = len(x)
        A = np.vstack([x, np.ones(number_of_points)]).T

        # lstsq returns: Least-squares solution (i.e. slope and offset),
        #                residuals,
        #                rank,
        #                singular values
        res = np.linalg.lstsq(A, y)

        return res

    def convert_to_xfel_format(self):
        print("Convert to XFEL format")

        for key in self.result:
            self.result[key]["data"] = (
                utils.convert_to_xfel_format(self.channel,
                                             self.result[key]["data"]))

        for key in self.shapes:
            self.shapes[key] = (
                utils.convert_to_xfel_format(self.channel,
                                             self.shapes[key]))

    def write_data(self):
        print("Start saving results at", self.output_fname)

        f = h5py.File(self.output_fname, "w", libver="latest")

        for key in self.result:
            f.create_dataset(self.result[key]["path"],
                             data=self.result[key]["data"],
                             dtype=self.result[key]["type"])

        # convert into unicode
        self.runs = [run.encode('utf8') for run in self.runs]
        f.create_dataset("run_number", data=self.runs)

        f.flush()
        f.close()
        print("Saving done")
