import h5py
import sys
import numpy as np
import time
import os
from datetime import date

try:
    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
except:
    CURRENT_DIR = os.path.dirname(os.path.realpath('__file__'))

BASE_PATH = os.path.dirname(os.path.dirname(CURRENT_DIR))
SRC_PATH = os.path.join(BASE_PATH, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import utils  # noqa E402


class AgipdProcessBase(object):
    def __init__(self, in_fname, out_fname, runs, use_xfel_format=False):

        self._out_fname = out_fname
        self._use_xfel_format = use_xfel_format

        # public attributes for use in inherited classes
        self.in_fname = in_fname

        self.runs = runs

        # TODO extract n_cols and n_rows from raw_shape
        self.n_rows = 128
        self.n_cols = 512

        in_fname = self.in_fname.format(run_number=self.runs[0])
        with h5py.File(in_fname, "r") as f:
            self.n_memcells = f['analog'].shape[1]

        self.shapes = {}
        self.result = {}

        in_fname = self.in_fname.format(run_number=self.runs[0])

        self.module, self.channel = self._get_module_and_channel(in_fname)

        print("\n\n\nStart process")
        print("in_fname:", self.in_fname)
        print("out_fname:", self._out_fname)
        print("module, channel:", self.module, self.channel)
        print()

        self.run()

    def _get_module_and_channel(self, in_fname):

        with h5py.File(in_fname, "r") as f:
            module = f['module'][()]
            channel = f['channel'][()]

#        self.channel = int(in_fname.rsplit("/", 1)[1].split("AGIPD")[1][:2])

        return module, channel

    def load_data(self, in_fname):
        with h5py.File(in_fname, "r") as f:
            analog = f['analog'][()]
            digital = f['digital'][()]

        return analog, digital

    def initiate(self):
        pass

    def run(self):

        total_time = time.time()

        self.initiate()

        self.calculate()

        if self._use_xfel_format:
            self.convert_to_xfel_format()

        print("Start saving results at {} ... ".format(self._out_fname), end='')
        self.write_data()
        print("Done.")

        print("Process took time: {}\n\n", time.time() - total_time)

    def get_mask(self, analog, digital):

        # find out if the col was effected by frame loss
        return (analog == 0)

    def mask_out_problems(self, analog, digital, mask=None):

        if mask is None:
            mask = self.get_mask(analog, digital)

        # remove the ones with frameloss
        m_analog = np.ma.masked_array(data=analog, mask=mask)
        m_digital = np.ma.masked_array(data=digital, mask=mask)

        return m_analog, m_digital

    def calculate(self):
        pass

    def fit_linear(self, x, y, mask=None):
        if mask is None:
            y_masked = y
            x_masked = x
        else:
            y_masked = y[~mask]
            x_masked = x[~mask]

        number_of_points = len(x)
        A = np.vstack([x_masked, np.ones(number_of_points)]).T

        # lstsq returns: Least-squares solution (i.e. slope and offset),
        #                residuals,
        #                rank,
        #                singular values
        res = np.linalg.lstsq(A, y_masked)

        return res

    def fit_linear_old(self, x, y):
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
            self.result[key]['data'] = (
                utils.convert_to_xfel_format(self.channel,
                                             self.result[key]['data']))

        for key in self.shapes:
            self.shapes[key] = (
                utils.convert_to_xfel_format(self.channel,
                                             self.shapes[key]))

    def write_data(self):
        with  h5py.File(self._out_fname, "w", libver='latest') as f:
            for key in self.result:
                f.create_dataset(self.result[key]['path'],
                                 data=self.result[key]['data'],
                                 dtype=self.result[key]['type'])

            # convert into unicode
            if type(self.runs[0]) == str:
                used_run_numbers = [run.encode('utf8') for run in self.runs]
            else:
                used_run_numbers = ["r{:04d}".format(run).encode('utf8')
                                    for run in self.runs]

            today = str(date.today())
            metadata_base_path = "collection"

            f.create_dataset("{}/run_number".format(metadata_base_path),
                             data=used_run_numbers)
            f.create_dataset("{}/creation_date".format(metadata_base_path),
                             data=today)

            f.flush()
