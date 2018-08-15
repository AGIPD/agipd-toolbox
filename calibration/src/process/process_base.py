# (c) Copyright 2017-2018 DESY, FS-DS
#
# This file is part of the FS-DS AGIPD toolbox.
#
# This software is free: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this software.  If not, see <http://www.gnu.org/licenses/>.

"""
@author: Manuela Kuhn <manuela.kuhn@desy.de>
         Jennifer Poehlsen <jennifer.poehlsen@desy.de>
"""

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

CALIBRATION_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
SRC_DIR = os.path.join(CALIBRATION_DIR, "src")

BASE_DIR = os.path.dirname(CALIBRATION_DIR)
SHARED_DIR = os.path.join(BASE_DIR, "shared")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import utils  # noqa E402

if SHARED_DIR not in sys.path:
    sys.path.insert(0, SHARED_DIR)

from _version import __version__


class NotSupported(Exception):
    pass


class ProcessBase(object):
    def __init__(self, in_fname, out_fname, runs, run_name):

        self._out_fname = out_fname

        # public attributes for use in inherited classes
        self.in_fname = in_fname

        self.runs = runs
        self.run_names = run_name

        self._row_location = None
        self._col_location = None
        self._memcell_location = None
        self._frame_location = None
        self._set_data_order()

        self._set_dims_and_metadata()

        self.shapes = {}
        self.result = {}

        print("\n\n\nStart process")
        print("in_fname:", self.in_fname)
        print("out_fname:", self._out_fname)
        print("module, channel:", self.module, self.channel)
        print()

        self.run()

    def _set_data_order(self):
        """Set the locations where the data is stored

        This give the different process methods the posibility to act genericly
        to data reordering.

        """
        self._row_location = 0
        self._col_location = 1
        self._memcell_location = 2
        self._frame_location = 3

    def _set_dims_and_metadata(self):
        run_number = self.runs[0]
        run_name = self.run_names[0]

        in_fname = self.in_fname.format(run_number=run_number, run_name=run_name)
        with h5py.File(in_fname, "r") as f:
            shape = f['analog'].shape

            self.module = f['collection/module'][()]
            self.channel = f['collection/channel'][()]

        self.n_rows = shape[self._row_location]
        self.n_cols = shape[self._col_location]
        self.n_memcells = shape[self._memcell_location]

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

        print("Start saving results at {} ... ".format(self._out_fname),
              end='')
        self.write_data()
        print("Done.")

        print("Process took time: {}\n\n".format(time.time() - total_time))

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

        number_of_points = len(x_masked)
        try:
            A = np.vstack([x_masked, np.ones(number_of_points)]).T
        except:
            print("number_of_points", number_of_points)
            print("x (after masking)", x_masked)
            print("y (after masking)", y_masked)
            print("len y_masked", len(y_masked))
            raise

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

    def write_data(self):

        # convert into unicode
        if type(self.runs[0]) == str:
            used_run_numbers = [run.encode('utf8') for run in self.runs]
        else:
            used_run_numbers = ["r{:04d}".format(run).encode('utf8')
                                for run in self.runs]

        collection = {
            "run_number": used_run_numbers,
            "creation_date": str(date.today()),
            "version": __version__
        }

        with h5py.File(self._out_fname, "w", libver='latest') as f:
            for key, dset in self.result.items():
                f.create_dataset(dset['path'],
                                 data=dset['data'],
                                 dtype=dset['type'])

            prefix = "collection"
            for key, value in collection.items():
                name = "{}/{}".format(prefix, key)
                f.create_dataset(name, data=value)

            f.flush()
