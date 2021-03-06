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
"""

import numpy as np
import os
import sys
import unittest

try:
    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
except:
    CURRENT_DIR = os.path.dirname(os.path.realpath('__file__'))

BASE_PATH = os.path.dirname(os.path.dirname(CURRENT_DIR))
SRC_PATH = os.path.join(BASE_PATH, "src")
GATHER_PATH = os.path.join(SRC_PATH, "gather")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

if GATHER_PATH not in sys.path:
    sys.path.insert(0, GATHER_PATH)

import utils  # noqa E402

from gather_base import GatherBase  # noqa E402
from gather_drspc import GatherDrspc  # noqa E402


def call_cfel_mode(measurement,
                   in_dir,
                   out_dir,
                   module,
                   runs,
                   asic,
                   max_part,
                   meas_spec):

        use_xfel_format = False  # input_format

        if measurement == "dark":
            Gather = GatherBase
        elif measurement == "drspc":
            Gather = GatherDrspc

        in_file_name = ("{}*_{}_{}_"
                        .format(module, measurement, meas_spec)
                        + "{run_number:05}_part{part:05}.nxs")
        in_fname = os.path.join(in_dir, in_file_name)

        out_dir = os.path.join(out_dir, "gather")
#        utils.create_dir(out_dir)

        if asic is None:
            out_file_name = ("{}_{}_{}.h5"
                             .format(module.split("_")[0],
                                     measurement,
                                     meas_spec))
        else:
            out_file_name = ("{}_{}_{}_asic{:02}.h5"
                             .format(module.split("_")[0],
                                     measurement,
                                     meas_spec,
                                     asic))
        out_fname = os.path.join(out_dir, out_file_name)

        use_interleaved = True
        properties = {
            "measurement": measurement,
            "n_rows_total": 128,
            "n_cols_total": 512,
            "max_pulses": 704,
            "n_memcells": 352
        }

        print("Used parameters:")
        print("in_fname=", in_fname)
        print("out_fname=", out_fname)
        print("runs=", runs)
        print("use_interleaved", use_interleaved)
        print("max_part=", max_part)
        print("asic=", asic)
        print("use_xfel_format=", use_xfel_format)
        print()

        obj = Gather(in_fname=in_fname,
                     out_fname=out_fname,
                     runs=runs,
                     properties=properties,
                     use_interleaved=use_interleaved,
                     max_part=max_part,
                     asic=asic,
                     layout="cfel_layout")
        return obj


class GatherBaseCfelTests(unittest.TestCase):

    # per test
    def setUp(self):
        measurement = "dark"

        in_base_path = "/gpfs/cfel/fsds/labs/agipd/calibration"
        # with frame loss
#        in_subdir = "raw/317-308-215-318-313/temperature_m15C/dark"
#        module = "M317"
#        runs = ["00001"]

        # no frame loss
        in_subdir = "raw/315-304-309-314-316-306-307/temperature_m25C/dark"
        module = "M304"
        runs = [12]

        asic = None  # asic (None means full module)
#        asic = 1

        max_part = False
#        out_base_path = ("/gpfs/exfel/exp/SPB/201730/p900009/scratch/user/"
#                          "kuhnm")
#        out_subdir = "tmp/cfel"
        meas_spec = "tint150ns"

        in_dir = os.path.join(in_base_path, in_subdir)
        out_dir = ("/gpfs/exfel/exp/SPB/201730/p900009/scratch/user/kuhnm/"
                   "tmp/cfel")

        self.gather_obj = call_cfel_mode(measurement=measurement,
                                         in_dir=in_dir,
                                         out_dir=out_dir,
                                         module=module,
                                         runs=runs,
                                         asic=asic,
                                         max_part=max_part,
                                         meas_spec=meas_spec)

    def test_pos_indices(self):

        run_idx = 1
        # reference values
        ref_row = slice(None)
        ref_col = slice(None)

        asis = None

        # get result calculated by set_pos_indices
        res = self.gather_obj.set_pos_indices(run_idx, asic)
        res = res[0]

        # compare result to reference
        self.assertTrue(np.all(res[0] == ref_row))
        self.assertTrue(np.all(res[1] == ref_col))

#    def test_class(self):
#        self.gather_obj.run()


if __name__ == "__main__":
    # Run only the tests in the specified classes

    test_classes_to_run = [
        GatherBaseCfelTests,
    ]

    loader = unittest.TestLoader()

    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)

    runner = unittest.TextTestRunner()
    results = runner.run(big_suite)
