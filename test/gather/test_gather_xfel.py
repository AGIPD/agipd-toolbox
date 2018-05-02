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

from gather_base import GatherBase  # noqa E402
from gather_pcdrs import GatherPcdrs  # noqa E402


def call_xfel_mode(measurement,
                   base_dir,
                   runs,
                   subdir,
                   channel,
                   asic):

        use_xfel_format = True  # input_format
        use_interleaved = True

        if measurement == "dark":
            Gather = GatherBase
        elif measurement == "pcdrs":
            Gather = GatherPcdrs

        in_file_name = ("RAW-R{run_number:04}-" +
                        "AGIPD{:02}".format(channel) +
                        "-S{part:05}.h5")
        in_fname = os.path.join(base_dir,
                                "raw",
                                "r{run_number:04}",
                                in_file_name)
        print("in_fname", in_fname)

        run_subdir = "r" + "-r".join(str(r).zfill(4) for r in runs)
        out_dir = os.path.join(base_dir,
                               subdir,
                               measurement,
                               run_subdir,
                               "gather")
#        utils.create_dir(out_dir)

        preproc_fname = os.path.join(base_dir,
                                     subdir,
                                     measurement,
                                     "r{run:04}",
                                     "R{run:04}-preprocessing.result")
        print("preproc_fname", preproc_fname)

        out_file_name = ("{}-AGIPD{:02}-gathered.h5"
                         .format(run_subdir.upper(), channel))
        out_fname = os.path.join(out_dir,
                                 out_file_name)
        print("out_fname", out_fname)

        properties = {
            "measurement": measurement,
            "n_rows_total": 128,
            "n_cols_total": 512,
            "max_pulses": 704,
            "n_memcells": 352
        }

        print("Used parameter for {} run:".format(measurement))
        print("in_fname=", in_fname)
        print("out_fname", out_fname)
        print("runs", runs)
        print("use_interleaved", use_interleaved)
        print("preproc_fname", preproc_fname)
        print("use_xfel_format=", use_xfel_format)
        print()

        obj = Gather(in_fname=in_fname,
                     out_fname=out_fname,
                     runs=runs,
                     properties=properties,
                     use_interleaved=use_interleaved,
                     preproc_fname=preproc_fname,
                     max_part=False,
                     asic=None,
                     layout="xfel_layout_2017")

        return obj


class GatherBaseXfelTests(unittest.TestCase):

    # per test
    def setUp(self):
        measurement = "dark"

        # base_dir = "/gpfs/exfel/exp/SPB/201730/p900009"
        # run_list = [["0428"], ["0429"], ["0430"]]

        base_dir = "/gpfs/exfel/exp/SPB/201730/p900009"
        runs = ["0819"]
        # runs = ["0488"]

        # base_dir = "/gpfs/exfel/exp/SPB/201701/p002012"
        # runs = ["0007"]

        subdir = "scratch/user/kuhnm/tmp"

        channel = 1
        asic = None

        self.gather_obj = call_xfel_mode(measurement=measurement,
                                         base_dir=base_dir,
                                         runs=runs,
                                         subdir=subdir,
                                         channel=channel,
                                         asic=asic)

    def test_pos_indices(self):

        run_idx = 1
        # reference values
        ref_row = slice(None)
        ref_col = slice(None)

        asic = None

        # get result calculated by set_pos_indices
        res = self.gather_obj.set_pos_indices(run_idx, asic)
        res = res[0]

        # compare result to reference
        self.assertTrue(np.all(res[0] == ref_row))
        self.assertTrue(np.all(res[1] == ref_col))

#    def test_class(self):
#        self.gather_obj.run()


class GatherPcdrsXfelTests(unittest.TestCase):

    # per test
    def setUp(self):
        measurement = "pcdrs"

        base_dir = "/gpfs/exfel/exp/SPB/201730/p900009"
#        runs = [488]
#        runs = [488, 489]
        runs = [488, 489, 490, 491, 492, 493, 494, 495]
        runs = [709, 710, 711, 712, 713, 714, 715, 716]

        subdir = "scratch/user/kuhnm/testing"

        channel = 1
        asic = None

        self.gather_obj = call_xfel_mode(measurement=measurement,
                                         base_dir=base_dir,
                                         runs=runs,
                                         subdir=subdir,
                                         channel=channel,
                                         asic=asic)

    def test_pos_indices1(self):

        run_idx = 1
        # reference values
        ref_row = [6, 14, 22, 30, 38, 46, 54, 62,
                   65, 73, 81, 89, 97, 105, 113, 121]
        ref_col = slice(None)

        asic = None

        # get result calculated by set_pos_indices
        res = self.gather_obj.set_pos_indices(run_idx, asic)[0]

        # compare result to reference
        self.assertTrue(np.all(res[0] == ref_row))
        self.assertTrue(np.all(res[1] == ref_col))

    def test_pos_indices2(self):

        run_idx = 2
        # reference values
        ref_row = [5, 13, 21, 29, 37, 45, 53, 61,
                   66, 74, 82, 90, 98, 106, 114, 122]
        ref_col = slice(None)

        asic = None

        # get result calculated by set_pos_indices
        res = self.gather_obj.set_pos_indices(run_idx, asic)[0]

        # compare result to reference
        self.assertTrue(np.all(res[0] == ref_row))
        self.assertTrue(np.all(res[1] == ref_col))

#    def test_class(self):
#        self.gather_obj.run()


if __name__ == "__main__":
    # Run only the tests in the specified classes

    test_classes_to_run = [
        GatherBaseXfelTests,
        GatherPcdrsXfelTests
    ]

    loader = unittest.TestLoader()

    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)

    runner = unittest.TextTestRunner()
    results = runner.run(big_suite)
