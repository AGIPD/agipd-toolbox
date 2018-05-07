#!/usr/bin/python3

import argparse
import os
import sys
import h5py
import numpy as np

class ReadData_three_times_same(object):
    def __init__(self, as_int=True):
        self.as_int = as_int

        self.fname = "/gpfs/exfel/exp/SPB/201830/p900019/raw/r0321/RAW-R0321-AGIPD00-S00000.h5"
        self.trainid_path = "INDEX/trainId"

#        with h5py.File(self.fname, "r") as f:
#            a = f[entry_to_test]

        f = h5py.File(self.fname, "r")
        a = f[self.trainid_path]
        f.close()

        self._open_in_function()

        with h5py.File(self.fname, "r") as h:
            print(self.fname, self.trainid_path)
            my_trainid = h[self.trainid_path][()]
            print("trainid", my_trainid[:150])

    def _open_in_function(self):
        with h5py.File(self.fname, "r") as g:
            in_data = g[self.trainid_path][()]
            #in_data = g[self.count_path][()]

            if self.as_int:
                in_data.astype(np.uint64)


class ReadData(object):
    def __init__(self, as_int=True):
        self.as_int = as_int

        self.fname = "/gpfs/exfel/exp/SPB/201830/p900019/raw/r0321/RAW-R0321-AGIPD00-S00000.h5"

        entry_to_test = "INDEX/SPB_DET_AGIPD1M-1/DET/0CH0:xtdf/image/count"
        self.trainid_path = "INDEX/trainId"
        self.count_path = "INSTRUMENT/SPB_DET_AGIPD1M-1/DET/0CH0:xtdf/header/pulseCount"

#        with h5py.File(self.fname, "r") as f:
#            a = f[entry_to_test]

        f = h5py.File(self.fname, "r")
        a = f[entry_to_test]
        f.close()

        self._open_in_function()

        with h5py.File(self.fname, "r") as h:
            print(self.fname, self.trainid_path)
            my_trainid = h[self.trainid_path][()]
            print("trainid", my_trainid[:150])

    def _open_in_function(self):
        with h5py.File(self.fname, "r") as g:
            in_data = g[self.count_path][()]

            if self.as_int:
                in_data.astype(np.uint64)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--as_int",
                        action="store_true",
                        default=False)
    args = parser.parse_args()


    obj = ReadData(args.as_int)
