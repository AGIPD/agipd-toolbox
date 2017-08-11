from __future__ import print_function

import h5py
import numpy as np
import time
import os

STORAGE_PATH = "/gpfs/cfel/fsds/labs/calibration/scratch/"


class ReadFile():
    def __init__(self, file_list):
        global STORAGE_PATH
        # Open files
        self.open_files = []
        for f in file_list:
            file_path = os.path.join(STORAGE_PATH, f)

            t = time.time()
            file_handler = h5py.File(file_path, "r", libver='latest')
            diff = time.time() - t

            self.open_files.append({
                "filename": f,
                "fhandler": file_handler,
                "chunksize": file_handler["analog"].chunks,
                "shape": file_handler["analog"].shape
            })

        self.shape = self.open_files[0]["shape"]

    def read_files_f(self):
        print("shape", self.shape, "access [:,:,:,1]")
        for f in self.open_files:
            t = time.time()
            np.histogram(f["fhandler"]["analog"][1, :, :, :])
            diff = time.time() - t

            print("chunk {: <{width}} - Needed time: {}"
                  .format(f["chunksize"], diff, width = len(str(self.shape))))

        print("shape", self.shape, "access [0,0,:,1]")
        for f in self.open_files:
            t = time.time()
            np.histogram(f["fhandler"]["analog"][0, 0, :,1])
            diff = time.time() - t

            print("chunk {: <{width}} - Needed time: {}"
                  .format(f["chunksize"], diff, width = len(str(self.shape))))

        print("shape", self.shape, "access [0,0,:,:]")
        for f in self.open_files:
            t = time.time()
            np.histogram(f["fhandler"]["analog"][0, 0, :, :])
            diff = time.time() - t

            print("chunk {: <{width}} - Needed time: {}"
                  .format(f["chunksize"], diff, width = len(str(self.shape))))


    def read_files_b(self):
        print("shape", self.shape, "access [1,:,:,:]")
        for f in self.open_files:
            t = time.time()
            np.histogram(f["fhandler"]["analog"][1, :, :, :])
            diff = time.time() - t

            print("chunk {: <{width}} - Needed time: {}"
                  .format(f["chunksize"], diff, width = len(str(self.shape))))

        print("shape", self.shape, "access [1,:,0,0]")
        for f in self.open_files:
            t = time.time()
            np.histogram(f["fhandler"]["analog"][1, :, 0, 0])
            diff = time.time() - t

            print("chunk {: <{width}} - Needed time: {}"
                  .format(f["chunksize"], diff, width = len(str(self.shape))))

        print("shape", self.shape, "access [:,:,0,0]")
        for f in self.open_files:
            t = time.time()
            np.histogram(f["fhandler"]["analog"][:, :, 0, 0])
            diff = time.time() - t

            print("chunk {: <{width}} - Needed time: {}"
                  .format(f["chunksize"], diff, width = len(str(self.shape))))

    def close_files(self):
        # Close file
        for f in self.open_files:
            f["fhandler"].close()
        self.open_files = []

    def __exit__(self):
        self.close_files()

    def __delete__(self):
        self.close_files()


if __name__ == "__main__":

    # Memory cell, charge, asic (l x w)
    f_list = [
        "output_file_b_352-1200-64-64.h5",
        "output_file_b_1-1-64-64.h5",
        "output_file_b_1-1200-64-64.h5",
        "output_file_b_1-1200-1-1.h5",
        "output_file_b_352-1200-1-1.h5",
    ]

    obj = ReadFile(f_list)
    obj.read_files_b()
    obj.close_files()

    print()

    # Asic (l x w), charge, memory cell
    f_list = [
        "output_file_f_64-64-1200-352.h5",
        "output_file_f_64-64-1-1.h5",
        "output_file_f_64-64-1200-1.h5",
        "output_file_f_1-1-1200-1.h5",
        "output_file_f_1-1-1200-352.h5",
    ]

    # Charge, memory cell, asic (l x w)
    obj = ReadFile(f_list)
    obj.read_files_f()
    obj.close_files()

    print()

    f_list = [
        "output_file_b_1200-352-64-64.h5",
        "output_file_b_1-1-64-64.h5",
        "output_file_b_1200-1-64-64.h5",
        "output_file_b_1200-1-1-1.h5",
        "output_file_b_1200-352-1-1.h5",
    ]

    obj = ReadFile(f_list)
    obj.read_files_b()
    obj.close_files()

    print()

    # Asic (l x w), memory cell, charge
    f_list = [
        "output_file_f_64-64-352-1200.h5",
        "output_file_f_64-64-1-1.h5",
        "output_file_f_64-64-1-1200.h5",
        "output_file_f_1-1-1-1200.h5",
        "output_file_f_1-1-352-1200.h5",

    ]

    obj = ReadFile(f_list)
    obj.read_files_f()
    obj.close_files()
