import h5py
import numpy as np
import time
import os

STORAGE_PATH = "/gpfs/cfel/fsds/labs/calibration/scratch/"


def create_file(filename, shape, chunksize):
    global STORAGE_PATH

    file_path = os.path.join(STORAGE_PATH, filename)

    o_file = h5py.File(file_path, "w", libver='latest')

    dset_analog = o_file.create_dataset("analog",
                                        shape=shape,
                                        chunks=chunksize,
                                        compression=None, dtype='int16')
    dset_digital = o_file.create_dataset("digital",
                                         shape=shape,
                                         chunks=chunksize,
                                         compression=None, dtype='int16')

    analog = np.random.rand(*shape)
    digital = np.random.rand(*shape)

    dset_analog[...] = analog
    dset_digital[...] = digital

    o_file.flush()
    o_file.close()

if __name__ == "__main__":

    # Memory cell, charge, asic (l x w)
    shape=(352, 1200, 64, 64)
    file_list = [
        [ "output_file_b_352-1200-64-64.h5", shape, (352, 1200, 64, 64)],
        [ "output_file_b_1-1-64-64.h5", shape, (1, 1, 64, 64)],
        [ "output_file_b_1-1200-64-64.h5", shape, (1, 1200, 64, 64)],
        [ "output_file_b_1-1200-1-1.h5", shape, (1, 1200, 1, 1)],
        [ "output_file_b_352-1200-1-1.h5", shape, (352, 1200, 1, 1)],
    ]

    # Asic (l x w), charge, memory cell
    shape=(64, 64, 1200, 352)
    file_list += [
        [ "output_file_f_64-64-1200-352.h5", shape, (64, 64, 1200, 352)],
        [ "output_file_f_64-64-1-1.h5", shape, (64, 64, 1, 1)],
        [ "output_file_f_64-64-1200-1.h5", shape, (64, 64, 1200, 1)],
        [ "output_file_f_1-1-1200-1.h5", shape, (1, 1, 1200, 1)],
        [ "output_file_f_1-1-1200-352.h5", shape, (1, 1, 1200, 352)],
    ]

    # Charge, memory cell, asic (l x w)
    shape=(1200, 352, 64, 64)
    file_list += [
        [ "output_file_b_1200-352-64-64.h5", shape, (1200, 352, 64, 64)],
        [ "output_file_b_1-1-64-64.h5", shape, (1, 1, 64, 64)],
        [ "output_file_b_1200-1-64-64.h5", shape, (1200, 1, 64, 64)],
        [ "output_file_b_1200-1-1-1.h5", shape, (1200, 1, 1, 1)],
        [ "output_file_b_1200-352-1-1.h5", shape, (1200, 352, 1, 1)],
    ]

    # Asic (l x w), memory cell, charge
    shape=(64, 64, 352, 1200)
    file_list += [
        [ "output_file_f_64-64-352-1200.h5", shape, (64, 64, 352, 1200)],
        [ "output_file_f_64-64-1-1.h5", shape, (64, 64, 1, 1)],
        [ "output_file_f_64-64-1-1200.h5", shape, (64, 64, 1, 1200)],
        [ "output_file_f_1-1-1-1200.h5", shape, (1, 1, 1, 1200)],
        [ "output_file_f_1-1-352-1200.h5", shape, (1, 1, 352, 1200)],
    ]

    for f, s, c in file_list:
        t = time.time()
        create_file(f, s, c)
        t2 = time.time()
        print("shape {}, chunk {} - time needed: {}".format(s, c, t2-t))
