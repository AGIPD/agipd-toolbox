from __future__ import print_function
import h5py
import numpy as np
import os

module_file_path = "/gpfs/cfel/fsds/labs/processed/calibration/processed/M314/temperature_40C/drscs/itestc150/"
axis_file_path = "/gpfs/cfel/fsds/labs/processed/calibration/processed/M314/temperature_40C/drscs/itestc150/"

g = h5py.File(os.path.join(module_file_path, "M314_drscs_itestc150_chunked.h5"), "r")

# index: col_start, col_stop, row_start, row_stop
asic_files = [
    {
        # ASIC 1
        #"file_name": "M314_drscs_itestc150_asic_1_chunked_.h5",
        "file_name": "M314_drscs_itestc150_asic1.h5",
        "index": [0*64, 1*64, 0*64, 1*64]
    },
    {
        # ASIC 2
        "file_name": "M314_drscs_itestc150_asic2.h5",
        "index": [0*64, 1*64, 1*64, 2*64]
    },
    {
        # ASIC 3
        "file_name": "M314_drscs_itestc150_asic3.h5",
        "index": [0*64, 1*64, 2*64, 3*64]
    },
    {
        # ASIC 4
        "file_name": "M314_drscs_itestc150_asic4.h5",
        "index": [0*64, 1*64, 3*64, 4*64]
    },
    {
        # ASIC 5
        "file_name": "M314_drscs_itestc150_asic5.h5",
        "index": [0*64, 1*64, 4*64, 5*64]
    },
    {
        # ASIC 6
        "file_name": "M314_drscs_itestc150_asic6.h5",
        "index": [0*64, 1*64, 5*64, 6*64]
    },
    {
        # ASIC 7
        "file_name": "M314_drscs_itestc150_asic7.h5",
        "index": [0*64, 1*64, 6*64, 7*64]
    },
    {
        # ASIC 8
        "file_name": "M314_drscs_itestc150_asic8.h5",
        "index": [0*64, 1*64, 7*64, 8*64]
    },
]

for asic in asic_files:

    f = h5py.File(os.path.join(axis_file_path, asic["file_name"]), "r")

    dataf = f["/analog"][:, :, :, :]
    datag = g["/analog"][:, :, asic["index"][0]:asic["index"][1], asic["index"][2]:asic["index"][3]]

    if np.all(dataf == datag):
        print(asic["file_name"], "correct")
    else:
        print(asic["file_name"], "not correct")

    f.close()
