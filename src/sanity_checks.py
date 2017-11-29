import numpy as np
import h5py
import sys
import os
import utils
import unittest
import glob


beamtime = "201730/p900009"
run = 709
file_raw_prefix_temp = None
file_raw_temp = None
data = None
data_sep = None

def read_in_data(file_raw_temp, channel, dict_key, read_in_path, seq_start,
                 seq_stop, data, data_sep, convert=False):

    data[channel][dict_key] = None
    data_sep[channel][dict_key] = []

    for seq in range(seq_start, seq_stop):
        fname = file_raw_temp.format(channel, seq)
        #print(fname)

        f = h5py.File(fname, "r")
        read_in_data = f[read_in_path][()].astype("int")
        f.close()

        if convert:
            read_in_data = utils.convert_to_agipd_format(channel, read_in_data)

        d = data[channel]
        if d[dict_key] is None:
            d[dict_key] = read_in_data
        else:
            d[dict_key] = np.append(d[dict_key], read_in_data, axis=0)

        data_sep[channel][dict_key].append(read_in_data)


# for the whole module
def setUpModule():
    global beamtime
    global run
    global file_raw_prefix_temp
    global file_raw_temp
    global data
    global data_sep

    file_raw_prefix_temp = ("/gpfs/exfel/exp/SPB/{bt}/raw/r{r:04d}/RAW-R{r:04d}"
                                     .format(bt=beamtime, r=run) +
                                     "-AGIPD{:02d}-S")
    file_raw_temp = file_raw_prefix_temp + "{:05d}.h5"

    path_temp = {
        "header_train_id": "/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/header/trainId",
        "image_train_id": "/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/image/trainId",
        "cell_id": "/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/image/cellId",
        "pulse_id": "/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/image/pulseId",
        "data": "/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/image/data"
    }

    seq_start = 0
    seq_stop = 3

    data = [{} for i in range(16)]
    data_sep = [{} for i in range(16)]

    for dict_key in ["header_train_id"]:
        for channel in np.arange(16):
            read_in_path = path_temp[dict_key].format(channel)
            read_in_data(file_raw_temp,
                         channel,
                         dict_key,
                         read_in_path,
                         seq_start,
                         seq_stop,
                         data,
                         data_sep)


# for the whole module
def tearDownModule():
    pass


class SanityChecks(unittest.TestCase):
    # for the whole class
    @classmethod
    def setUpClass(cls):
        global file_raw_prefix_temp
        global file_raw_temp
        global data
        global data_sep

        cls._file_raw_prefix_temp = file_raw_prefix_temp
        cls._file_raw_temp = file_raw_temp
        cls._data = data
        cls._data_sep = data_sep

    # per test
    def setUp(self):
        pass

    def test_n_seqs_equal(self):
        print("Test that all modules have the same number of sequences")

        res = []
        for channel in range(16):
            fname_prefix = self._file_raw_prefix_temp.format(channel) + "*.h5"
            seq_files = sorted(glob.glob(fname_prefix))

            res.append(len(seq_files))

        self.assertEqual(len(np.unique(res)), 1)

    def test_n_train_equal(self):
        print("Check if number of trains are equal for all module (per seq)")

        seq_start = 0
        seq_stop = 3

        res = []
        for channel in np.arange(16):
            d_sep = self._data_sep[channel]["header_train_id"]
            res.append([len(d_sep[seq]) for seq in range(len(d_sep))])

        res = np.array(res)
        for seq in range(len(d_sep)):
            msg = ("Not all modules have the same number of trains for seq {}\n"
                   "(n_trains={})".format(seq, res[:, seq]))
            unique = np.unique(res[:, seq])

            self.assertEqual(len(unique), 1, msg)

    # per test
    def tearDown(self):
        pass

    # for the whole class
    @classmethod
    def tearDownClass(cls):
        pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
