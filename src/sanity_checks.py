import numpy as np
import h5py
import sys
import os
import utils
import unittest
import glob
import argparse


beamtime = None
run = None
file_raw_prefix_temp = None
file_raw_temp = None
data = None

def get_arguments():
    description = {
        "test_n_seqs_equal": "Tests that all modules have the same number of \n"
                             "sequences",
        "test_n_train_equal": "Checks if number of trains is equal for all module\n"
                              "(per seq)",
        "test_dims_header": ("Checks if the first dimension is equal for all\n"
                             "datasets contained in 'header' (per module and per seq)"),
        "test_dims_image": ("Checks if the first dimension is equal for all datasets\n"
                            "contained in 'image' (per module and per seq)"),
        "test_dims_trailer": ("Checks if the first dimension is equal for all\n"
                              "datasets contained in 'trailer' (per module and per seq)"),
        "test_train_id_shift": ("Checks if the first train id value is equal for all\n"
                                "modules"),
        "test_train_id_shift": ("Checks if the train ids taken from detector, header\n"
                                "and trailer are equal (per module)"),
    }

    # determine how long the space for the keys should be
    max_key_len = 0
    for key in description:
        l = len(key)
        if l > max_key_len:
            max_key_len = l

    # maximum line length
    max_len = 80

    epilog = "Test descriptions:\n"
    epilog += "-" * max_len + "\n"

    for key in description:
        test_desc = "| {} {}".format((key + ":").ljust(max_key_len + 1),
                                     description[key])
        desc_split = test_desc.split("\n")

        test_desc = desc_split[0].ljust(max_len - 1) + "|\n"
        for s in desc_split[1:]:
            # pad the description to key length
            line = (" " * (max_key_len + 3) + s)
            # fill up line to maximum length
            line = line.ljust(max_len - 2)
            test_desc += "|" + line + "|\n"

        epilog += test_desc
        epilog += "|" + "-" * (max_len - 2) + "|\n"

    # remove the '|' from the last line again
    li = epilog.rsplit("|", 2)
    epilog = "-".join(li)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog)

    parser.add_argument("--instrument_cycle",
                        type=str,
                        required=True,
                        help="The instrument_cycle the beamtime was taken, e.g 201701")
    parser.add_argument("--beamtime",
                        type=str,
                        required=True,
                        help="The beamtime to check")
    parser.add_argument("--run",
                        type=int,
                        required=True,
                        help="The run to check")

    args = parser.parse_args()
    return args


def read_in_data(file_raw_temp, channel, dict_key, read_in_path, seq_start,
                 seq_stop, data, convert=False):

    data[channel][dict_key] = []

    for seq in range(seq_start, seq_stop):
        fname = file_raw_temp.format(channel, seq)
        #print(fname)

        f = h5py.File(fname, "r")
        read_in_data = f[read_in_path][()].astype("int")
        f.close()

        if convert:
            read_in_data = utils.convert_to_agipd_format(channel, read_in_data)

        data[channel][dict_key].append(read_in_data)


# for the whole module
def setUpModule():
    global beamtime
    global run
    global file_raw_prefix_temp
    global file_raw_temp
    global data

    file_raw_prefix_temp = ("/gpfs/exfel/exp/SPB/{bt}/raw/r{r:04d}/RAW-R{r:04d}"
                                     .format(bt=beamtime, r=run) +
                                     "-AGIPD{:02d}-S")
    file_raw_temp = file_raw_prefix_temp + "{:05d}.h5"

    path_temp = {
        "detector_train_id": "/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/detector/trainId",
        "header_train_id": "/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/header/trainId",
        "image_train_id": "/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/image/trainId",
        "trailer_train_id": "/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/trailer/trainId",
        "cell_id": "/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/image/cellId",
        "pulse_id": "/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/image/pulseId",
        "data": "/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/image/data"
    }

    seq_start = 0
    seq_stop = 3

    data = [{} for i in range(16)]

    for dict_key in ["detector_train_id", "header_train_id", "trailer_train_id"]:
        for channel in np.arange(16):
            read_in_path = path_temp[dict_key].format(channel)
            read_in_data(file_raw_temp,
                         channel,
                         dict_key,
                         read_in_path,
                         seq_start,
                         seq_stop,
                         data)


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

        cls._file_raw_prefix_temp = file_raw_prefix_temp
        cls._file_raw_temp = file_raw_temp
        cls._data = data

        cls._path = {
            "header": "/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/header",
            "image": "/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/image",
            "trailer": "/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/trailer"
        }

        cls._seq_start = 0
        cls._seq_stop = 3
        cls._n_sequences = len(data[0]["header_train_id"])

    # per test
    def setUp(self):
        pass

    def test_n_seqs_equal(self):
        #print("\n\tTests that all modules have the same number of sequences")

        res = []
        for channel in range(16):
            fname_prefix = self._file_raw_prefix_temp.format(channel) + "*.h5"
            seq_files = sorted(glob.glob(fname_prefix))

            res.append(len(seq_files))

        self.assertEqual(len(np.unique(res)), 1)

        self._n_sequences = res[0]

    def test_n_train_equal(self):
        #print("\n\tChecks if number of trains is equal for all module (per seq)")

        seq_start = 0
        seq_stop = 3

        res = []
        for channel in np.arange(16):
            d = self._data[channel]["header_train_id"]
            res.append([len(d[seq]) for seq in range(self._n_sequences)])

        res = np.array(res)
        for seq in range(len(d)):
            msg = ("Not all modules have the same number of trains for seq {}\n"
                   "(n_trains={})".format(seq, res[:, seq]))
            unique = np.unique(res[:, seq])

            self.assertEqual(len(unique), 1, msg)

    def test_dims_header(self):
        #print("\n\tChecks if the first dimension is equal for all datasets \n"
        #      "\tcontained in '{}' (per module and per seq)".format("header"))

        for channel in range(16):
            for seq in range(self._seq_start, self._seq_stop):
                fname = self._file_raw_temp.format(channel, seq)
                #print(fname)

                group_name = self._path["header"].format(channel)

                f = h5py.File(fname, "r")
                keys = list(f[group_name].keys())

                res = []
                for key in keys:
                    path = "{}/{}".format(group_name, key)
                    res.append(f[path].shape[0])

                f.close()

                dims = dict(zip(keys, res))
                msg = ("Channel {}, sep {}: dimensions in header are not the same\n"
                       "(dimensions are {})".format(channel, seq, dims))
                unique = np.unique(res)
                self.assertEqual(len(unique), 1, msg)

    def test_dims_image(self):
        #print("\n\tChecks if the first dimension is equal for all datasets \n"
        #      "\tcontained in '{}' (per module and per seq)".format("image"))

        for channel in range(16):
            for seq in range(self._seq_start, self._seq_stop):
                fname = self._file_raw_temp.format(channel, seq)
                #print(fname)

                group_name = self._path["image"].format(channel)

                f = h5py.File(fname, "r")
                keys = list(f[group_name].keys())

                res = []
                for key in keys:
                    path = "{}/{}".format(group_name, key)
                    res.append(f[path].shape[0])

                f.close()

                dims = dict(zip(keys, res))
                msg = ("Channel {}, sep {}: dimensions in header are not the same\n"
                       "(dimensions are {})".format(channel, seq, dims))
                unique = np.unique(res)
                self.assertEqual(len(unique), 1, msg)

    def test_dims_trailer(self):
        #print("\n\tChecks if the first dimension is equal for all datasets \n"
        #      "\tcontained in '{}' (per module and per seq)".format("trailer"))

        for channel in range(16):
            for seq in range(self._seq_start, self._seq_stop):
                fname = self._file_raw_temp.format(channel, seq)
                #print(fname)

                group_name = self._path["trailer"].format(channel)

                f = h5py.File(fname, "r")
                keys = list(f[group_name].keys())

                res = []
                for key in keys:
                    path = "{}/{}".format(group_name, key)
                    res.append(f[path].shape[0])

                f.close()

                dims = dict(zip(keys, res))
                msg = ("Channel {}, sep {}: dimensions in header are not the same\n"
                       "(dimensions are {})".format(channel, seq, dims))
                unique = np.unique(res)
                self.assertEqual(len(unique), 1, msg)

    def test_train_id_shift(self):
        #print("\n\tChecks if the first train id value is equal for all modules")

        usable_start = 2

        first_train_ids = [d["header_train_id"][0][usable_start + 0] for d in data]
        train_id_start = np.min(first_train_ids)

        diff_first_train = np.where(first_train_ids != train_id_start)[0]

        msg = ("Channels with shifted first train id: {}\n".format(diff_first_train))
        for i, train_id in enumerate(first_train_ids):
            msg += "channel {:02}: {}\n".format(i, train_id)

        unique = np.unique(first_train_ids)
        self.assertEqual(len(unique), 1, msg)

    def test_train_id_equal(self):
        #print("\n\tChecks if the train ids taken from detector, header and trailer are equal (per module)")

        seq_start = 0
        seq_stop = 3

        res = []
        for channel in np.arange(16):
            for seq in range(self._n_sequences):
                d_detector = self._data[channel]["detector_train_id"][seq]
                d_header = self._data[channel]["header_train_id"][seq]
                d_trailer = self._data[channel]["trailer_train_id"][seq]

                detector_vs_header = (d_detector == d_header).all()
                header_vs_trailer = (d_header == d_trailer).all()
                res = np.logical_and( detector_vs_header, header_vs_trailer)

                msg = ("train ids from detector, header and trailer do not "
                       "match for channel {}, seq {}".format(channel, seq))
                self.assertTrue(res, msg)

    # per test
    def tearDown(self):
        pass

    # for the whole class
    @classmethod
    def tearDownClass(cls):
        pass


if __name__ == "__main__":

    #instrument_cycle = "201730"
    #bt = "p900009"
    #r = 709
    args = get_arguments()

    beamtime = "{}/{}".format(args.instrument_cycle, args.beamtime)
    run = args.run

    itersuite = unittest.TestLoader().loadTestsFromTestCase(SanityChecks)
    runner = unittest.TextTestRunner(verbosity=2).run(itersuite)

    #unittest.main(verbosity=2)
