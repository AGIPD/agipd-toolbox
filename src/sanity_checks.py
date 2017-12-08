import numpy as np
import h5py
import sys
import os
#import utils
import unittest
import glob
import argparse


beamtime = None
run = None
file_raw_prefix_temp = None
file_raw_temp = None
data = None
path_temp = None


description = {
    # -------------------------------------------------------------------------------- #
    "test_n_seqs_equal":   ("Tests that all modules have the same number of \n"
                            "sequences"),
    "test_n_train_equal":  ("Checks if number of trains is equal for all module\n"
                            "(per seq)"),
    "test_dims_header":    ("Checks if the first dimension is equal for all\n"
                            "datasets contained in 'header' (per module and per seq)"),
    "test_dims_image":     ("Checks if the first dimension is equal for all datasets\n"
                            "contained in 'image' (per module and per seq)"),
    "test_dims_trailer":   ("Checks if the first dimension is equal for all\n"
                            "datasets contained in 'trailer' (per module and per seq)"),
    "test_train_id_shift": ("Checks if the first train id value is equal for all\n"
                            "modules"),
    "test_train_id_equal": ("Checks if the train ids taken from detector, header\n"
                            "and trailer are equal (per module)"),
    "test_data_vs_pulsec": ("Checks if the sum of the pulseCount entries is\n"
                            "corresponding to the data"),
    "test_train_id_diff":  ("Checks if the trainId is monotonically increasing"),
    "test_train_id_tzero": ("Checks number of placeholder in trainId and if they are\n"
                            "always at the end"),
    "test_train_id_zeros": ("Checks if trainId containes zeros which are not at the\n"
                            "end"),
    "test_train_loss":     ("Checks for missing entries in trainId"),
    "test_data_tzeros":    ("Check if additional data entries are trailing zeros"),
    "test_data_vs_tr_id":  ("Checks if the dimension of the image trainId is\n"
                            "corresponding to the data)"),
}


def create_epilog():
    global description

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

    return epilog


def get_arguments():

    epilog = create_epilog()

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

#        if convert:
#            read_in_data = utils.convert_to_agipd_format(channel, read_in_data)

        data[channel][dict_key].append(read_in_data)


def get_trailing_zeros(data):
    res = []

    for seq in range(len(data)):

        eq_zero_idx = np.where(data[seq] == 0)[0]
        #print(eq_zero_idx)
        diff_zeros = np.diff(eq_zero_idx)
        #print(diff_zeros)
        non_consecutive_idx = np.where(diff_zeros != 1)[0]
        #print(non_consecutive_idx)

        if eq_zero_idx.size == 0:
            res.append(0)
            continue
        last_eq_zero_idx = len(eq_zero_idx)
        if last_eq_zero_idx not in non_consecutive_idx:
            # calculate how many zeros there are
            if non_consecutive_idx.size == 0:
                trailing_zeros_start = eq_zero_idx[0]
            else:
                trailing_zeros_start = eq_zero_idx[non_consecutive_idx[-1] + 1]
            trailing_zeros_stop = eq_zero_idx[last_eq_zero_idx - 1] + 1
            #print(trailing_zeros_start, trailing_zeros_stop)
            n_zeros = trailing_zeros_stop - trailing_zeros_start
            #print("number of zeros: ", n_zeros)

            res.append(n_zeros)

    return res


# for the whole module
def setUpModule():
    global beamtime
    global run
    global file_raw_prefix_temp
    global file_raw_temp
    global data
    global path_temp

    file_raw_prefix_temp = ("/gpfs/exfel/exp/SPB/{bt}/raw/r{r:04d}/RAW-R{r:04d}"
                                     .format(bt=beamtime, r=run) +
                                     "-AGIPD{:02d}-S")
    file_raw_temp = file_raw_prefix_temp + "{:05d}.h5"

    path_temp = {
        "header": "/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/header",
        "image": "/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/image",
        "trailer": "/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/trailer",
        "detector_train_id": "/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/detector/trainId",
        "header_train_id": "/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/header/trainId",
        "image_train_id": "/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/image/trainId",
        "trailer_train_id": "/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/trailer/trainId",
        "pulse_count": "/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/header/pulseCount",
        "cell_id": "/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/image/cellId",
        "pulse_id": "/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/image/pulseId",
        "data": "/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/image/data"
    }

    seq_start = 0
    seq_stop = 3

    data = [{} for i in range(16)]

    keys_to_read_in = ["detector_train_id",
                       "header_train_id",
                       "trailer_train_id",
                       "pulse_count"]

    for dict_key in keys_to_read_in:
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
        global path_temp

        cls._file_raw_prefix_temp = file_raw_prefix_temp
        cls._file_raw_temp = file_raw_temp
        cls._data = data
        cls._path = path_temp

        cls._seq_start = 0
        cls._seq_stop = 3
        cls._n_sequences = len(data[0]["header_train_id"])
        cls._n_channels = 16

        cls._usable_start = 2


    # per test
    def setUp(self):
        pass

    def test_n_seqs_equal(self):
        """
        Tests that all modules have the same number of sequences
        """

        res = []
        for channel in range(self._n_channels):
            fname_prefix = self._file_raw_prefix_temp.format(channel) + "*.h5"
            seq_files = sorted(glob.glob(fname_prefix))

            res.append(len(seq_files))

        self.assertEqual(len(np.unique(res)), 1)

        self._n_sequences = res[0]

    def test_n_train_equal(self):
        """
        Checks if number of trains is equal for all module (per seq)
        """

        res = []
        for channel in np.arange(self._n_channels):
            d = self._data[channel]["header_train_id"]
            res.append([len(d[seq]) for seq in range(self._n_sequences)])

        assert_failed = False
        msg = ""
        res = np.array(res)
        for seq in range(len(d)):
#            msg = ("\nNot all modules have the same number of trains for seq {}\n"
#                   "(n_trains={})".format(seq, res[:, seq]))
            unique = np.unique(res[:, seq])

            try:
                self.assertEqual(len(unique), 1)
            except AssertionError:
                # tried to use subtest for this but it broke the new lines
                # in the check summary
                assert_failed = True
                msg += ("\nNot all modules have the same number of trains for seq {}\n"
                       "(n_trains={})".format(seq, res[:, seq]))

        # for clarity only print one error message for all sequences
        if assert_failed:
            self.fail(msg)

    def test_dims_header(self):
        """
        Checks if the first dimension is equal for all datasets contained
        in 'header' (per module and per seq)
        """

        assert_failed = False
        msg = "Dimensions in header are not the same for:"
        for channel in range(self._n_channels):
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

                unique = np.unique(res)
                try:
                    self.assertEqual(len(unique), 1)
                except AssertionError:
                    assert_failed = True
                    dims = dict(zip(keys, res))
                    msg += ("\nChannel {:02}, sep {} (dimensions are {})"
                            .format(channel, seq, dims))

        # for clarity only print one error message for all sequences
        if assert_failed:
            self.fail(msg)

    def test_dims_image(self):
        """
        Checks if the first dimension is equal for all datasets contained
        in 'image' (per module and per seq)"
        """

        assert_failed = False
        msg = "Dimensions in image are not the same for: "
        for channel in range(self._n_channels):
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

                unique = np.unique(res)
                try:
                    self.assertEqual(len(unique), 1)
                except AssertionError:
                    assert_failed = True
                    dims = dict(zip(keys, res))
                    msg += ("\nChannel {:02}, sep {} (dimensions are {})"
                            .format(channel, seq, dims))

        # for clarity only print one error message for all sequences
        if assert_failed:
            self.fail(msg)

    def test_dims_trailer(self):
        """
        Checks if the first dimension is equal for all datasets contained in
        'trailer' (per module and per seq)"
        """

        assert_failed = False
        msg = "Dimensions in trailer are not the same for:"
        for channel in range(self._n_channels):
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

                unique = np.unique(res)
                try:
                    self.assertEqual(len(unique), 1)
                except AssertionError:
                    assert_failed = True
                    dims = dict(zip(keys, res))
                    msg = ("\nChannel {:02}, sep {} (dimensions are {})"
                           .format(channel, seq, dims))

        # for clarity only print one error message for all sequences
        if assert_failed:
            self.fail(msg)

    def test_train_id_shift(self):
        """
        Checks if the first train id value is equal for all modules
        """

        first_train_ids = [d["header_train_id"][0][self._usable_start + 0] for d in data]
        train_id_start = np.min(first_train_ids)

        diff_first_train = np.where(first_train_ids != train_id_start)[0]

        msg = ("\nChannels with shifted first train id: {}\n".format(diff_first_train))
        for i, train_id in enumerate(first_train_ids):
            msg += "channel {:02}: {}\n".format(i, train_id)

        unique = np.unique(first_train_ids)
        self.assertEqual(len(unique), 1, msg)

    def test_train_id_equal(self):
        """
        Checks if the train ids taken from detector, header and trailer are
        equal (per module)
        """

        assert_failed = False
        msg = ""
        res = []
        for channel in np.arange(self._n_channels):
            for seq in range(self._n_sequences):
                d_detector = self._data[channel]["detector_train_id"][seq]
                d_header = self._data[channel]["header_train_id"][seq]
                d_trailer = self._data[channel]["trailer_train_id"][seq]

                detector_vs_header = (d_detector == d_header).all()
                header_vs_trailer = (d_header == d_trailer).all()
                res = np.logical_and( detector_vs_header, header_vs_trailer)

                try:
                    self.assertTrue(res)
                except AssertionError:
                    assert_failed = True
                    msg = ("\nTrainIds from detector, header and trailer do not "
                            "match for channel {:02}, seq {}".format(channel, seq))

        # for clarity only print one error message for all sequences
        if assert_failed:
            self.fail(msg)

    def test_data_vs_pulsec(self):
        """
        Checks if the sum of the pulseCount entries is corresponding to the data)
        """

        assert_failed = False
        msg = ("\nPulseCount and data shape do not match for the following\n"
               "channels and sequences (pulseCount sum vs data shape):")
        for channel in range(self._n_channels):
            for seq in range(self._n_sequences):
                n_total_pulses = np.sum(self._data[channel]["pulse_count"][seq])

                fname = self._file_raw_temp.format(channel, seq)
                #print(fname)
                group_name = self._path["data"].format(channel)

                f = h5py.File(fname, "r")
                data_shape = f[group_name].shape
                f.close()

                try:
                    self.assertEqual(n_total_pulses, data_shape[0])
                except AssertionError:
                    assert_failed = True
                    msg += ("\nChannel {:02}, sequence {} ({} vs {})"
                            .format(channel, seq, n_total_pulses, data_shape[0]))

        # for clarity only print one error message for all channels and sequences
        if assert_failed:
            self.fail(msg)

    def test_train_id_diff(self):
        """
        Checks if the trainId is monotonically increasing
        """

        assert_failed = False
        msg = ("\nTrainId is not monotonically increasing for the following "
               "channels and sequences")
        for channel in range(self._n_channels):
            for seq in range(self._n_sequences):
                train_id = self._data[channel]["header_train_id"][seq]

                # remove placeholders
                train_id = np.where(train_id != 0)

                diff = np.diff(train_id)

                try:
                    self.assertTrue(np.all(diff > 0))
                except AssertionError:
                    assert_failed = True
                    msg += ("\nChannel {:02}, sequence {}"
                            .format(channel, seq))

        # for clarity only print one error message for all channels and sequences
        if assert_failed:
            self.fail(msg)

    def test_train_id_tzero(self):
        """
        Checks number of placeholder in trainId and if they are always at the end
        """

        res = [[] for i in range(self._n_sequences)]

        for ch, _ in enumerate(data):
            d = data[ch]["header_train_id"]
            seq_n_zeros = get_trailing_zeros(d)

            for seq in range(self._n_sequences):
                res[seq].append(seq_n_zeros[seq])

        assert_failed = False
        msg = ""
        for seq, r in enumerate(res):
            unique = np.unique(r)

            try:
                self.assertEqual(len(unique), 1)
            except AssertionError:
                assert_failed = True
                msg += ("\nSequence {} has not the same number of trailing "
                        "zeros for all channels\n({})".format(seq, r))

        # for clarity only print one error message for all sequences
        if assert_failed:
            self.fail(msg)

    def test_train_id_zeros(self):
        """
        Checks if trainId contains zeros which are not at the end
        """

        assert_failed = False
        msg = ("\nTrainid contains zeros which are not at the end for following"
               " channels and sequences:")
        for ch, _ in enumerate(data):
            d = data[ch]["header_train_id"]

            for seq in range(len(d)):

                eq_zero_idx = np.where(data[seq] == 0)[0]
                #print(eq_zero_idx)
                diff_zeros = np.diff(eq_zero_idx)
                #print(diff_zeros)
                non_consecutive_idx = np.where(diff_zeros != 1)[0]
                #print(non_consecutive_idx)

                try:
                    self.assertEqual(non_consecutive_idx.size, 0)
                except AssertionError:
                    assert_failed = True
                    msg += ("\nChannel {:02}, seq {}".format(ch, seq))

        # for clarity only print one error message for all channels and sequences
        if assert_failed:
            self.fail(msg)

    def test_train_loss(self):
        """
        Checks for missing entries in trainId
        """

        assert_failed = False
        msg = ""
        for ch, _ in enumerate(data):
            for seq in range(self._n_sequences):
                d = data[ch]["header_train_id"][seq]

                if seq == 0:
                    d = d[self._usable_start:]

                idx_orig = np.arange(len(d))

                # check trainId within a sequence
                non_zero_idx = np.where(d != 0)

                d_no_zeros = d[non_zero_idx]
                # mapping the new indices to the original ones
                idx_no_zeros = idx_orig[non_zero_idx]

                diff = np.diff(d_no_zeros)
                train_loss_idx = np.where(diff != 1)[0]

                try:
                    self.assertEqual(train_loss_idx.size, 0, msg)
                except AssertionError:
                    assert_failed = True
                    msg += ("\nChannel {:02}, seq {}: Train loss found at indices:\n"
                            .format(ch, seq))
                    for idx  in train_loss_idx:
                        idx_o = idx_no_zeros[idx]
                        msg += ("\tidx {}: ... {} ..."
                                .format(idx_o,
                                        str(d[idx_o - 1:idx_o + 3])[1:-1]))

                # check transition between two sequences
                if seq != 0:
                    try:
                        self.assertEqual(last_seq + 1, d_no_zeros[0])
                    except AssertionError:
                        assert_failed = True
                        msg = ("\nChannel {:02}: Train loss found between sequences"
                               " {} and {}\n".format(ch, seq - 1 , seq))
                        msg += ("(end of seq {}: {}, start of seq {}: {})"
                                .format(seq - 1, last_seq, seq, d_no_zeros[0]))

                # keep the last trainId for the next iteration
                last_seq = d_no_zeros[-1]

        # for clarity only print one error message for all channels and sequences
        if assert_failed:
            self.fail(msg)

    def test_data_tzeros(self):
        """
        Check if extra data entries are trailing zeros
        """

        assert_failed = False
        msg = "\nData containes extra data which is not zero for:"
        for channel in range(self._n_channels):
            for seq in range(self._n_sequences):
                n_total_pulses = np.sum(self._data[channel]["pulse_count"][seq])

                fname = self._file_raw_temp.format(channel, seq)
                #print(fname)
                group_name = self._path["data"].format(channel)

                f = h5py.File(fname, "r")
                data_shape = f[group_name].shape

                if data_shape[0] > n_total_pulses:
                    extra_data = f[group_name][n_total_pulses:]
                else:
                    extra_data = np.array([])

                f.close()

                try:
                    self.assertTrue(not np.any(extra_data))
                except AssertionError:
                    assert_failed = True
                    msg += ("\nChannel {:02}, seq{} (number of zeros {})"
                           .format(channel, seq, extra_data.shape[0]))

        # for clarity only print one error message for all channels and sequences
        if assert_failed:
            self.fail(msg)

    def test_data_vs_tr_id(self):
        """
        Checks if the dimension of the image trainId is corresponding to the data)
        """

        assert_failed = False
        msg = ""
        msg = ("\nTrainId and data shape do not match for:")
        for channel in range(self._n_channels):
            for seq in range(self._n_sequences):
                fname = self._file_raw_temp.format(channel, seq)
                #print(fname)

                group_name = self._path["image_train_id"].format(channel)
                data_name = self._path["data"].format(channel)

                f = h5py.File(fname, "r")
                train_id_shape = f[group_name].shape
                data_shape = f[data_name].shape
                f.close()

                try:
                    self.assertEqual(train_id_shape[0], data_shape[0])
                except AssertionError:
                    assert_failed = True
                    msg += ("Channel {:02}, sequence {} ({} vs {})\n"
                            .format(channel, seq, train_id_shape[0], data_shape[0]))

        # for clarity only print one error message for all channels and sequences
        if assert_failed:
            self.fail(msg)

    # per test
    def tearDown(self):
        pass

    # for the whole class
    @classmethod
    def tearDownClass(cls):
        pass


def suite():
    suite = unittest.TestSuite()

    # to make sure that this test always is the first one to be executed
    suite.addTest(SanityChecks("test_n_seqs_equal"))

    # get the other tests
    test_list = list(description.keys())
    test_list.remove("test_n_seqs_equal")

    # add them in a reproducable way
    for key in sorted(test_list):
        suite.addTest(SanityChecks(key))

    return suite


if __name__ == "__main__":

    #instrument_cycle = "201730"
    #bt = "p900009"
    #r = 709
    args = get_arguments()

    beamtime = "{}/{}".format(args.instrument_cycle, args.beamtime)
    run = args.run

#    itersuite = unittest.TestLoader().loadTestsFromTestCase(SanityChecks)
    itersuite = suite()
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(itersuite)
