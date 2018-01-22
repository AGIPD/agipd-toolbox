#!/usr/bin/python3

"""
Checks XFEL data for sanity
(dimension checks, data versus metadata checks, ...)

@author: Manuela Kuhn (manuela.kuhn@desy.de, DESY, FS-DS)
"""

import numpy as np
import h5py
# import utils
import unittest
import glob
import argparse


beamtime = None
run = None
detail_level = None
file_raw_prefix_temp = None
file_raw_temp = None
data = None
path_temp = None
run_information = {}


description = {
    # ------------------------------------------------------------------------------- #   # noqa E501
    'test_n_seqs_equal':   ("Tests that all modules have the same number of \n"           # noqa E241
                            "sequences"),
    'test_n_train_equal':  ("Checks if number of trains is equal for all module\n"        # noqa E241
                            "(per seq)"),
    'test_dims_header':    ("Checks if the first dimension is equal for all\n"            # noqa E241
                            "datasets contained in 'header' (per module and per seq)"),   # noqa E501
    'test_dims_image':     ("Checks if the first dimension is equal for all datasets\n"   # noqa E241
                            "contained in 'image' (per module and per seq)"),
    'test_dims_trailer':   ("Checks if the first dimension is equal for all\n"            # noqa E241
                            "datasets contained in 'trailer' (per module and per seq)"),  # noqa E501
    'test_train_id_shift': ("Checks if the first train id value is equal for all\n"       # noqa E241
                            "modules"),
    'test_train_id_equal': ("Checks if the train ids taken from detector, header\n"       # noqa E241
                            "and trailer are equal (per module)"),
    'test_data_vs_pulsec': ("Checks if the sum of the pulseCount entries is\n"            # noqa E241
                            "corresponding to the data"),
    'test_train_id_diff':  ("Checks if the trainId is monotonically increasing"),         # noqa E241
    'test_train_id_tzero': ("Checks number of placeholder in trainId and if they are\n"   # noqa E241
                            "always at the end"),
    'test_train_id_zeros': ("Checks if trainId containes zeros which are not at the\n"    # noqa E241
                            "end"),
    'test_train_loss':     ("Checks for missing entries in trainId"),                     # noqa E241
    'test_data_tzeros':    ("Check if additional data entries are trailing zeros"),       # noqa E241
    'test_data_vs_tr_id':  ("Checks if the dimension of the image trainId is\n"           # noqa E241
                            "corresponding to the data)"),
    'test_dim_first_last': ("Checks if the dimensions of the arrays providing the\n"
                            "information about the start and end of the train are of\n"
                            "the same dimensions"),
    'test_pulse_loss':     ("Checks if all trains have the same number of pulses")
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

    parser.add_argument('--instrument_cycle',
                        type=str,
                        required=True,
                        help="The instrument_cycle the beamtime was taken, "
                             "e.g 201701")
    parser.add_argument('--beamtime',
                        type=str,
                        required=True,
                        help="The beamtime to check")
    parser.add_argument('--run',
                        type=int,
                        required=True,
                        help="The run to check")

    parser.add_argument('--detail_level',
                        type=int,
                        choices=[0, 1, 2],
                        default=1,
                        help=("Defines the detail level of the output if the "
                              "test failes (0: no output, 1: general "
                              "information, 2: more detailed information"))

    parser.add_argument('--show_info',
                        action='store_true',
                        help=("Give additional information about the run "
                              "(like how many pulses where taken, ..."))

    args = parser.parse_args()
    return args


def read_in_data(file_raw_temp,
                 channel,
                 dict_key,
                 read_in_path,
                 seq_start,
                 seq_stop,
                 data,
                 convert=False):

    data[channel][dict_key] = []

    for seq in range(seq_start, seq_stop):
        fname = file_raw_temp.format(channel, seq)

        f = h5py.File(fname, 'r')#, driver='core')
        read_in_data = f[read_in_path][()].astype(np.int)
        f.close()

#        if convert:
#            read_in_data = utils.convert_to_agipd_format(channel,
#                                                         read_in_data)

        data[channel][dict_key].append(read_in_data)


def get_trailing_zeros(data):
    res = []

    for seq in range(len(data)):

        eq_zero_idx = np.where(data[seq] == 0)[0]
        diff_zeros = np.diff(eq_zero_idx)
        non_consecutive_idx = np.where(diff_zeros != 1)[0]

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

            n_zeros = trailing_zeros_stop - trailing_zeros_start

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
    global run_information

    file_raw_prefix_temp = ("/gpfs/exfel/exp/SPB/{bt}/raw/r{r:04d}/"
                            "RAW-R{r:04d}".format(bt=beamtime, r=run) +
                            "-AGIPD{:02d}-S")
    # this is not done in file_ra_prefix_temp because later a regex of
    # file_raw_prefix_temp is needed
    file_raw_temp = file_raw_prefix_temp + "{:05d}.h5"

    path_temp = {
        'image_first': "INDEX/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/image/first",
        'image_last': "INDEX/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/image/last",
#        'status': "INDEX/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/image/status",
        'header': "INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/header",
        'image': "INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/image",
        'trailer': "INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/trailer",
        'detector_train_id': "INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/detector/trainId",  # noqa E501
        'header_train_id': "INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/header/trainId",  # noqa E501
        'image_train_id': "INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/image/trainId",  # noqa E501
        'trailer_train_id': "INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/trailer/trainId",  # noqa E501
        'pulse_count': "INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/header/pulseCount",  # noqa E501
        'cell_id': "INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/image/cellId",
        'pulse_id': "INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/image/pulseId",  # noqa E501
        'data': "INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/image/data"
    }

    n_channels = 16

    fname_regex = file_raw_prefix_temp.format(0) + "*.h5"
    n_sequences = len(glob.glob(fname_regex))

    data = [{} for i in range(n_channels)]

    keys_to_read_in = ['detector_train_id',
                       'header_train_id',
                       'trailer_train_id',
                       'pulse_count',
                       'image_first',
                       'image_last']
#                       'status']

    for dict_key in keys_to_read_in:
        for channel in np.arange(n_channels):
            read_in_path = path_temp[dict_key].format(channel)
            read_in_data(file_raw_temp=file_raw_temp,
                         channel=channel,
                         dict_key=dict_key,
                         read_in_path=read_in_path,
                         seq_start=0,
                         seq_stop=n_sequences,
                         data=data)

    run_information = {
        'n_channels': n_channels,
        'usable_start': 2,
        'n_sequences': n_sequences,
        'n_trains': None,
        'n_pulses': None,
    }

# for the whole module
def tearDownModule():
    pass


class SanityChecks(unittest.TestCase):
    # variables shared betreen all tests

    # for the whole class
    @classmethod
    def setUpClass(cls):
        global file_raw_prefix_temp
        global file_raw_temp
        global data
        global path_temp
        global run_information
        global detail_level

        cls._file_raw_prefix_temp = file_raw_prefix_temp
        cls._file_raw_temp = file_raw_temp
        cls._data = data
        cls._path = path_temp
        cls._run_information = run_information

        cls._n_channels = run_information['n_channels']
        cls._n_sequences = run_information['n_sequences']

        cls._usable_start = run_information['usable_start']

        cls._detail_level = detail_level

        cls._n_trains = None
        cls._n_pulses = None

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
        self.__class__._n_sequences = self._n_sequences
        # TODO if res is different from run_information['n_sequence'] the rest
        # of the data has to be loaded

    def test_n_train_equal(self):
        """
        Checks if number of trains is equal for all module (per seq)
        """

        res = []
        last_idxs = []
        for channel in np.arange(self._n_channels):
            d = self._data[channel]['header_train_id']

            tmp = []
            lidx_tmp =[]
            for seq in range(self._n_sequences):
                # instead of getting the length of d (without the trailing
                # zeros) just getting the index of the last entry with content
                last_idx = np.squeeze(np.where(d[seq] != 0))[-1]

                lidx_tmp.append(last_idx)
                tmp.append(last_idx + 1)

            last_idxs.append(lidx_tmp)
            res.append(tmp)

        assert_failed = False
        msg = ""
        res = np.array(res)
        self.__class__._n_trains = [None for i in range(self._n_sequences)]
        for seq in range(self._n_sequences):
            unique = np.unique(res[:, seq])

            try:
                self.assertEqual(len(unique), 1)

                self.__class__._n_trains[seq] = res[0, seq]
            except AssertionError:
                # tried to use subtest for this but it broke the new lines
                # in the check summary
                assert_failed = True
                msg += ("\nNot all modules have the same number of trains "
                        "for seq {}".format(seq))

                if self._detail_level == 1:
                    for ch in range(self._n_channels):
                        d = self._data[ch]['header_train_id']
                        msg += "\nChannel {:02}: {}".format(ch, res[ch, seq])

                        msg += "\t(trainid: "
                        last_idx = last_idxs[ch][seq]
                        if seq == 0:
                            msg += ("{} ... {}"
                                    .format(d[0][self._usable_start], d[0][last_idx]))
                        else:
                            msg += "{} ... {}".format(d[seq][0], d[seq][last_idx])
                        msg += ")"

        # for clarity only print one error message for all sequences
        if assert_failed:
            self.fail(msg)

    def test_dims_header(self):
        """
        Checks if the first dimension is equal for all datasets contained
        in 'header' (per module and per seq)
        """

        if self._detail_level == 1:
            msg = "Dimensions in header are not the same for:"

        assert_failed = False
        for channel in range(self._n_channels):
            for seq in range(self._n_sequences):
                fname = self._file_raw_temp.format(channel, seq)

                group_name = self._path['header'].format(channel)

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

                    if self._detail_level == 1:
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

        if self._detail_level == 1:
            msg = "Dimensions in image are not the same for: "

        assert_failed = False
        for channel in range(self._n_channels):
            for seq in range(self._n_sequences):
                fname = self._file_raw_temp.format(channel, seq)

                group_name = self._path['image'].format(channel)

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

                    if self._detail_level == 1:
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

        if self._detail_level == 1:
            msg = "Dimensions in trailer are not the same for:"

        assert_failed = False
        for channel in range(self._n_channels):
            for seq in range(self._n_sequences):
                fname = self._file_raw_temp.format(channel, seq)

                group_name = self._path['trailer'].format(channel)

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

                    if self._detail_level == 1:
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
        msg = ""

        first_train_ids = [d['header_train_id'][0][self._usable_start + 0]
                           for d in data]
        train_id_start = np.min(first_train_ids)

        diff_first_train = np.where(first_train_ids != train_id_start)[0]

        if self._detail_level == 1:
            msg = ("\nChannels with shifted first train id: {}\n"
                   .format(diff_first_train))
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
                d_detector = self._data[channel]['detector_train_id'][seq]
                d_header = self._data[channel]['header_train_id'][seq]
                d_trailer = self._data[channel]['trailer_train_id'][seq]

                detector_vs_header = (d_detector == d_header).all()
                header_vs_trailer = (d_header == d_trailer).all()
                res = np.logical_and(detector_vs_header, header_vs_trailer)

                try:
                    self.assertTrue(res)
                except AssertionError:
                    assert_failed = True

                    if self._detail_level == 1:
                        msg = ("\nTrainIds from detector, header and trailer "
                               "do not match for channel {:02}, seq {}"
                               .format(channel, seq))

        # for clarity only print one error message for all sequences
        if assert_failed:
            self.fail(msg)

    def test_data_vs_pulsec(self):
        """
        Checks if the sum of the pulseCount entries is corresponding to the
        data
        """

        if self._detail_level == 1:
            msg = ("\nPulseCount and data shape do not match for the following\n"
                   "channels and sequences (pulseCount sum vs data shape):")

        assert_failed = False
        for channel in range(self._n_channels):
            for seq in range(self._n_sequences):
                d = self._data[channel]['pulse_count'][seq]
                n_total_pulses = np.sum(d)

                fname = self._file_raw_temp.format(channel, seq)
                group_name = self._path['data'].format(channel)

                f = h5py.File(fname, "r")
                data_shape = f[group_name].shape
                f.close()

                try:
                    self.assertEqual(n_total_pulses, data_shape[0])
                except AssertionError:
                    assert_failed = True

                    if self._detail_level == 1:
                        msg += ("\nChannel {:02}, sequence {} ({} vs {})"
                                .format(channel, seq, n_total_pulses,
                                        data_shape[0]))

        # for clarity only print one error message for all channels and
        # sequences
        if assert_failed:
            self.fail(msg)

    def test_train_id_diff(self):
        """
        Checks if the trainId is monotonically increasing
        """

        if self._detail_level == 1:
            msg = ("\nTrainId is not monotonically increasing for the "
                   "following channels and sequences")

        assert_failed = False
        for channel in range(self._n_channels):
            for seq in range(self._n_sequences):
                train_id = self._data[channel]['header_train_id'][seq]

                # remove placeholders
                train_id = np.where(train_id != 0)

                diff = np.diff(train_id)

                try:
                    self.assertTrue(np.all(diff > 0))
                except AssertionError:
                    assert_failed = True

                    if self._detail_level == 1:
                        msg += ("\nChannel {:02}, sequence {}"
                                .format(channel, seq))

        # for clarity only print one error message for all channels and
        # sequences
        if assert_failed:
            self.fail(msg)

    def test_train_id_tzero(self):
        """
        Checks number of placeholder in trainId and if they are always at the
        end
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

                if self._detail_level == 1:
                    msg += ("\nSequence {} has not the same number of trailing "
                            "zeros for all channels\n({})".format(seq, r))

        # for clarity only print one error message for all sequences
        if assert_failed:
            self.fail(msg)

    def test_train_id_zeros(self):
        """
        Checks if trainId contains zeros which are not at the end
        """

        if self._detail_level == 1:
            msg = ("\nTrainid contains zeros which are not at the end for "
                   "following channels and sequences:")

        assert_failed = False
        for ch, _ in enumerate(data):
            d = data[ch]['header_train_id']

            for seq in range(len(d)):

                eq_zero_idx = np.where(data[seq] == 0)[0]
                diff_zeros = np.diff(eq_zero_idx)
                non_consecutive_idx = np.where(diff_zeros != 1)[0]

                try:
                    self.assertEqual(non_consecutive_idx.size, 0)
                except AssertionError:
                    assert_failed = True

                    if self._detail_level == 1:
                        msg += ("\nChannel {:02}, seq {}".format(ch, seq))

        # for clarity only print one error message for all channels and
        # sequences
        if assert_failed:
            self.fail(msg)

    def test_train_loss(self):
        """
        Checks for missing entries in trainId
        """
        # If there is massive train loss this gives the option to investigate
        # the train loss indices
        # but by default the trainIDs are not displayed (for massive train loss)
        if self._detail_level == 2:
            show_idx_in_short_msg = True
        else:
            show_idx_in_short_msg = False

        assert_failed = False
        msg = ""
        last_seq = None
        n_trains_lost = {}
        n_trains_lost_total = 0
        msg_tmp = ["" for ch in range(self._n_channels)]
        for ch, _ in enumerate(data):
            n_trains_lost[ch] = 0
            for seq in range(self._n_sequences):

                d = data[ch]['header_train_id'][seq]

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

                    if self._detail_level == 1:
                        msg += (
                            "\nChannel {:02}, seq {}: Train loss found at indices:"
                            .format(ch, seq)
                        )

                    elif self._detail_level == 2:
                        msg_tmp[ch] += (
                            "\nseq {}: Train loss found at indices:"
                            .format(seq)
                        )

                    for i, idx in enumerate(train_loss_idx):
                        idx_o = idx_no_zeros[idx]

                        if idx_o > 0:
                            start = idx_o - 1
                        else:
                            start = idx_o
                        stop = idx_o + 3

                        if seq == 0:
                            m = ("\nidx {}: ... {} ..."
                                 .format(idx_o + self._usable_start, str(d[start:stop])[1:-1]))
                        else:
                            m = ("\nidx {}: ... {} ..."
                                 .format(idx_o, str(d[start:stop])[1:-1]))

                        if self._detail_level == 1:
                            msg += m
                        elif self._detail_level == 2 and i < 3:
                            msg_tmp[ch] += m

                        n_trains_lost[ch] += diff[idx] - 1

                # check transition between two sequences
                if seq != 0:
                    try:
                        self.assertEqual(last_seq + 1, d_no_zeros[0])
                    except AssertionError:
                        assert_failed = True

                        if sefl._detail_level == 1:
                            msg += ("\nChannel {:02}: Train loss found between "
                                   "sequences {} and {}\n"
                                   .format(ch, seq - 1, seq))
                            msg += ("(end of seq {}: {}, start of seq {}: {})"
                                    .format(seq - 1, last_seq, seq, d_no_zeros[0]))

                        n_trains_lost[ch] += d_no_zeros[0] - last_seq - 1

                # keep the last trainId for the next iteration
                last_seq = d_no_zeros[-1]

            n_trains_lost_total += n_trains_lost[ch]

        # for clarity only print one error message for all channels and
        # sequences
        if assert_failed:
            if n_trains_lost_total > 5:
                # This overwrites msg
                short_msg = ""

                if self._detail_level == 2:
                    for ch in range(self._n_channels):
                        short_msg += ("\nChannel {:02}:".format(ch))
                        if msg_tmp[ch]:
                            short_msg += msg_tmp[ch]
                        else:
                            short_msg += " None"

                    short_msg += "\n\n"

                if self._detail_level != 0:
                    short_msg += "Train loss found for:"
                    for ch in n_trains_lost:
                        short_msg += ("\nChannel {:02}: {}"
                                      .format(ch, n_trains_lost[ch]))

                    short_msg += ("\nTotal number of trains lost: {}"
                                  .format(n_trains_lost_total))

                self.fail(short_msg)
            else:
                if self._detail_level != 0:
                    msg += ("\n\nTotal number of trains lost: {}"
                            .format(n_trains_lost_total))
                    for ch in n_trains_lost:
                        if n_trains_lost[ch]:
                            msg += "\nChannel {:02}: {}".format(ch, n_trains_lost[ch])

                self.fail(msg)

    def test_data_tzeros(self):
        """
        Check if extra data entries are trailing zeros
        """

        if self._detail_level == 1:
            msg = "\nData containes extra data which is not zero for:"

        assert_failed = False
        for channel in range(self._n_channels):
            for seq in range(self._n_sequences):
                d = self._data[channel]['pulse_count'][seq]
                n_total_pulses = np.sum(d)

                fname = self._file_raw_temp.format(channel, seq)
                group_name = self._path['data'].format(channel)

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

                    if self._detail_level == 1:
                        msg += ("\nChannel {:02}, seq{} (number of zeros {})"
                                .format(channel, seq, extra_data.shape[0]))

        # for clarity only print one error message for all channels and
        # sequences
        if assert_failed:
            self.fail(msg)

    def test_data_vs_tr_id(self):
        """
        Checks if the dimension of the image trainId is corresponding to the
        data
        """

        if self._detail_level == 1:
            msg = "\nTrainId and data shape do not match for:"

        assert_failed = False
        for channel in range(self._n_channels):
            for seq in range(self._n_sequences):
                fname = self._file_raw_temp.format(channel, seq)

                group_name = self._path['image_train_id'].format(channel)
                data_name = self._path['data'].format(channel)

                f = h5py.File(fname, "r")
                train_id_shape = f[group_name].shape
                data_shape = f[data_name].shape
                f.close()

                try:
                    self.assertEqual(train_id_shape[0], data_shape[0])
                except AssertionError:
                    assert_failed = True

                    if self._detail_level == 1:
                        msg += ("Channel {:02}, sequence {} ({} vs {})\n"
                                .format(channel, seq, train_id_shape[0],
                                        data_shape[0]))

        # for clarity only print one error message for all channels and
        # sequences
        if assert_failed:
            self.fail(msg)

    def test_dim_first_last(self):
        """
        Checks if the dimensions of the arrays providing the information about
        the start and end of the train are of the same dimensions
        """

        if self._detail_level == 1:
            msg = ("\nNumber of train start and train end indices do not match "
                   "for:")

        assert_failed = False
        for channel in range(self._n_channels):
            for seq in range(self._n_sequences):
                len_first = len(self._data[channel]['image_first'][seq])
                len_last = len(self._data[channel]['image_last'][seq])

                try:
                    self.assertEqual(len_first, len_last)
                except AssertionError:
                    assert_failed = True

                    if self._detail_level == 1:
                        msg += ("Channel {:02}, sequence {} ({} vs {})\n"
                                .format(channel, seq, len_first, len_last))

        # for clarity only print one error message for all channels and
        # sequences
        if assert_failed:
            self.fail(msg)

    def test_pulse_loss(self):
        """
        Checks if all trains have the same number of pulses
        """

        if self._detail_level == 1:
            msg = "\nPulse loss found for:\n"

        assert_failed = False
        for channel in range(self._n_channels):
            pulses = []
            for seq in range(self._n_sequences):
                p = list(np.unique(self._data[channel]['pulse_count'][seq]))
                pulses += p

            # TODO access pulses where status != 0
            n_pulses = np.unique(pulses[pulses != 0])
            n_pulses_lost = max(n_pulses) - min(n_pulses)

            try:
                self.assertEqual(n_pulses_lost, 0)
            except AssertionError:
                assert_failed = True

                if self._detail_level == 1:
                    msg += "Channel {:02}: {}\n".format(channel, n_pulses_lost)

        # for clarity only print one error message for all channels and
        # sequences
        if assert_failed:
            self.fail(msg)

    # per test
    def tearDown(self):
        pass

    # for the whole class
    @classmethod
    def tearDownClass(cls):
        global run_information

        run_information['n_trains'] = cls._n_trains
        run_information['n_pulses'] = cls._n_pulses


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


def get_info(instrument_cycle, beamtime, run):
    global file_raw_temp
    global path_temp
    global run_information

    # get information

    n_channels = run_information['n_channels']
    n_sequences = run_information['n_sequences']

    n_trains = run_information['n_trains']

    data_shape = [[] for i in range(n_channels)]
    n_pulses = 0
    for channel in range(n_channels):
        for seq in range(n_sequences):
            d = data[channel]['pulse_count'][seq]
            n_pulses = max(n_pulses, np.max(d))

        seq = 0
        fname = file_raw_temp.format(channel, seq)
        group_name = path_temp['data'].format(channel)

        f = h5py.File(fname, 'r')
        data_shape[channel] = f[group_name].shape[0]
        f.close()

    # display information

    max_len = 70
    print()
    print("-" * max_len)
    print("Information about the run {} ".format(run))
    print("(instrument cycle {}, beamtime {})"
          .format(instrument_cycle, beamtime))
    print("-" * max_len)

    print("Number of channels:", n_channels)
    print("Number of sequences:", n_sequences)

    if len(np.unique(data_shape)) == 1:
        print("Data shape:", data_shape[0])
    else:
        print("Data dimension differ for different channels")
        for channel in range(n_channels):
            print("Channel {:02}: {}".format(channel, data_shape[channel]))

    if None not in n_trains:
        print("Number of trains: {}".format(np.sum(n_trains)))
    else:
        not_none = [t for t in n_trains if t is not None]
        print("Number of trains: >{}".format(np.sum(not_none)))
    print("Number of trains per sequence:", n_trains)

    print("Number of pulses:", n_pulses)


if __name__ == "__main__":

    # instrument_cycle = "201730"
    # bt = "p900009"
    # r = 709
    args = get_arguments()

    show_info = args.show_info
    instrument_cycle = args.instrument_cycle
    bt = args.beamtime

    beamtime = "{}/{}".format(instrument_cycle, bt)
    run = args.run
    detail_level = args.detail_level

#    itersuite = unittest.TestLoader().loadTestsFromTestCase(SanityChecks)
    itersuite = suite()
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(itersuite)

    if show_info:
        get_info(instrument_cycle, bt, run)

