import configparser
import copy
import glob
import h5py
import numpy as np
import os
import sys

try:
    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
except:
    CURRENT_DIR = os.path.dirname(os.path.realpath('__file__'))

BASE_PATH = os.path.dirname(os.path.dirname(CURRENT_DIR))
SRC_PATH = os.path.join(BASE_PATH, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import utils  # noqa E402

from _version import __version__


class Preprocess(object):
    def __init__(self, in_fname, out_fname, use_interleaved=False, interactive=False):

        self._in_fname = in_fname
        self._out_fname = out_fname
        self._use_interleaved = use_interleaved
        self._interactive = interactive

        self._usable_start_first_seq = 0
        #self._usable_start_first_seq = 2

        self._n_channels = 16
        self._path = {
            'image_first': ("INDEX/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/"
                            "image/first"),
            'train_count': ("INDEX/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/"
                            "image/count"),
            'trainid': ("INDEX/trainId"),
            'pulse_count': ("INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/"
                            "header/pulseCount"),
            'cellid': ("INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/"
                            "image/cellId"),
#            'status': ("INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/"
#                       "image/status"),
            #'header_trainid': ("INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/"
            #                   "header/trainId")
        }

        # trainids which are jumping by this amount are seen as outlier and
        # are being removed
        self._outlier_threshold = 1000

        self._prop = {}

    def run(self):

        for ch in range(self._n_channels):
            self._prop["channel{:02}".format(ch)] = {
                "n_seqs": self._get_n_seqs(channel=ch),
                "n_trains": [],
                "train_pos": [],
                "trainid_outliers": [],
            }

        self._prop["software"] = {
            "version": __version__
        }

        self._prop["general"] = {
            "n_memcells": self._get_n_memcells(),
            "shifting": np.zeros(self._n_channels, dtype=np.int),
            "max_shifting": 0,
            "max_n_trains": [],
            "max_n_seqs": max([self._prop["channel{:02}".format(ch)]["n_seqs"]
                               for ch in range(self._n_channels)])
        }

        self._evaluate_trainid()

        if self._interactive:
            return self._prop
        else:
            self._write()

    def _get_n_seqs(self, channel):
        split = self._in_fname.rsplit("-AGIPD", 1)
        regex = split[0] + "-AGIPD{:02}*"

        found_files = sorted(glob.glob(regex.format(channel)))

        parts = len(found_files)

#        print("parts/sequences for channel {}: {}".format(channel, parts))
        return parts

    def _get_n_memcells(self):
        seq = 0
        channel = 0

        fname = self._in_fname.format(channel, part=seq)
        #read_in_path = self._path['pulse_count'].format(channel)
        read_in_path = self._path['cellid'].format(channel)

        with h5py.File(fname, "r") as f:
            in_data = f[read_in_path][()].astype(np.int)

        n_memcells = max(in_data)[0] + 1
        #print("n_memcells: ", n_memcells)
        if self._use_interleaved:
            # _n_memcells has to be an odd number because every memory cell
            # need analog and digital data
            if n_memcells % 2 != 0:
                n_memcells += 1

            n_memcells = n_memcells // 2

        return n_memcells

    def _evaluate_trainid(self):
        max_n_seqs = self._prop["general"]["max_n_seqs"]
        max_n_trains = self._prop["general"]["max_n_trains"]

        seq_offset = [0 for ch in range(self._n_channels)]

        for seq in range(max_n_seqs):
            if seq == 0:
                usable_start = self._usable_start_first_seq
            else:
                usable_start = 0

            trainids = []
            channels_with_trains = []
            for ch in range(self._n_channels):
                fname = self._in_fname.format(ch, part=seq)

                #status_path = self._path['status'].format(ch)
                trainid_path = self._path['trainid'].format(ch)
                #firstpath = self._path['image_first'].format(ch)
                count_path = self._path['train_count'].format(ch)

                if self._prop["channel{:02}".format(ch)]["n_seqs"] <= seq:
                    print("Channel {} does not have a sequence {}"
                          .format(ch, seq))
                    continue

                with h5py.File(fname, "r") as f:
                    count = f[count_path][()].astype(np.int)

                    count_not_zero = np.where(count != 0)[0]
                    n_tr = len(count_not_zero)

                    ch_str = "channel{:02}".format(ch)
                    n_trains = self._prop[ch_str]["n_trains"]
                    n_trains.append(n_tr)

                    if n_tr == 0:
                        continue

                    channels_with_trains.append(ch)

                    first_index = count_not_zero[0]
                    if n_tr == 1:
                        # if the file only contains one train
                        last_index = count_not_zero[0] + 1
                    else:
                        last_index = count_not_zero[-1] + 1
#                    print("first_index", first_index)
#                    print("last_index", last_index)

                    # do not read in leading or trailing zeros
                    tr = f[trainid_path][first_index:last_index].astype(np.int)
#                    if seq == 0:
#                        print("ch", ch, "tr", tr[0])

                trainids.append(tr)

#            print("channels_with_trains", channels_with_trains)

            if trainids == []:
                max_n_trains.append(0)
                continue

            # find the starting trainid
            first_trainids = [tr[usable_start] for tr in trainids]
            trainid_start = np.min(first_trainids)

            # find the channels which start with a different trainid
            cond = first_trainids != trainid_start
            diff_first_train = np.squeeze(np.where(cond))
            corr_first_train = np.array(
                [i for i in range(len(first_trainids))
                 if i not in diff_first_train]
            )

            if seq == 0:
                # determine shifting
                shifting = self._prop["general"]["shifting"]
                for i in diff_first_train:
                    cond = trainids[corr_first_train[0]] == first_trainids[i]
                    idx = np.squeeze(np.where(cond))

                    if idx.size != 1:
                        raise Exception("trainid was found more than once")
                    idx = int(idx) - usable_start
                    #idx = int(idx) - usable_start - 1

                    shifting[i] = idx

                self._prop["general"]["max_shifting"] = max(shifting)
                if self._prop["general"]["max_shifting"] != 0:
                    print("first_trainids", first_trainids)

            else:
                # check transition between sequences for train loss
                first_trainid = [tr[0] for tr in trainids]
                # some channels have lesser number of sequences
                channels_still_with_data = [prev_channels_with_trains.index(i)
                                            for i in channels_with_trains]
                prev_last_trainid_mod = [
                    tr for i, tr in enumerate(prev_last_trainid)
                    if i in channels_still_with_data
                ]
                seq_step = (np.array(first_trainid) -
                            np.array(prev_last_trainid_mod))  # noqa F821

                seq_offset = [
                    # last trainid of last sequence
                    self._prop["channel{:02}".format(ch)]["train_pos"][-1][-1]
                    + seq_step[i]
                    for i, ch in enumerate(channels_with_trains)
                ]

            prev_last_trainid = [tr[-1] for tr in trainids]  # noqa F841
            prev_channels_with_trains = copy.copy(channels_with_trains)

            i  = 0
            # finding train loss inside a sequence
            for ch in range(self._n_channels):
                p = self._prop["channel{:02}".format(ch)]

                n_tr = p["n_trains"]
                n_seqs = p["n_seqs"]
                if n_seqs <= seq or n_tr[seq] == 0:
                    continue

                tr = trainids[i]
                # this also indicates train loss
                train_number = tr - tr[usable_start] + usable_start
                if seq == 0:
                    train_number[:usable_start] = range(usable_start)
                    train_number += shifting[ch]

                outliers = self._find_outlier_trainids(train_number)

                p["trainid_outliers"].append(outliers)

                # outliers should not be seen as trains
                n_trains = self._prop["channel{:02}".format(ch)]["n_trains"]
                n_trains[seq] -= len(outliers)

                train_number += seq_offset[i]
                p["train_pos"].append(train_number)

                i += 1

#            n_tr = max([self._prop["channel{:02}".format(ch)]["n_trains"][seq]
#                        for ch in channels_with_trains])

            # determine max number of trains for each sequence
            # use train_pos instead of n_trains to also cover lost trains
            train_pos_of_seq = [
                self._prop["channel{:02}".format(ch)]["train_pos"][seq]
                - seq_offset[i]
                - shifting[i]
                for i, ch in enumerate(channels_with_trains)
            ]

            n_tr = 0
            for i, trn in enumerate(train_pos_of_seq):
                # get rid of outliers
                ch = self._prop["channel{:02}".format(channels_with_trains[i])]
                outliers = ch["trainid_outliers"][seq]
                trn[outliers] = 0

                # trn starts counting with 0
                n_tr = max(n_tr, max(trn) + 1)

            max_n_trains.append(n_tr)

        # determine number of total trains
        n_trains_individual = [
            np.sum(self._prop["channel{:02}".format(ch)]["n_trains"])
            for ch in range(self._n_channels)
        ]

        self._prop["general"]["n_trains_total"] = (
            np.max(n_trains_individual)
            + self._prop["general"]["max_shifting"]
        )

        print("shifting:", self._prop["general"]["shifting"])
        print("max_shifting:", self._prop["general"]["max_shifting"])
        print("max_n_trains:", self._prop["general"]["max_n_trains"])
        print("n_trains_total:", self._prop["general"]["n_trains_total"])

    def _find_outlier_trainids(self, train_number):
        outliers = []

        tr_diff = np.diff(train_number)
        pot_outliers = np.squeeze(np.where(tr_diff > self._outlier_threshold))
        # diff is refering to the index before the actual outlier
        pot_outliers += 1

        if pot_outliers.size == 1:
            pot_outliers = [int(pot_outliers)]

        for p_o in pot_outliers:
            if train_number[p_o - 1] + 1 == train_number[p_o + 1]:
                outliers.append(p_o)

        return outliers

    def _write(self):
        config = configparser.RawConfigParser()

        write_order = ["software", "general"]
        write_order += sorted([s for s in self._prop
                               if s not in ["software", "general"]])

        for section in write_order:
            config.add_section(section)

            subsec_order = sorted(list(self._prop[section].keys()))
            for key in subsec_order:
                value = self._prop[section][key]

                try:
                    if type(value) == np.ndarray:
                        value = value.tolist()
                    elif type(value) == list and value == []:
                        value = None
                    elif type(value) == list and type(value[0]) == np.ndarray:
                        value = [i.tolist() for i in value]
                except:
                    print(key, value)
                    raise

                config.set(section, key, value)

        with open(self._out_fname, 'w') as configfile:
            config.write(configfile)


if __name__ == "__main__":
    run = 459
    beamtime = "201830/p900019"

    base_path = "/gpfs/exfel/exp/SPB"
    subdir = "scratch/user/kuhnm/tmp"
    run_subdir = "r{:04}".format(run)

    file_raw_temp = ("/gpfs/exfel/exp/SPB/{}/raw/r{:04}/RAW-R{:04}-"
                     .format(beamtime, run, run) +
                     "AGIPD{:02}-S{part:05}.h5")

    preprocessing_file = os.path.join(base_path,
                                      beamtime,
                                      subdir,
                                      run_subdir,
                                      "{}-preprocessing.result"
                                      .format(run_subdir.upper()))
#    preprocessing_file = os.path.join(BASE_PATH, "preprocessing.result")


    p = Preprocess(in_fname=file_raw_temp,
                   out_fname=preprocessing_file,
                   interactive=True)
    my_result = p.run()

    print(my_result.keys())
