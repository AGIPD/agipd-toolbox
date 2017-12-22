import h5py
import numpy as np
import os
import sys
import configparser

try:
    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
except:
    CURRENT_DIR = os.path.dirname(os.path.realpath('__file__'))

BASE_PATH = os.path.dirname(os.path.dirname(CURRENT_DIR))
SRC_PATH = os.path.join(BASE_PATH, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import utils  # noqa E402


class PreprocessingXfel():
    def __init__(self, in_fname, out_fname):

        self.in_fname = in_fname
        self.out_fname = out_fname

        self.n_channels = 16
        self.path_temp = {
            'status': ("INDEX/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/"
                       "image/status"),
            'pulse_count': ("INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/"
                            "header/pulseCount"),
            'header_trainid': ("INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/"
                               "header/trainId")
        }

        self.prop = {}

    def run(self):

        self.prop["general"] = {
            "n_seqs": self.get_n_seqs(),
            "n_memcells": self.get_n_memcells(),
            "shifting": np.zeros(self.n_channels, dtype=np.int),
            "max_shifting": 0,
            "n_trains": [],
        }

        for ch in range(self.n_channels):
            self.prop["channel{:02}".format(ch)] = {
                "train_pos": []
            }

        self.evaluate_trainid()

        self.write()

    def get_n_seqs(self):
        return 3

    def get_n_memcells(self):
        seq = 0
        channel = 0

        fname = self.in_fname.format(channel, seq)
        read_in_path = self.path_temp['pulse_count'].format(channel)

        f = h5py.File(fname, "r")
        in_data = f[read_in_path][()].astype(np.int)
        f.close()

        return max(in_data)

    def evaluate_trainid(self):
        n_seqs = self.prop["general"]["n_seqs"]
        n_trains = self.prop["general"]["n_trains"]

        seq_offset = [0 for ch in range(self.n_channels)]

        for seq in range(n_seqs):
            if seq == 0:
                usable_start = 2
            else:
                usable_start = 0

            trainids = []
            for ch in range(self.n_channels):
                fname = self.in_fname.format(ch, seq)

                status_path = self.path_temp['status'].format(ch)
                trainid_path = self.path_temp['header_trainid'].format(ch)

                f = h5py.File(fname, "r")

                # number of trains actually stored for this channel + sequence
                s = f[status_path][()].astype(np.int)
                n_tr = len(np.squeeze(np.where(s != 0)))

                # do not read in the trailing zeros
                tr = f[trainid_path][:n_tr].astype(np.int)

                f.close()

                trainids.append(tr)

            # find the starting trainid
            first_trainids = [tr[usable_start + 1] for tr in trainids]
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
                shifting = self.prop["general"]["shifting"]
                for i in diff_first_train:
                    cond = trainids[corr_first_train[0]] == first_trainids[i]
                    idx = np.squeeze(np.where(cond))

                    if idx.size != 1:
                        raise Exception("trainid was found more than once")
                    idx = int(idx) - usable_start - 1

                    shifting[i] = idx

                self.prop["general"]["max_shifting"] = max(shifting)

            else:
                # check transition between sequences for train loss
                first_trainid = [tr[0] for tr in trainids]
                seq_step = (np.array(first_trainid) -
                            np.array(prev_last_trainid))  # noqa F821

                seq_offset = [
                    self.prop["channel{:02}".format(ch)]["train_pos"][-1][-1]
                    + seq_step[ch]
                    for ch in range(self.n_channels)
                ]

            prev_last_trainid = [tr[-1] for tr in trainids]  # noqa F841

            for ch in range(self.n_channels):
                p = self.prop["channel{:02}".format(ch)]

                tr = trainids[ch]
                # this also indicates train loss
                train_number = tr - tr[usable_start] + usable_start
                if seq == 0:
                    train_number[:usable_start] = range(usable_start)
                    train_number += shifting[ch]

                train_number += seq_offset[ch]
                p["train_pos"].append(train_number)

            train_pos_of_seq = [
                self.prop["channel{:02}".format(ch)]["train_pos"][seq]
                - seq_offset[ch]
                - shifting[ch]
                for ch in range(self.n_channels)
            ]

            n_tr = 0
            for trn in train_pos_of_seq:
                # trn starts counting with 0
                n_tr = max(n_tr, max(trn) + 1)

            n_trains.append(n_tr)

            self.prop["general"]["n_trains_total"] = (
                sum(n_trains) + self.prop["general"]["max_shifting"]
            )

        print("shifting:", self.prop["general"]["shifting"])
        print("max_shifting", self.prop["general"]["max_shifting"])
        print(self.prop["general"]["n_trains"])

    def write(self):
        config = configparser.RawConfigParser()

        for section in self.prop:
            config.add_section(section)

            for key in self.prop[section]:
                value = self.prop[section][key]

                if type(value) == np.ndarray:
                    value = value.tolist()
                elif type(value) == list and type(value[0]) == np.ndarray:
                    value = [i.tolist() for i in value]

                config.set(section, key, value)

        with open(self.out_fname, 'w') as configfile:
            config.write(configfile)

if __name__ == "__main__":
    run = 709
    beamtime = "201730/p900009"

    file_raw_temp = ("/gpfs/exfel/exp/SPB/{}/raw/r{:04d}/RAW-R{:04d}-"
                     .format(beamtime, run, run) +
                     "AGIPD{:02d}-S{:05d}.h5")

    preprocessing_file = os.path.join(BASE_PATH, "preprocessing.result")

    p = PreprocessingXfel(file_raw_temp, preprocessing_file)
    p.run()
