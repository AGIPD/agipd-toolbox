import h5py
import numpy as np
import os
import sys
import configparser

try:
    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
except:
    CURRENT_DIR = os.path.dirname(os.path.realpath('__file__'))

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
SRC_PATH = os.path.join(BASE_PATH, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import utils  # noqa E402
import cfel_optarg  # noqa E402


class XfelLayout():
    def __init__(self,
                 in_fname,
                 runs,
                 use_interleaved,
                 preproc_fname=None,
                 max_part=False,
                 asic=None):

        self.in_fname = in_fname
        self.runs = runs
        self.use_interleaved = use_interleaved
        self.preprocessing_fname = preproc_fname
        self.max_part = max_part
        self.asic = asic

        self.path_temp = {
            'status': "INDEX/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/image/status",
            'image_first': ("INDEX/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/"
                            "image/first"),
            'image_last': "INDEX/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/image/last",

            'data': "INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/image/data",
            'cellid': ("INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/"
                       "image/cellId"),

        }

        self.path = {}

    def get_preproc_res(self):
        if self.preprocessing_fname is None:
            return {}
        else:
            config = configparser.RawConfigParser()
            config.read(self.preprocessing_fname)

            return cfel_optarg.parse_parameters(config)

    def initiate(self, n_rows, n_cols):

        self.preproc = self.get_preproc_res()

        self.n_memcells = self.preproc['general']['n_memcells']
        if self.use_interleaved:
            # TODO: should this go into the preprocessing?
            # n_memcells has to be an odd number because every memory cell
            # need analog and digital data
            if self.n_memcells % 2 != 0:
                self.n_memcells += 1

            self.n_memcells = self.n_memcells // 2

            # xfel format has swapped rows and cols
            self.raw_shape = (self.n_memcells, 2, 2,
                              n_cols, n_rows)
        else:
            self.raw_shape = (self.n_memcells, 2,
                              n_cols, n_rows)

        print("in_fname", self.in_fname)
        print("Number of memory cells found", self.n_memcells)

        split_tmp = self.in_fname.split("-")
        self.channel = int(split_tmp[-2].split("AGIPD")[1])
        print("channel", self.channel)

        self.n_frames_total = (self.preproc['general']['n_trains_total'] *
                               self.n_memcells)
        print("n_frames_total", self.n_frames_total)

        ch = 'channel{:02}'.format(self.channel)
        self.train_pos = self.preproc[ch]['train_pos']

        for key in self.path_temp:
            self.path[key] = self.path_temp[key].format(self.channel)

        return (self.n_memcells,
                self.n_frames_total,
                self.raw_shape,
                self.path['data'])

    def load(self,
             fname,
             seq,
             load_idx_rows,
             load_idx_cols,
             file_content,
             tmp_data,
             pos_idxs):

        f = None
        try:
            f = h5py.File(fname, "r")
            raw_data = f[self.path['data']][()]

            status = np.squeeze(f[self.path['status']][()]).astype(np.int)
            first = np.squeeze(f[self.path['image_first']][()]).astype(np.int)
            last = np.squeeze(f[self.path['image_last']][()]).astype(np.int)
            cellid = np.squeeze(f[self.path['cellid']][()]).astype(np.int)
        finally:
            if f is not None:
                f.close()

        print("raw_data.shape", raw_data.shape)
        print("self.raw_shape", self.raw_shape)

        if self.use_interleaved:
            # currently the splitting in digital and analog does not work
            # for XFEL
            # -> all data is in the first entry of the analog/digital
            #    dimension
            raw_data = raw_data[:, 0, ...]

        raw_data = utils.convert_to_agipd_format(self.channel, raw_data)

        last_index = np.squeeze(np.where(status != 0))[-1]
        print("last_index", last_index)

        for i, source_fidx in enumerate(first[:last_index + 1]):
            source_lidx = last[i] + 1

            # Get train position (taken care of train loss)
            target_fidx = self.train_pos[seq][i] * self.n_memcells

            # Detect pulse loss
            diff = np.diff(cellid[source_fidx:source_lidx])
            cell_loss = np.squeeze(np.where(diff != 1))
            if cell_loss.size != 0:
                print("cell_loss", cell_loss)

            # Fill up pulse loss
            source_interval = [source_fidx, source_fidx]
            for cidx in np.concatenate((cell_loss, [source_lidx])):
                source_interval[1] = cidx

                t_start = target_fidx + cellid[source_interval[0]]
                t_stop = target_fidx + cellid[source_interval[1] - 1] + 1
                target_interval = [t_start, t_stop]

#                print("intervals", source_interval, target_interval)

                for index_set in pos_idxs:
                    pos_idx_rows = index_set[0]
                    pos_idx_cols = index_set[1]

                    source_idx = (slice(*source_interval),
                                  Ellipsis,
                                  pos_idx_rows,
                                  pos_idx_cols)
                    target_idx = (slice(*target_interval),
                                  Ellipsis,
                                  pos_idx_rows,
                                  pos_idx_cols)

                    try:
                        tmp_data[target_idx] = raw_data[source_idx]
                    except:
                        print("tmp_data.shape", tmp_data.shape)
                        print("raw_data.shape", raw_data.shape)
                        print("target_idx", target_idx)
                        print("source_idx", source_idx)
                        raise

                source_interval[0] = source_interval[1]
