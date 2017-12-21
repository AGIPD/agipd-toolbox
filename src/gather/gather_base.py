import h5py
import numpy as np
import os
import sys
import time
import glob

try:
    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
except:
    CURRENT_DIR = os.path.dirname(os.path.realpath('__file__'))

BASE_PATH = os.path.dirname(os.path.dirname(CURRENT_DIR))
SRC_PATH = os.path.join(BASE_PATH, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import utils  # noqa E402


class AgipdGatherBase():
    def __init__(self, in_fname, out_fname, runs, max_part=False,
                 asic=None, use_xfel_format=False, backing_store=True):

        self.in_fname = in_fname
        self.out_fname = out_fname
        self.runs = [int(r) for r in runs]

        self.max_part = max_part
        self.asic = asic
        self.use_xfel_format = use_xfel_format
        self.backing_store = backing_store

        self.use_xfel_format = use_xfel_format

        self.analog = None
        self.digital = None

        self.raw_shape = None
        self.tmp_shape = None
        self.n_frames = None
        self.target_shape = None

        self.target_index = None
        self.target_index_full_size = None
        self.source_index = None
        self.source_seq_number = None
        self.seq_number = None
        self.max_pulses = None

        # to use the interleaved or not interleaved format
        # self.use_interleaved = True
        self.use_interleaved = False

        self.n_rows_total = 128
        self.n_cols_total = 512

        self.a_row_start = None
        self.a_row_stop = None
        self.a_col_start = None
        self.a_col_stop = None

        self.get_parts()

        if self.n_parts == 0:
            msg = "No parts to gather found\n"
            msg += "in_fname={}".format(self.in_fname)
            raise Exception(msg)

        if self.asic is None:
            self.n_rows = self.n_rows_total
            self.n_cols = self.n_cols_total
        else:
            print("asic {}".format(self.asic))
            self.n_rows = self.asic_size
            self.n_cols = self.asic_size

            asic_order = utils.get_asic_order()
            mapped_asic = utils.calculate_mapped_asic(asic_order)
            print("mapped_asic={}".format(mapped_asic))

            (self.a_row_start,
             self.a_row_stop,
             self.a_col_start,
             self.a_col_stop) = utils.determine_asic_border(mapped_asic,
                                                            self.asic_size)

        self.intiate()

        print("\n\n\n"
              "start gather\n"
              "in_fname = {}\n"
              "out_fname ={}\n"
              "data_path = {}\n"
              .format(self.in_fname,
                      self.out_fname,
                      self.data_path))

        self.run()

    def get_parts(self):
        # remove extension
        prefix = self.in_fname.rsplit(".", 1)[0]
        # removet the part section
        prefix = prefix[:-10]
        # use the first run number to determine number of parts
        run_number = self.runs[0]
        prefix = prefix.format(run_number=run_number)
        print("prefix={}".format(prefix))

        part_files = glob.glob("{}*".format(prefix))

        self.n_parts = self.max_part or len(part_files)
        print("n_parts {}".format(self.n_parts))

    def intiate(self):
        if self.use_xfel_format:
            self.init_xfel()
        else:
            self.init_cfel()

        self.define_needed_data_paths()

        # tmp data is already converted into agipd format
        if self.use_interleaved:
            self.raw_tmp_shape = (self.n_frames_total,
                                  self.n_rows, self.n_cols)
        else:
            self.raw_tmp_shape = (self.n_frames_total, 2,
                                  self.n_rows, self.n_cols)

        self.tmp_shape = (-1, self.n_memcells, 2, self.n_rows, self.n_cols)

        self.target_shape = (-1, self.n_memcells, self.n_rows, self.n_cols)
        print("target shape:", self.target_shape)

    def init_xfel(self):
        data_path_prefix = "INSTRUMENT/SPB_DET_AGIPD1M-1/DET"
        data_path_postfix = "image/data"
        pulse_count_postfix = "header/pulseCount"

        # use part number 0 to get the information from
        run_number = self.runs[0]
        fname = self.in_fname.format(run_number=run_number, part=0)

        f = None
        try:
            f = h5py.File(fname, "r")

            # xfel uses the channel index inside the data path
            # automatically detect the path
            k = list(f[data_path_prefix].keys())[0]
            base_path = os.path.join(data_path_prefix, k)

            self.data_path = os.path.join(base_path, data_path_postfix)
            self.pulse_count_path = os.path.join(base_path,
                                                 pulse_count_postfix)

            raw_data_shape = f[self.data_path].shape
        except:
            print("Problems when reading file {}".format(fname))
            if f is not None:
                f.close()
                f = None

        self.get_number_of_frames()
        print("n_frames", self.n_frames)
        print("n_frames_total", self.n_frames_total)

        self.n_memcells = self.max_pulses
        if self.use_interleaved:
            self.n_memcells = self.n_memcells // 2
        print("Number of memory cells found", self.n_memcells)

        # xfel format has swapped rows and cols
        if self.use_interleaved:
            self.raw_shape = (self.n_memcells, 2, 2,
                              self.n_cols, self.n_rows)
        else:
            self.raw_shape = (self.n_memcells, 2,
                              self.n_cols, self.n_rows)

        self.channel = int(k.split("CH")[0])
        self.in_wing2 = utils.located_in_wing2(self.channel)

    def init_cfel(self):
        self.data_path = "entry/instrument/detector/data"
        self.digital_data_path = "entry/instrument/detector/data_digital"

        self.collection_path = "entry/instrument/detector/collection"
        self.seq_number_path = "entry/instrument/detector/sequence_number"
        self.total_lost_frames_path = ("{}/total_loss_frames"
                                       .format(self.collection_path))
        self.error_code_path = "{}/error_code".format(self.collection_path)
        self.frame_number_path = ("{}/frame_numbers"
                                  .format(self.collection_path))

        run_number = self.runs[0]
        fname = self.in_fname.format(run_number=run_number, part=0)

        f = None
        try:
            f = h5py.File(fname, "r")
            raw_data_shape = f[self.data_path].shape
        finally:
            if f is not None:
                f.close()

        # dark
        self.max_pulses = 704
        self.n_memcells = 352
        # xray
#            self.max_pulses = 2
#            self.n_memcells = 1

        self.get_number_of_frames()
        print("n_frames {}".format(self.n_frames))
        print("n_frames_total {}".format(self.n_frames_total))

        self.raw_shape = (self.n_memcells, 2, self.n_rows, self.n_cols)

    # to give classes which inherite from this class the possibility to define
    # file internal paths they need
    def define_needed_data_paths(self):
        pass

    def get_number_of_frames(self):
        f = None

        # take first run and asume that the others have as many frames
        # TODO check this
        run_number = self.runs[0]
        self.n_frames = 0
        self.max_pulses = 0
        n_trains = 0

        if self.use_xfel_format:
            for i in range(self.n_parts):
                fname = self.in_fname.format(run_number=run_number, part=i)
                try:
                    f = h5py.File(fname, "r")
                    pulse_count = f[self.pulse_count_path][()]
                finally:
                    if f is not None:
                        f.close()

                self.max_pulses = int(np.max((np.max(pulse_count),
                                              self.max_pulses)))

                # max_pulses has to be an odd number because every memory cell
                # need analog and digital data
                if self.use_interleaved and self.max_pulses % 2 != 0:
                    self.max_pulses += 1

                n_trains += pulse_count.size

            print("pulse_count.size", pulse_count.size)
            print("n_trains", n_trains)

            self.n_frames = n_trains
            self.n_frames_total = self.max_pulses * n_trains

        else:
            try:
                fname = self.in_fname.format(run_number=run_number, part=0)
                f = h5py.File(fname, "r", libver="latest", drivers="core")

                # TODO: verify that the shape is always right and not
                #       dependant on frame loss
                source_shape = f[self.data_path].shape
                exp_total_frames = f[self.frame_number_path][0]

            except:
                print("Error when getting shape")
                raise
            finally:
                if f is not None:
                    f.close()

            if (source_shape[1] != self.n_rows_total and
                    source_shape[2] != self.n_cols_total):
                msg = "Shape of file {} ".format(fname)
                msg += "does not match requirements\n"
                msg += "source_shape = {}".format(source_shape)
                raise RuntimeError(msg)

            self.n_frames_total = int(exp_total_frames)
            self.n_frames = int(exp_total_frames // 2 // self.n_memcells)

    def run(self):

        totalTime = time.time()

        self.load_data()

        print("Start saving")
        print("out_fname = {}".format(self.out_fname))
        f = None
        try:
            f = h5py.File(self.out_fname, "w", libver='latest')
            f.create_dataset("analog", data=self.analog, dtype=np.int16)
            f.create_dataset("digital", data=self.digital, dtype=np.int16)

            # save metadata from original files
            idx = 0
            for set_name, set_value in iter(self.metadata.items()):
                    gname = "metadata_{}".format(idx)

                    name = "{}/source".format(gname)
                    f.create_dataset(name, data=set_name)

                    for key, value in iter(set_value.items()):
                        try:
                            name = "{}/{}".format(gname, key)
                            f.create_dataset(name, data=value)
                        except:
                            print("Error in", name, value.dtype)
                            raise
                    idx += 1
            print("Saving done")

            f.flush()
        finally:
            if f is not None:
                f.close()

        print("gather took time:", time.time() - totalTime, "\n\n")

    def load_data(self):

        print("raw_tmp_shape", self.raw_tmp_shape)
        tmp_data = np.zeros(self.raw_tmp_shape, dtype=np.int16)

        self.metadata = {}
        self.seq_number = None

        for run_idx, run_number in enumerate(self.runs):
            print("\n\nrun {}".format(run_number))

            self.pos_idxs = self.set_pos_indices(run_idx)
            print("pos_idxs", self.pos_idxs)

            load_idx_rows = slice(self.a_row_start, self.a_row_stop)
            load_idx_cols = slice(self.a_col_start, self.a_col_stop)
            print("load idx: {}, {}".format(load_idx_rows, load_idx_cols))

            self.source_seq_number = [0]
            target_offset = 0
            for i in range(self.n_parts):
                fname = self.in_fname.format(run_number=run_number, part=i)
                print("loading file {}".format(fname))

                excluded = [self.data_path]
                file_content = utils.load_file_content(fname, excluded)

                # load data
                if self.use_xfel_format:
                    self.load_xfel(fname,
                                   load_idx_rows, load_idx_cols,
                                   file_content,
                                   tmp_data,
                                   target_offset)
                else:
                    self.load_cfel(fname,
                                   load_idx_rows, load_idx_cols,
                                   file_content,
                                   tmp_data)

                self.metadata[fname] = file_content

        print("self.tmp_shape", self.tmp_shape)
        print("tmp_data.shape", tmp_data.shape)

        tmp_data.shape = self.tmp_shape
        print("tmp_data.shape", tmp_data.shape)

        self.analog = tmp_data[:, :, 0, ...]
        self.digital = tmp_data[:, :, 1, ...]

    def load_xfel(self, fname, load_idx_rows, load_idx_cols,
                  file_content, tmp_data, target_offset):
        # load data
        f = None
        try:
            f = h5py.File(fname, "r")
            raw_data = f[self.data_path][()]
        finally:
            if f is not None:
                f.close()

        print("raw_data.shape", raw_data.shape)
        print("self.raw_shape", self.raw_shape)

        n_pulses = (file_content[self.pulse_count_path].astype(np.int16))
        print("First 10 burst lengths: {} (min={}, max={})"
              .format(n_pulses[:10], np.min(n_pulses), np.max(n_pulses)))

        if self.use_interleaved:
            # currently the splitting in digital and analog does not work
            # for XFEL
            # -> all data is in the first entry of the analog/digital
            #    dimension
            raw_data = raw_data[:, 0, ...]

        raw_data = utils.convert_to_agipd_format(self.channel, raw_data)

        source_offset = 0
        print("n_bursts", n_pulses.size)

        for burst in n_pulses:
            for index_set in self.pos_idxs:
                pos_idx_rows = index_set[0]
                pos_idx_cols = index_set[1]

                source_idx = (slice(source_offset, source_offset + burst),
                              Ellipsis,
                              pos_idx_rows,
                              pos_idx_cols)
                target_idx = (slice(target_offset, target_offset + burst),
                              Ellipsis,
                              pos_idx_rows,
                              pos_idx_cols)

                try:
                    tmp_data[target_idx] = raw_data[source_idx]
                except:
                    print("burst", burst)
                    print("tmp_data.shape", tmp_data.shape)
                    print("raw_data.shape", raw_data.shape)
                    print("target_idx", target_idx)
                    print("source_idx", source_idx)
                    raise

            if burst == 0:
                print("burst == 0")

            source_offset += burst
            target_offset += self.max_pulses

    def load_cfel(self, fname, load_idx_rows, load_idx_cols,
                  file_content, tmp_data):
        # load data
        f = None
        try:
            f = h5py.File(fname, "r")
            idx = (Ellipsis, load_idx_rows, load_idx_cols)
            raw_data = f[self.data_path][idx]
        finally:
            if f is not None:
                f.close()

        print("raw_data.shape", raw_data.shape)
        print("self.raw_shape", self.raw_shape)
        self.get_seq_number(file_content[self.seq_number_path])
        self.get_frame_loss_indices()
        self.fillup_frame_loss(tmp_data,
                               raw_data,
                               self.target_index_full_size)

    def set_pos_indices(self, run_idx):
        pos_idx_rows = slice(None)
        pos_idx_cols = slice(None)

        # retuns a list of row/col indixes to give the possibility to
        # define subgroups
        # e.g. top half should use these cols and bottom half those ones
        return [[pos_idx_rows, pos_idx_cols]]

    def get_seq_number(self, source_seq_number):
        # if there is frame loss this is recognizable by a missing
        # entry in seq_number. seq_number should be loaded before
        # doing operations on it due to performance benefits
        seq_number_last_entry_previous_file = self.source_seq_number[-1]
        self.source_seq_number = source_seq_number

        print("seq_number_last_entry_previous_file={}"
              .format(seq_number_last_entry_previous_file))
        print("seq_number before modifying: {}"
              .format(self.source_seq_number))
        self.seq_number = (self.source_seq_number
                           # seq_number starts counting with 1
                           - 1
                           # the seq_number refers to the whole run
                           # not one file
                           - seq_number_last_entry_previous_file)
        print("seq_number: {}".format(self.seq_number))

    def get_frame_loss_indices(self):
        # The borders (regarding the expected shape) of
        # continuous blocks of data written into the target
        # (in between these blocks there will be zeros)
        self.target_index = [[0, 0]]
        # original sequence number starts with 1
        self.target_index_full_size = [[self.source_seq_number[0] - 1, 0]]
        # The borders (regarding the source_shape) of
        # continuous blocks of data read from the source
        # (no elements in between these blocks)
        self.source_index = [[0, 0]]
        stop = 0
        stop_full_size = 0
        stop_source = 0
        for i in np.arange(len(self.seq_number)):

            # a gap in the numbering occured
            if stop - self.seq_number[i] < -1:

                # the number before the gab gives the end of
                # the continuous block of data
                self.target_index[-1][1] = stop
                # the next block starts now
                self.target_index.append([self.seq_number[i], 0])

                # the number before the gab gives the end of
                # the continuous block of data in the fully sized array
                self.target_index_full_size[-1][1] = stop_full_size
                # the next block starts now
                # original sequence number started with 1
                seqlst = [self.source_seq_number[i] - 1, 0]
                self.target_index_full_size.append(seqlst)

                self.source_index[-1][1] = stop_source
                # the end of the block in the source
                self.source_index.append([i, 0])

            stop_source = i
            stop = self.seq_number[i]
            # original sequence number started with 1
            stop_full_size = self.source_seq_number[i] - 1

        # the last block ends with the end of the data
        self.target_index[-1][1] = self.seq_number[-1]
        self.target_index_full_size[-1][1] = self.source_seq_number[-1] - 1
        self.source_index[-1][1] = len(self.seq_number) - 1
        print("target_index {}".format(self.target_index))
        print("target_index_full_size {}".format(self.target_index_full_size))
        print("source_index {}".format(self.source_index))

        # check to see the values of the sequence number in the first frame
        # loss gap
        if len(self.target_index_full_size) > 1:
            start = self.source_index[0][1] - 1
            stop = self.source_index[1][0] + 2
            seq_num = self.source_seq_number[start:stop]
            print("seq number in first frame loss region: {}"
                  .format(seq_num))

    def fillup_frame_loss(self, raw_data, loaded_raw_data, target_index):
        # getting the blocks from source to target
        s_start = 0
        for i in range(len(target_index)):

            # start and stop of the block in the target
            # [t_start, t_stop)
            t_start = target_index[i][0]
            t_stop = target_index[i][1] + 1

            # start and stop of the block in the source
            # s_start was set in the previous loop iteration
            # (or for i=0 is set to 0)
            s_start = self.source_index[i][0]
            s_stop = self.source_index[i][1] + 1

            for index_set in self.pos_idxs:
                pos_idx_rows = index_set[0]
                pos_idx_cols = index_set[1]

                raw_idx = (slice(t_start, t_stop),
                           Ellipsis,
                           pos_idx_rows,
                           pos_idx_cols)
                loaded_idx = (slice(s_start, s_stop),
                              Ellipsis,
                              pos_idx_rows,
                              pos_idx_cols)

                raw_data[raw_idx] = loaded_raw_data[loaded_idx]

if __name__ == "__main__":
    import multiprocessing

    module_mapping = {
        "M305": "00",
    }

    use_xfel_format = True
#    use_xfel_format = False

    if use_xfel_format:
        base_path = "/gpfs/exfel/exp/SPB/201730/p900009"
        run_list = [["0428"], ["0429"], ["0430"]]

#        base_path = "/gpfs/exfel/exp/SPB/201730/p900009"
#        run_list = [["0488"]]

#        base_path = "/gpfs/exfel/exp/SPB/201701/p002012"
#        run_list = [["0007"]]

        subdir = "scratch/user/kuhnm/tmp"

#        number_of_runs = 1
#        channeld_per_run = 1
        number_of_runs = 2
        channels_per_run = 16 // number_of_runs
        for runs in run_list:
            process_list = []
            for j in range(number_of_runs):
                for i in range(channels_per_run):
                    channel = str(j * channels_per_run + i).zfill(2)
                    in_file_name = ("RAW-R{run_number}-" +
                                    "AGIPD{}".format(channel) +
                                    "-S{part:05d}.h5")
                    in_fname = os.path.join(base_path,
                                            "raw",
                                            "r{run_number}",
                                            in_file_name)

                    run_subdir = "r" + "-r".join(runs)
                    out_dir = os.path.join(base_path,
                                           subdir,
                                           run_subdir,
                                           "gather")
                    utils.create_dir(out_dir)

                    out_file_name = ("{}-AGIPD{}-gathered.h5"
                                     .format(run_subdir.upper(), channel))
                    out_fname = os.path.join(out_dir,
                                             out_file_name)

                    p = multiprocessing.Process(target=AgipdGatherBase,
                                                args=(in_fname,
                                                      out_fname,
                                                      runs,
                                                      False,  # max_part
                                                      True,  # split_asics
                                                      use_xfel_format))
                    p.start()
                    process_list.append(p)

                for p in process_list:
                    p.join()
    else:

        in_base_path = "/gpfs/cfel/fsds/labs/agipd/calibration"
        # with frame loss
#        in_subdir = "raw/317-308-215-318-313/temperature_m15C/dark"
#        module = "M317_m2"
#        runs = ["00001"]

        # no frame loss
        in_subdir = "raw/315-304-309-314-316-306-307/temperature_m25C/dark"
        module = "M304_m2"
        runs = ["00012"]
#        asic = None  # asic (None means full module)
        asic = 1

        max_part = False
        out_base_path = "/gpfs/exfel/exp/SPB/201730/p900009/scratch/user/kuhnm"
        out_subdir = "tmp"
        meas_type = "dark"
        meas_spec = {
            "dark": "tint150ns",
        }

        in_file_name = ("{}_{}_{}_"
                        .format(module,
                                meas_type,
                                meas_spec[meas_type]) +
                        "{run_number}_part{part:05d}.nxs")
        in_fname = os.path.join(in_base_path,
                                in_subdir,
                                in_file_name)

        out_dir = os.path.join(out_base_path,
                               out_subdir,
                               "gather")
        utils.create_dir(out_dir)

        if asic is None:
            out_file_name = ("{}_{}_{}.h5"
                             .format(module.split("_")[0],
                                     meas_type,
                                     meas_spec[meas_type]))
        else:
            out_file_name = ("{}_{}_{}_asic{:02d}.h5"
                             .format(module.split("_")[0],
                                     meas_type,
                                     meas_spec[meas_type],
                                     asic))
        out_fname = os.path.join(out_dir, out_file_name)

        AgipdGatherBase(in_fname,
                        out_fname,
                        runs,
                        max_part,
                        asic,
                        use_xfel_format)
