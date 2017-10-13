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

import utils


class AgipdGatherBase():
    def __init__(self, input_file, output_file_name, max_part=False,
                 split_asics=True, use_xfel_format=False, backing_store=True):

        self.input_fname = input_fname
        self.output_fname = output_fname

        self.split_asics = split_asics
        self.max_part = max_part
        self.split_asics = split_asics
        self.use_xfel_format = use_xfel_format
        self.backing_store = backing_store

        self.use_xfel_format = use_xfel_format

        #TODO extract n_cols and n_rows from raw_shape
        self.n_rows = 128
        self.n_cols = 512
        self.n_memcells = None
        self.asic_size = 64

        # how the asics are located on the module
        self.asic_mapping = [[16, 15, 14, 13, 12, 11, 10, 9],
                             [1,   2,  3,  4,  5,  6,  7, 8]]

        self.analog = None
        self.digital = None

        self.raw_shape = None
        self.tmp_shape = None
        self.n_frames_per_file = None
        self.n_frames = None
        self.target_shape = None

        self.target_index = None
        self.target_index_full_size = None
        self.source_index = None
        self.source_seq_number = None
        self.seq_number = None

        self.get_parts()

        if self.n_parts == 0:
            msg = "No parts to gather found\n"
            msg += "input_fname={}".format(self.input_fname)
            raise Exception(msg)

        self.intiate()

        print("\n\n\n"
              "start gather\n"
              "input_fname = {}\n"
              "output_fname ={}\n"
              "data_path = {}\n"
              .format(self.input_fname,
                      self.output_fname,
                      self.data_path))

        self.run()

    def get_parts(self):
        # remove extension
        prefix = self.input_fname.rsplit(".", 1)[0]
        part_files = glob.glob("{}*".format(prefix[:-6]))

        self.n_parts = self.max_part or len(part_files)
        print("n_parts {}".format(self.n_parts))
        #self.n_parts = 2

    def intiate(self):
        if self.use_xfel_format:
            data_path_prefix = "INSTRUMENT/SPB_DET_AGIPD1M-1/DET"
            data_path_postfix = "image/data"
            pulse_count_postfix = "header/pulseCount"

            # use part number 0 to get the information from
            fname = self.input_fname.format(0)

            f = h5py.File(fname, "r")

            # xfel uses the module index inside the data path
            # automatically detect the path
            k = list(f[data_path_prefix].keys())[0]
            base_path = os.path.join(data_path_prefix, k)

            self.data_path = os.path.join(base_path, data_path_postfix)
            self.pulse_count_path = os.path.join(base_path,
                                                 pulse_count_postfix)

            raw_data_shape = f[self.data_path].shape
            self.n_memcells = f[self.pulse_count_path][0].astype(int)//2
            print("Number of memoy cells found", self.n_memcells)

            f.close()

            # xfel format has swapped rows and cols
            self.raw_shape = (self.n_memcells, 2, 2, self.n_cols, self.n_rows)
            # tmp data is already converted into agipd format
            # n_frames has to filled in once it is known
            self.raw_tmp_shape = (1, self.n_memcells, 2, self.n_cols, self.n_rows)
            self.tmp_shape = (self.n_memcells, 2, self.n_rows, self.n_cols)

            module = int(k.split("CH")[0])
            self.in_wing2 = utils.located_in_wing2(module)

        else:
            filename_template = "{prefix}_col{column}"

            self.data_path = "entry/instrument/detector/data"
            self.digital_data_path = "entry/instrument/detector/data_digital"

            self.collection_path = "entry/instrument/detector/collection"
            self.seq_number_path = "entry/instrument/detector/sequence_number"
            self.total_lost_frames_path = ("{}/total_loss_frames"
                                           .format(self.collection_path))
            self.error_code_path = "{}/error_code".format(self.collection_path)
            self.frame_number_path = ("{}/frame_numbers"
                                      .format(self.collection_path))

            fname = self.input_fname.format(0)

            f = h5py.File(fname, "r")
            raw_data_shape = f[self.data_path].shape
            f.close()

            # dark
            self.n_memcells = 352
            # xray
            #self.n_memcells = 1

            self.raw_shape = (self.n_memcells, 2, self.n_rows, self.n_cols)
            # n_frames has to filled in once it is known
            self.raw_tmp_shape = (1 * self.n_memcells * 2, self.n_rows, self.n_cols)
            self.tmp_shape = self.raw_shape

        self.n_frames_per_file = int(raw_data_shape[0] // 2 // self.n_memcells)
        print("n_frames_per_file", self.n_frames_per_file)
        self.get_number_of_frames()
        print("n_frames", self.n_frames)

        self.raw_tmp_shape = (self.raw_tmp_shape[0] * self.n_frames,) + self.raw_tmp_shape[1:]

        self.target_shape = (self.n_frames, self.n_memcells, self.n_rows, self.n_cols)
        print("target shape:", self.target_shape)

    def get_number_of_frames(self):
        f = None

        if self.use_xfel_format:
            self.n_frames = 0
            for i in range(self.n_parts):
                fname = self.input_fname.format(i)

                try:
                    f = h5py.File(fname, "r")
                    raw_data_shape = f[self.data_path].shape
                finally:
                    if f is not None:
                        f.close()

                n_frames_per_file = int(raw_data_shape[0] // 2 // self.n_memcells)
                self.n_frames = self.n_frames + n_frames_per_file

        else:
            try:
                fname = self.input_fname.format(0)
                f = h5py.File(fname, "r", libver="latest", drivers="core")

                # TODO: verify that the shape is always right and not dependant on
                #       frame loss
                source_shape = f[self.data_path].shape
                exp_total_frames = f[self.frame_number_path][0]

            except:
                print("Error when getting shape")
                raise
            finally:
                if f is not None:
                    f.close()

            if (source_shape[1] != self.n_rows and
                    source_shape[2] != self.n_cols):
                msg = "Shape of file {} ".format(fname)
                msg += "does not match requirements\n"
                msg += "source_shape = {}".format(source_shape)
                raise RuntimeError(msg)

            self.n_frames = int(exp_total_frames // 2 // self.n_memcells)

        print("frames={}".format(self.n_frames))

#    def get_charges(self):
#        source_file = None
#        try:
#            fname = self.files[0][0]
#            source_file = h5py.File(fname, "r", libver="latest",
#                                    drivers="core")
#
#            # TODO: verify that the shape is always right and not dependant on
#            #       frame loss
#            source_shape = source_file[self.data_path].shape
#
#            self.exp_total_frames = source_file[self.frame_number_path][0]
#
#            # if there is frame loss this is recognizable by a missing entry
#            # in seq_number
#            self.seq_number = source_file[self.seq_number_path][()]
#            print("len seq_number: {}".format(len(self.seq_number)))
#
#            # total_lost_frames are totally missing frames as well as
#            # frames where only a package was lost
#            total_lost_frames = source_file[self.total_lost_frames_path][0]
#
#            # frames where a package is lost occure in error_code with
#            # an nonzero entry whereas complety lost frames do not have
#            # an entry in error code at all
#            error_code = source_file[self.error_code_path][()]
#            frames_with_pkg_loss = error_code.nonzero()
#            n_frames_with_pkg_loss = len(np.hstack(frames_with_pkg_loss))
#            print("Frames with packet loss: {}".format(n_frames_with_pkg_loss))
#        except:
#            print("Error when getting shape")
#            raise
#        finally:
#            if source_file is not None:
#                source_file.close()
#
#        if (source_shape[1] != self.n_rows and
#                source_shape[2] != self.n_cols):
#            msg = "Shape of file {} does not match requirements\n".format(fname)
#            msg += "source_shape = {}".format(source_shape)
#            raise RuntimeError(msg)
#
#        # due to a "bug" in Tango the total_lost_frames can only be found in
#        # the next file
#        # TODO once Tango has fixed this, adjust it here as well
#        source_file = None
#        try:
#            fname = self.files[0][1]
#            source_file = h5py.File(fname, "r", libver="latest",
#                                    driver="core")
#            # total_lost_frames are totally missing frames as well as
#            # frames where only a package was lost
#            total_lost_frames = source_file[self.total_lost_frames_path][0]
#        except Exception as e:
#            raise RuntimeError("Error when getting shape: {}".format(e))
#        finally:
#            if source_file is not None:
#                source_file.close()
#
#        self.exp_n_frames_per_file = (len(self.seq_number)
#                                          # for the very first file
#                                          # total_lost_frames was not yet
#                                          # aggregated
#                                          + total_lost_frames
#                                          - n_frames_with_pkg_loss)
#
#        self.n_frames = int(self.exp_total_frames / 2 / self.n_memcells)
#
#        print("exp_total_frames={}".format(self.exp_total_frames))
#        print("exp_n_frames_per_file={}".format(self.exp_n_frames_per_file))
#        print("frames={}".format(self.n_frames))

    def run(self):

        totalTime = time.time()

        self.load_data()

        save_file = h5py.File(self.output_fname, "w", libver='latest')
        dset_analog = save_file.create_dataset("analog",
                                               shape=self.target_shape,
                                               dtype=np.int16)
        dset_digital = save_file.create_dataset("digital",
                                                shape=self.target_shape,
                                                dtype=np.int16)

        print("Start saving")
        dset_analog[...] = self.analog
        dset_digital[...] = self.digital

        idx = 0
        for set_name, set_value in iter(self.metadata.items()):
                gname = "metadata_{}".format(idx)

                name = "{}/source".format(gname)
                dset_metadata = save_file.create_dataset(name, data=set_name)

                for key, value in iter(set_value.items()):
                    try:
                        name = "{}/{}".format(gname, key)
                        dset_metadata = save_file.create_dataset(name, data=value)
                    except:
                        print("Error in", name, value.dtype)
                        raise
                idx += 1
        print("Saving done")

        save_file.flush()
        save_file.close()

        print("gather took time:", time.time() - totalTime, "\n\n")

    def load_data(self):

        print("raw_tmp_shape", self.raw_tmp_shape)
        tmp_data = np.zeros(self.raw_tmp_shape, dtype=np.int16)

        self.metadata = {}
        self.source_seq_number = [0]
        self.seq_number = None

        idx_offset = 0

        for i in range(self.n_parts):
            fname = self.input_fname.format(i)

            file_content = utils.load_file_content(fname)

            raw_data_shape = file_content[self.data_path].shape
            raw_data = file_content[self.data_path]

            self.n_frames_per_file = int(raw_data_shape[0] // 2 // self.n_memcells)

            print("raw_data.shape", raw_data.shape)
            print("self.n_frames_per_file", self.n_frames_per_file)
            print("self.raw_shape", self.raw_shape)

            # currently the splitting in digital and analog does not work for XFEL
            # -> all data is in the first entry of the analog/digital dimension
            if self.use_xfel_format:
                raw_data.shape = (self.n_frames_per_file,) + self.raw_shape

                raw_data = raw_data[:, :, :, 0, ...]

                target_idx = (slice(idx_offset,
                                    idx_offset + self.n_frames_per_file),
                              Ellipsis)
                idx_offset += self.n_frames_per_file

                tmp_data[target_idx] = raw_data[...]

            else:
                self.get_seq_number(file_content[self.seq_number_path])
                self.get_frame_loss_indices()
                self.fillup_frame_loss(tmp_data,
                                       raw_data,
                                       self.target_index_full_size)

            del file_content[self.data_path]
            self.metadata[fname] = file_content

        if self.use_xfel_format:
            [tmp_data], _ = utils.convert_to_agipd_format(module,
                                                          [tmp_data])
        else:
            tmp_data.shape = (self.n_frames,) + self.tmp_shape


        self.analog = tmp_data[:, :, 0, ...]
        self.digital = tmp_data[:, :, 1, ...]

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

        # check to see the values of the sequence number in the first frame loss gap
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

            raw_data[t_start:t_stop, ...] = (
                loaded_raw_data[s_start:s_stop, ...])

if __name__ == "__main__":
    import multiprocessing

    module_mapping = {
        "M305": "00",
        }

    #use_xfel_format = True
    use_xfel_format = False

    if use_xfel_format:
        base_path = "/gpfs/exfel/exp/SPB/201701/p002012"
        run_list = ["r0007"]
        subdir = "scratch/user/kuhnm"

        number_of_runs = 1
        modules_per_run = 1
        #number_of_runs = 2
        #modules_per_run = 16//number_of_runs
        for run_number in run_list:
            process_list = []
            for j in range(number_of_runs):
                for i in range(modules_per_run):
                    module = str(j*modules_per_run+i).zfill(2)
                    input_fname = os.path.join(
                        base_path,
                        "raw",
                        run_number,
                        "RAW-{}-AGIPD{}-".format(run_number.upper(),
                                                 module) + "S{:05d}.h5")

                    output_dir = os.path.join(base_path,
                                              subdir,
                                              run_number,
                                              "gather")
                    utils.create_dir(output_dir)
                    output_fname = os.path.join(
                        output_dir,
                        "{}-AGIPD{}-gathered.h5".format(run_number.upper(),
                                                        module))

                    p = multiprocessing.Process(target=AgipdGatherBase,
                                                args=(input_fname,
                                                      output_fname,
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
        in_subdir = "raw/317-308-215-318-313/temperature_m15C/dark"
        module = "M317_m2"
        run_number = "00001"
        max_part = False

         # no frame loss
#        in_subdir = "raw/315-304-309-314-316-306-307/temperature_m25C/dark"
#        module = "M304_m2"
#        run_number = "00012"
#        max_part = False

        out_base_path = "/gpfs/exfel/exp/SPB/201701/p002012/scratch/user/kuhnm"
        out_subdir = "tmp"
        meas_type = "dark"
        meas_spec = {"dark": "tint150ns"}

        input_file_name = ("{}_{}_{}_{}_"
                           .format(module,
                                   meas_type,
                                   meas_spec[meas_type],
                                   run_number)
                           + "part{:05d}.nxs")
        input_fname = os.path.join(in_base_path,
                                   in_subdir,
                                   input_file_name)

        output_dir = os.path.join(out_base_path,
                                  out_subdir,
                                  "gather")
        utils.create_dir(output_dir)
        output_file_name = ("{}_{}_{}.h5"
                            .format(module.split("_")[0],
                                    meas_type,
                                    meas_spec[meas_type]))
        output_fname = os.path.join(output_dir, output_file_name)

        obj = AgipdGatherBase(input_fname,
                              output_fname,
                              max_part,
                              True,  # split_asics
                              use_xfel_format)
