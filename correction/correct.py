# (c) Copyright 2017-2018 DESY, FS-DS
#
# This file is part of the FS-DS AGIPD toolbox.
#
# This software is free: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this software.  If not, see <http://www.gnu.org/licenses/>.

"""
@author: Manuela Kuhn <manuela.kuhn@desy.de>
         Jennifer Poehlsen <jennifer.poehlsen@desy.de>
"""

import h5py
import sys
import numpy as np
import time
import os

SRC_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print("SRC_PATH", SRC_PATH)

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import utils  # noqa E402


class Correct(object):
    def __init__(self,
                 data_fname,
                 dark_fname,
                 gain_fname,
                 output_fname,
                 photon_energy,
                 use_xfel_format=False):

        self.data_fname = data_fname
        self.dark_fname = dark_fname
        self.gain_fname = gain_fname
        self.output_fname = output_fname
        self.use_xfel_format = use_xfel_format
        self.use_xfel_input_format = True

        self.data_path_prefix = "INSTRUMENT/SPB_DET_AGIPD1M-1/DET"
        self.data_path_postfix = "image/data"
        self.gain_path_postfix = "image/gain"
        self.pulse_count_postfix = "header/pulseCount"

        self.n_rows = None
        self.n_cols = None
        self.n_memcells = None
        self.output_data_shape = None

        self.channel = self.data_fname.rsplit("/", 1)[1].split("AGIPD")[1][:2]
        self.in_wing2 = utils.located_in_wing2(self.channel)

        print("\n\n\nStart correcting")
        print("data_fname = ", self.data_fname)
        print("dark_fname = ", self.dark_fname)
        print("gain_fname = ", self.gain_fname)
        print("output_fname = ", self.output_fname, "\n")

        self.get_dims()

        self.run()

    def get_dims(self):
        fname = self.data_fname

        with h5py.File(fname, "r") as f:
            k = list(f[self.data_path_prefix].keys())[0]
            self.base_path = os.path.join(self.data_path_prefix, k)

            self.data_path = os.path.join(self.base_path,
                                          self.data_path_postfix)
            raw_data_shape = f[self.data_path].shape

            self.pulse_count_path = os.path.join(self.base_path,
                                                 self.pulse_count_postfix)
            self.n_memcells = f[self.pulse_count_path][0].astype(int) // 2

        self.gain_path = os.path.join(self.base_path, self.gain_path_postfix)

        self.n_rows = raw_data_shape[-2]
        self.n_cols = raw_data_shape[-1]
        # xfel format has swapped rows and cols
        if self.use_xfel_input_format:
            self.raw_shape = (self.n_memcells, 2, 2, self.n_rows, self.n_cols)

        self.output_data_shape = (-1, self.n_rows, self.n_cols)

        print("n_memcells:", self.n_memcells)
        print("n_rows", self.n_rows)
        print("n_cols:", self.n_cols)

    def run(self):

        total_time = time.time()

        print("Start loading dark from", self.dark_fname)
        with h5py.File(self.dark_fname, "r") as f:
            self.offset = f["/offset"][()]
            self.threshold = f["/threshold"][()]
        print("Loading done")

        print("Start loading data from", self.data_fname)
        self.load_data()
        print("Loading done")

        print("Start correcting")
        self.correct_data()
        print("Done correcting")

#        if self.use_xfel_format:
#            utils.convert_to_xfel_format(self.channel, self.analog)
#            utils.convert_to_xfel_format(self.channel, self.digital)
#            utils.convert_to_xfel_format(self.channel, self.thresholds_shape)
#            utils.convert_to_xfel_format(self.channel, self.offsets_shape)

        self.write_data()

        print('correction took time:  ', time.time() - total_time, '\n\n')

    def load_data(self):

        self.raw_data_content = utils.load_file_content()

        raw_data = self.raw_data_content[self.data_path]

        self.n_frames = int(raw_data.shape[0] / 2 / self.n_memcells)
        raw_data.shape = (self.n_frames,) + self.raw_shape

        # currently the splitting in digital and analog does not work for XFEL
        # -> all data is in the first entry of the analog/digital dimension
        if self.use_xfel_input_format:
            raw_data = raw_data[:, :, :, 0, ...]

        self.analog = raw_data[:, :, 0, ...]
        self.digital = raw_data[:, :, 1, ...]

    def compute_gain_stage(self, i):
        self.gain_stage = np.zeros((self.n_memcells, self.n_rows, self.n_cols),
                                   dtype=np.uint8)

        self.gain_stage[self.digital[i] > self.threshold[0, ...]] = 1
        self.gain_stage[self.digital[i] > self.threshold[1, ...]] = 2

    def correct_data(self):
        self.analog_corrected = np.empty(self.analog.shape, np.float)

        for i in range(self.n_frames):
            self.compute_gain_stage(i)

            offset = np.choose(self.gain_stage, (self.offset[0, ...],
                                                 self.offset[1, ...],
                                                 self.offset[2, ...]))
            self.analog_corrected[i] = self.analog[i].astype(np.int32) - offset

#        print("Verification:")
#        print(self.analog.shape)
#        print(offset.shape)
#        print(self.analog_corrected.shape)
#        idx = (1, 2, 1)
#        print("analog", self.analog[(0,) + idx])
#        print("offest", offset[idx])
#        print("corrected", self.analog_corrected[(0,)+ idx])

    def write_data(self):
        print("Start saving results at", self.output_fname)

        self.analog_corrected.shape = self.output_data_shape
        self.digital.shape = self.output_data_shape

        save_file = h5py.File(self.output_fname, "w", libver="latest")

        for key in self.raw_data_content:
            if key != self.data_path:
                save_file.create_dataset(key, data=self.raw_data_content[key])

        save_file.create_dataset(self.data_path,
                                 data=self.analog_corrected)
        save_file.create_dataset(self.gain_path,
                                 data=self.digital)

        if self.dark_fname is None:
            self.dark_fname = "None"
        if self.gain_fname is None:
            self.gain_fname = "None"
        correction_params = [self.dark_fname.encode('utf8'),
                             self.gain_fname.encode('utf8')]
        save_file.create_dataset("correction",
                                 data=correction_params)

        save_file.flush()

        save_file.close()
        print("Saving done")


if __name__ == "__main__":
    import multiprocessing
    import glob

    base_dir = "/gpfs/exfel/exp/SPB/201701/p002012/"
    data_dir = os.path.join(base_dir, "raw")
    dark_dir = os.path.join(base_dir,
                            "scratch/user/kuhnm/dark/r0037-r0038-r0039")
    gain_dir = os.path.join(base_dir, "scratch/user/kuhnm/gain")
    output_dir = os.path.join(base_dir, "scratch/user/kuhnm")

#    run_list = ["r0068"]
    run_number = "r0068"

    use_xfel_format = True
#    use_xfel_format = False
    photon_energy = 1

    number_of_runs = 1
    modules_per_run = 1
#    number_of_runs = 2
#    modules_per_run = 16 // number_of_runs
    process_list = []
    offset = 3
    for j in range(number_of_runs):
        for i in range(modules_per_run):
            module = str(offset + j * modules_per_run + i).zfill(2)
            print("module", module)
            part = 0

            fname = ("RAW-{}-AGIPD{}-S{:05d}.h5"
                     .format(run_number.upper(), module, part))
            data_fname = os.path.join(data_dir,
                                      run_number,
                                      fname)

            if use_xfel_format:
                fname_prefix = "dark_AGIPD{}_xfel".format(module)
            else:
                fname_prefix = "dark_AGIPD{}_agipd_".format(module)
            dark_fname_prefix = os.path.join(dark_dir, fname_prefix)
            dark_fname = glob.glob("{}*".format(dark_fname_prefix))
            if dark_fname:
                dark_fname = dark_fname[0]
            else:
                print("No dark constants found. Quitting.")
                sys.exit(1)
            print(dark_fname)

            if use_xfel_format:
                fname_prefix = "gain_AGIPD{}_xfel".format(module)
            else:
                fname_prefix = "gain_AGIPD{}_agipd_".format(module)
            gain_fname_prefix = os.path.join(gain_dir, fname_prefix)
            gain_fname = glob.glob("{}*".format(gain_fname_prefix))
            if gain_fname:
                gain_fname = gain_fname[0]
            else:
                print("No gain constants found.")
#                print("No gain constants found. Quitting.")
#                sys.exit(1)

            output_dir = os.path.join(output_dir, run_number)
            utils.create_dir(output_dir)

            fname = "corrected_AGIPD{}-S{:05d}.h5".format(module, part)
            output_fname = os.path.join(output_dir, fname)

            p = multiprocessing.Process(target=Correct, args=(data_fname,
                                                              dark_fname,
                                                              None,
                                                              # gain_fname,
                                                              output_fname,
                                                              photon_energy,
                                                              use_xfel_format))
            p.start()
            process_list.append(p)

        for p in process_list:
            p.join()
