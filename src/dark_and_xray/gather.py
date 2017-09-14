import h5py
import numpy as np
import os
import sys
import time
import glob

class Gather():
    def __init__(self, input_fname, output_fname, use_xfel_format=False):

        self.input_fname = input_fname
        self.output_fname = output_fname

        self.use_xfel_format = use_xfel_format

        self.analog = None
        self.digital = None

        self.get_parts()

        self.intiate()

        print('\n\n\nstart gatherXRayTubeData')
        print('input_fname = ', self.input_fname)
        print('output_fname = ', self.output_fname)
        print('data_path = ', self.data_path)
        print(' ')

        self.run()

    def get_parts(self):

        part_files = glob.glob("{}*".format(self.input_fname[:-9]))

        self.n_parts = len(part_files)
        #self.n_parts = 2

    def intiate(self):
        if self.use_xfel_format:
            data_path_prefix = "/INSTRUMENT/SPB_DET_AGIPD1M-1/DET"
            data_path_postfix = "image/data"
            pulse_count_postfix = "header/pulseCount"

            fname = self.input_fname.format(0)

            f = h5py.File(fname, "r")

            k = list(f[data_path_prefix].keys())[0]

            self.base_path = os.path.join(data_path_prefix, k)

            self.data_path = os.path.join(self.base_path, data_path_postfix)
            raw_data_shape = f[self.data_path].shape

            self.pulse_count_path = os.path.join(self.base_path, pulse_count_postfix)
            self.n_memcells = f[self.pulse_count_path][0]
            print("Number of memoy cells found", self.n_memcells)

            f.close()

            self.raw_shape = (self.n_memcells, 2, 2, 128, 512)

        else:
            self.data_path = '/entry/instrument/detector/data'

            f = h5py.File(self.input_fname, "r")
            raw_data_shape = f[self.data_path].shape
            f.close()

            self.n_memcells = 1
            self.raw_shape = (self.n_memcells, 2, 128, 512)

        self.n_frames_per_file = int(raw_data_shape[0] / 2 / self.n_memcells)
        print("n_frames_per_file", self.n_frames_per_file)
        self.n_frames = self.n_frames_per_file * self.n_parts

        self.target_shape = (self.n_frames, self.n_memcells, 128, 512)


    def load_data(self):

        self.analog = np.zeros((self.n_frames, self.n_memcells, 128, 512), dtype=np.int16)
        self.digital = np.zeros((self.n_frames, self.n_memcells, 128, 512), dtype=np.int16)

        for i in range(self.n_parts):
            fname = self.input_fname.format(i)

            f = h5py.File(fname, 'r')
            print('start loading')
            raw_data_shape = f[self.data_path].shape
            raw_data = np.array(f[self.data_path])
            print('loading done')
            f.close()

            self.n_frames_per_file = int(raw_data_shape[0] / 2 / self.n_memcells)

            print("raw_data.shape", raw_data.shape)
            print("self.n_frames_per_file", self.n_frames_per_file)
            print("res", self.n_frames_per_file * self.raw_shape[0] * 2)
            print("self.raw_shape", self.raw_shape)
            raw_data.shape = (self.n_frames_per_file,) + self.raw_shape

            # currently the splitting in digital and analog does not work for XFEL
            # -> all data is in the first entry of the analog/digital dimension
            if self.use_xfel_format:
                raw_data = raw_data[:, :, :, 0, ...]

            target_idx = (slice(i * self.n_frames_per_file,
                               (i + 1) * self.n_frames_per_file),
                          Ellipsis)

            self.analog[target_idx] = raw_data[:, :, 0, ...]
            self.digital[target_idx] = raw_data[:, :, 1, ...]


    def run(self):

        totalTime = time.time()

        self.load_data()

        saveFile = h5py.File(self.output_fname, "w", libver='latest')
        dset_analog = saveFile.create_dataset("analog",
                                              shape=self.target_shape,
                                              dtype=np.int16)
        dset_digital = saveFile.create_dataset("digital",
                                               shape=self.target_shape,
                                               dtype=np.int16)

        print('start saving')
        dset_analog[...] = self.analog
        dset_digital[...] = self.digital
        print('saving done')

        saveFile.flush()
        saveFile.close()

        print('gatherXRayTubeData took time:  ', time.time() - totalTime, '\n\n')

if __name__ == "__main__":
    #input_fname = "/gpfs/cfel/fsds/labs/agipd/calibration/raw/302-303-314-305/temperature_m15C/xray/M302_m3_xray_Cu_mc112_00000.nxs"
    #output_fname = "/gpfs/cfel/fsds/labs/agipd/calibration/processed/M302/temperature_m15C/xray/test.h5"
    #use_xfel_format = False

    #{}/RAW-{}-AGIPD{:02d}-S{:05d}.h5"
    input_fname = "/gpfs/exfel/exp/SPB/201730/p900009/raw/r0377/RAW-R0377-AGIPD00-S{:05d}.h5"
    output_fname = "/gpfs/cfel/fsds/labs/agipd/calibration/processed/M302/temperature_m15C/xray/test_AGIPD00_s00000.h5"
    use_xfel_format = True

    obj = Gather(input_fname, output_fname, use_xfel_format)
