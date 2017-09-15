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

        self.n_rows = 128
        self.n_cols = 512

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
        print("n_parts", self.n_parts)
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
            self.n_memcells = f[self.pulse_count_path][0].astype(int)//2
            print("Number of memoy cells found", self.n_memcells)

            f.close()

            # xfel format has swapped rows and cols
            self.raw_shape = (self.n_memcells, 2, 2, self.n_cols, self.n_rows)

            module = int(k.split("CH")[0])
            module_order = [[12, 13, 14, 15, 8, 9, 10, 11],
                            [0, 1, 2, 3, 4, 5, 6, 7]]

            if module in module_order[1]:
                print("in wing2 (module {})".format(module))
                self.in_wing2 = True
            else:
                self.in_wing2 = False

        else:
            self.data_path = '/entry/instrument/detector/data'

            f = h5py.File(self.input_fname, "r")
            raw_data_shape = f[self.data_path].shape
            f.close()

            self.n_memcells = 1

            self.raw_shape = (self.n_memcells, 2, self.n_rows, self.n_cols)

            self.module_order = None

        self.n_frames_per_file = int(raw_data_shape[0] / 2 / self.n_memcells)
        print("n_frames_per_file", self.n_frames_per_file)
        self.n_frames = self.n_frames_per_file * self.n_parts
        print("n_frames", self.n_frames)

        self.target_shape = (self.n_frames, self.n_memcells, self.n_rows, self.n_cols)
        print("target shape:", self.target_shape)


    def load_data(self):

        self.analog = np.zeros((self.n_frames, self.n_memcells, self.n_rows, self.n_cols), dtype=np.int16)
        self.digital = np.zeros((self.n_frames, self.n_memcells, self.n_rows, self.n_cols), dtype=np.int16)

        for i in range(self.n_parts):
            fname = self.input_fname.format(i)

            f = h5py.File(fname, 'r')
            #print('start loading')
            raw_data_shape = f[self.data_path].shape
            raw_data = np.array(f[self.data_path])
            #print('loading done')
            f.close()

            self.n_frames_per_file = int(raw_data_shape[0] / 2 / self.n_memcells)

            print("raw_data.shape", raw_data.shape)
            print("self.n_frames_per_file", self.n_frames_per_file)
            print("self.raw_shape", self.raw_shape)
            raw_data.shape = (self.n_frames_per_file,) + self.raw_shape

            # currently the splitting in digital and analog does not work for XFEL
            # -> all data is in the first entry of the analog/digital dimension
            if self.use_xfel_format:
                raw_data = raw_data[:, :, :, 0, ...]

            target_idx = (slice(i * self.n_frames_per_file,
                               (i + 1) * self.n_frames_per_file),
                          Ellipsis)

            # fix geometry
            if self.use_xfel_format:
                if self.in_wing2:
                    raw_data = raw_data[..., ::-1, :]
                else:
                    raw_data = raw_data[..., :, ::-1]

                raw_data = np.rollaxis(raw_data, 4, 3)

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
    import multiprocessing

    module_mapping = {
        "M305": "00",
        }

    #temperature = "temperature_m15C"
    #module = "M305"
    #source_base_path = "/gpfs/exfel/exp/SPB/201730/p900009/raw/"
    #target_base_path = "/gpfs/cfel/fsds/labs/agipd/calibration/processed/"
    #run_type = "xray"
    #run_number = "r0377"

    temperature = "temperature_m15C"
    module = "M305"
    source_base_path = "/gpfs/exfel/exp/SPB/201701/p002012/raw/"
    target_base_path = "/gpfs/cfel/fsds/labs/agipd/calibration/processed/"
    run_type = "dark"
    run_number = "r0008"

    temperature = "temperature_m15C"
    module = "M305"
    source_base_path = "/gpfs/exfel/exp/SPB/201701/p002012/raw/"
    target_base_path = "/gpfs/cfel/fsds/labs/agipd/calibration/processed/"
    run_type = "dark"
    run_number = "r0008"

    #input_fname = "/gpfs/cfel/fsds/labs/agipd/calibration/raw/302-303-314-305/temperature_m15C/xray/M302_m3_xray_Cu_mc112_00000.nxs"
    #output_fname = "/gpfs/cfel/fsds/labs/agipd/calibration/processed/M302/temperature_m15C/xray/test.h5"
    #use_xfel_format = False

    #{}/RAW-{}-AGIPD{:02d}-S{:05d}.h5"
    input_fname = os.path.join(source_base_path,
                               run_number,
                               "RAW-{}-AGIPD{}-".format(run_number.upper(),
                                                        module_mapping[module]) + "S{:05d}.h5")
    output_fname = os.path.join(target_base_path,
                                module,
                                temperature,
                                run_type,
                                "gather",
                                "{}_{}_AGIPD{}.h5".format(module, run_type, module_mapping[module]))

    use_xfel_format = True

#    obj = Gather(input_fname, output_fname, use_xfel_format)

    process_list = []
    for i in range(16):
        module = str(i).zfill(2)
        input_fname = "/gpfs/exfel/exp/SPB/201730/p900009/raw/r0391/RAW-R0391-AGIPD{}-".format(module) + "S{:05d}.h5"
        output_fname = "/gpfs/exfel/exp/SPB/201730/p900009/scratch/kuhnm/R0391-AGIPD{}-gathered.h5".format(module)

        p = multiprocessing.Process(target=Gather, args=(input_fname, output_fname, use_xfel_format))
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()