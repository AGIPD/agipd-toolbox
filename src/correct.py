import h5py
import sys
import numpy as np
import time
import os
tmp_raw_data = {}

def get_file_content(name, obj):
    global tmp_raw_data

    if isinstance(obj, h5py.Dataset):
        tmp_raw_data[name] = obj[()]

class Correct():
    def __init__(self,ctype, channel, data_fname, dark_fname, gain_fname, base_cal_fname, output_fname,
                  photon_energy, use_xfel_format=False):
        self.ctype = ctype
        self.channel = channel
        print(type(channel))
        self.data_fname = data_fname
        self.dark_fname = dark_fname
        self.gain_fname = gain_fname
        self.base_cal_fname = base_cal_fname
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

        self.module_order = [[12, 13, 14, 15, 8, 9, 10, 11],
                             [0, 1, 2, 3, 4, 5, 6, 7]]

        self.module = self.data_fname.rsplit("/", 1)[1].split("AGIPD")[1][:2]
        if self.module in self.module_order[1]:
            print("in wing2 (module {})".format(self.module))
            self.in_wing2 = True
        else:
            self.in_wing2 = False

        print("\n\n\nStart correcting")
        print("data_fname = ", self.data_fname)
        print("dark_fname = ", self.dark_fname)
        print("gain_fname = ", self.gain_fname)
        print("base_cal_fname = ", self.base_cal_fname)
        print("output_fname = ", self.output_fname, "\n")

        self.get_dims()

        self.run()


    def get_dims(self):
        fname = self.data_fname

        f = h5py.File(fname, "r")

        k = list(f[self.data_path_prefix].keys())[0]
        self.base_path = os.path.join(self.data_path_prefix, k)

        self.data_path = os.path.join(self.base_path, self.data_path_postfix)
        raw_data_shape = f[self.data_path].shape

        self.pulse_count_path = os.path.join(self.base_path, self.pulse_count_postfix)
        self.n_memcells = f[self.pulse_count_path][0].astype(int)//2

        f.close()

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
        f = h5py.File(self.dark_fname, "r")
        self.offset = f["/offset"][()]
        self.threshold = f["/threshold"][()]
        #print("Offset shape", self.offset.shape)
        f.close()
        print("Loading darks done")

        if self.ctype == "ff":
            print("Start loading data form", self.base_cal_fname)
            f = h5py.File(self.base_cal_fname, "r")
            qm = "Q{}M{}".format(self.channel // 4 + 1, self.channel % 4 + 1)
            self.rel_gain = f["{}/RelativeGain/0/data".format(qm)][()]
            self.offset_ci = f["{}/BaseOffset/0/data".format(qm)][()]
            f.close()
            self.rel_gain = np.rollaxis(self.rel_gain,3,0)
            self.rel_gain = np.rollaxis(self.rel_gain, 3, 1)
            self.rel_gain = np.rollaxis(self.rel_gain, 3, 2)
            self.offset_ci = np.rollaxis(self.offset_ci, 3)
            self.offset_ci = np.rollaxis(self.offset_ci, 3, 1)
            self.offset_ci = np.rollaxis(self.offset_ci, 3, 2)
            #print("Relative gain shape: ", self.rel_gain.shape)
            print("Loading agipd_base_cal done")
        print("Start loading data from", self.data_fname)
        self.load_data()
        print("Loading data done")

        print("Start correcting")
        self.correct_data()
        print("Done correcting")

        #if self.use_xfel_format:
        #    self.convert_to_xfel_format()

        self.write_data()

        print('correction took time:  ', time.time() - total_time, '\n\n')

    def load_data(self):
        global tmp_raw_data

        f = h5py.File(self.data_fname, "r")
        f.visititems(get_file_content)
        f.close()

        raw_data = tmp_raw_data[self.data_path]

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
        if self.ctype == "ff":
            #Steffen's offset
#            print("1: ", self.offset_ci.shape)

#            print("2: ", self.offset_ci.shape)
            offset_median = np.median(self.offset_ci - self.offset, axis=(-1, -2))
            #print("1: ", offset_median.shape)
            self.offset = np.swapaxes(self.offset, 1, -1)
            self.offset = np.swapaxes(self.offset, 0, -2)
            #print("1: ", offset_median.shape)
            moffset = self.offset - offset_median
            moffset = np.swapaxes(moffset, 1, -1)
            moffset = np.swapaxes(moffset, 0, -2)
            self.offset = np.swapaxes(self.offset, 1, -1)
            self.offset = np.swapaxes(self.offset, 0, -2)




        for i in range(self.n_frames):
            self.compute_gain_stage(i)
            if self.ctype == "agipd":
                offset = np.choose(self.gain_stage, (self.offset[0, ...],
                                                     self.offset[1, ...],
                                                     self.offset[2, ...]))
                self.analog_corrected[i] = self.analog[i].astype(np.int32) - offset
            elif self.ctype =="ff":
                offset = np.choose(self.gain_stage, (moffset[0, ...],
                                                     moffset[1, ...],
                                                     moffset[2, ...]))
                rel_gain = np.choose(self.gain_stage,(self.rel_gain[0, ...],
                                                      self.rel_gain[1, ...],
                                                      self.rel_gain[2, ...]))

                self.analog_corrected[i] = (self.analog[i].astype(np.int32) - offset) / rel_gain


        #print("Verification:")
        #print(self.analog.shape)
        #print(offset.shape)
        #print(self.analog_corrected.shape)
        #idx = (1, 2, 1)
        #print("analog", self.analog[(0,) + idx])
        #print("offest", offset[idx])
        #print("corrected", self.analog_corrected[(0,)+ idx])

    def convert_to_xfel_format(self):
        if self.in_wing2:
            self.analog = self.analog[..., ::-1, :]
            self.digital = self.digital[..., ::-1, :]
        else:
            self.analog = self.analog[..., :, ::-1]
            self.digital = self.digital[..., :, ::-1]

        self.analog = np.swapaxes(self.analog, 3, 2)
        self.digital = np.swapaxes(self.digital, 3, 2)

        s = self.thresholds_shape
        self.thresholds_shape = s[:-2] + (s[-1], s[-2])

        s = self.offsets_shape
        self.offsets_shape = s[:-2] + (s[-1], s[-2])

    def write_data(self):
        global tmp_raw_data

        print("Start saving results at", self.output_fname)

        self.analog_corrected.shape = self.output_data_shape
        self.digital.shape = self.output_data_shape

        save_file = h5py.File(self.output_fname, "w", libver="latest")

        for key in tmp_raw_data:
            if key != self.data_path:
                save_file.create_dataset(key, data=tmp_raw_data[key])

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
    from datetime import date
    import glob

    SRC_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    print("SRC_PATH", SRC_PATH)

    if SRC_PATH not in sys.path:
        sys.path.insert(0, SRC_PATH)

    from utils import  create_dir

    data_dir = "/gpfs/exfel/exp/SPB/201701/p002012/raw"
    dark_dir = "/gpfs/exfel/exp/SPB/201701/p002012/scratch/user/kuhnm/dark/r0037-r0038-r0039"
    gain_dir = "/gpfs/exfel/exp/SPB/201701/p002012/scratch/user/kuhnm/gain"
    output_dir = "/gpfs/exfel/exp/SPB/201701/p002012/scratch/user/kuhnm"
    #run_list = ["r0068"]
    run_number = "r0068"
    use_xfel_format = True
    #use_xfel_format = False
    photon_energy = 1

    number_of_runs = 1
    modules_per_run = 1
    #number_of_runs = 2
    #modules_per_run = 16//number_of_runs
    process_list = []
    offset = 3
    for j in range(number_of_runs):
        for i in range(modules_per_run):
            module = str(offset + j*modules_per_run+i).zfill(2)
            print("module", module)
            part = 0

            fname = "RAW-{}-AGIPD{}-S{:05d}.h5".format(run_number.upper(), module, part)
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
                #print("No gain constants found. Quitting.")
                #sys.exit(1)

            output_dir = os.path.join(output_dir, run_number)
            create_dir(output_dir)

            fname = "corrected_AGIPD{}-S{:05d}.h5".format(module, part)
            output_fname = os.path.join(output_dir, fname)

            p = multiprocessing.Process(target=Correct, args=(data_fname,
                                                              dark_fname,
                                                              None,
                                                              #gain_fname,
                                                              output_fname,
                                                              photon_energy,
                                                              use_xfel_format))
            p.start()
            process_list.append(p)

        for p in process_list:
            p.join()
