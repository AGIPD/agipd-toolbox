import h5py
import sys
import numpy as np
import time
import os


class ProcessDark():
    def __init__(self, input_fname, output_fname, use_xfel_format=False):
        if type(input_fname) == list:
            self.input_fnames = input_fname
        else:
            self.input_fnames = [input_fname]
        self.output_fname = output_fname
        self.use_xfel_format = use_xfel_format

        self.run_list = [os.path.split(fname)[1].split("-AGIPD", 1)[0].lower()
                         for fname in self.input_fnames]

        self.n_rows = 128
        self.n_cols = 512

        self.n_memcells = None
        self.offsets_shape = None
        self.thresholds_shape = None

        self.module_order = [[12, 13, 14, 15, 8, 9, 10, 11],
                             [0, 1, 2, 3, 4, 5, 6, 7]]

        self.module = int(self.input_fnames[0].rsplit("/", 1)[1].split("AGIPD")[1][:2])
        if self.module in self.module_order[1]:
            print("in wing2 (module {})".format(self.module))
            self.in_wing2 = True
        else:
            self.in_wing2 = False

        print("\n\n\nStart process_dark")
        print("input_fnames = ", self.input_fnames)
        print("output_fname = ", self.output_fname, "\n")

        self.run()

    def run(self):

        total_time = time.time()

        f = h5py.File(self.input_fnames[0], "r")
        self.n_memcells = f["analog"].shape[1]
        f.close()

        self.offsets_shape = (3, self.n_memcells, self.n_rows, self.n_cols)
        self.thresholds_shape = (2, self.n_memcells, self.n_rows, self.n_cols)

        self.means = np.empty(self.offsets_shape)
        self.stddevs = np.empty(self.offsets_shape)
        self.means_digital = np.empty(self.offsets_shape)

        self.thresholds = np.empty(self.thresholds_shape)

        for i, input_fname in enumerate(self.input_fnames):

            print("Start loading data from", input_fname)
            f = h5py.File(input_fname, "r")
            analog = f["analog"][()]
            digital = f["digital"][()]
            f.close()
            print("Loading done")

            print("Start computing means and standard deviations")
            self.means[i, ...] = np.mean(analog, axis=0)
            self.means_digital[i, ...] = np.mean(digital, axis=0)

            s = self.stddevs[i, ...]
            for cell in np.arange(self.n_memcells):
                s[cell, ...] = np.std(analog[:, cell, :, :].astype("float"), axis=0)
            print("Done computing means and standard deviations")

        md = self.means_digital
        self.thresholds[0, ...] = (md[0, ...] + md[1, ...]) // 2
        self.thresholds[1, ...] = (md[1, ...] + md[2, ...]) // 2
        #self.thresholds[0, ...] = np.mean([md[0, ...], md[1, ...]])
        #self.thresholds[1, ...] = np.mean([md[1, ...], md[2, ...]])
        #print("means_digital", self.means_digital[0, :3, :3, :3],  self.means_digital[1, :3, :3, :3])
        #print("threshold", self.thresholds[0, :3, :3, :3])

        if self.use_xfel_format:
            self.convert_to_xfel_format()

        save_file = h5py.File(self.output_fname, "w", libver="latest")
        dset_offset = save_file.create_dataset("offset",
                                               shape=self.offsets_shape,
                                               dtype=np.int16)
        dset_gainlevel_mean = save_file.create_dataset("gainlevel_mean",
                                                       shape=self.offsets_shape,
                                                       dtype=np.int16)
        dset_thresholds = save_file.create_dataset("threshold",
                                                   shape=self.thresholds_shape,
                                                   dtype=np.int16)
        dset_stddevs = save_file.create_dataset("stddev",
                                                shape=self.offsets_shape,
                                                dtype="float")

        # convert into unicode
        self.run_list = [run.encode('utf8') for run in self.run_list]
        dset_run_number = save_file.create_dataset("run_number",
                                                   data=self.run_list)

        print("Start saving results at", self.output_fname)
        dset_offset[...] = self.means
        dset_gainlevel_mean[...] = self.means_digital
        dset_thresholds[...] = self.thresholds
        dset_stddevs[...] = self.stddevs
        save_file.flush()
        print("Saving done")

        save_file.close()

        print('ProcessDark took time:  ', time.time() - total_time, '\n\n')

    def convert_to_xfel_format(self):
        s = self.thresholds_shape
        self.thresholds_shape = s[:-2] + (s[-1], s[-2])

        s = self.offsets_shape
        self.offsets_shape = s[:-2] + (s[-1], s[-2])

        self.means = np.swapaxes(self.means, 2, 3)
        self.means_digital = np.swapaxes(self.means_digital, 2, 3)
        self.thresholds = np.swapaxes(self.thresholds, 2, 3)
        self.stddevs = np.swapaxes(self.stddevs, 2, 3)

        if self.in_wing2:
            self.means = self.means[..., ::-1, :]
            self.means_digital = self.means_digital[..., ::-1, :]
            self.thresholds = self.thresholds[..., ::-1, :]
            self.stddevs = self.stddevs[..., ::-1, :]
        else:
            self.means = self.means[..., :, ::-1]
            self.means_digital = self.means_digital[..., :, ::-1]
            self.thresholds = self.thresholds[..., :, ::-1]
            self.stddevs = self.stddevs[..., :, ::-1]



if __name__ == "__main__":
    import os
    import multiprocessing
    from datetime import date

    SRC_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    print("SRC_PATH", SRC_PATH)

    if SRC_PATH not in sys.path:
        sys.path.insert(0, SRC_PATH)

    from utils import  create_dir

    base_path = "/gpfs/exfel/exp/SPB/201730/p900009/scratch/kuhnm"
    run_list = ["r0428", "r0429", "r0430"]
    #use_xfel_format = False
    use_xfel_format = True

    today = str(date.today())

    #number_of_runs = 1
    #modules_per_run = 1
    number_of_runs = 2
    modules_per_run = 16//number_of_runs
    process_list = []
    for j in range(number_of_runs):
        for i in range(modules_per_run):
            module = str(j*modules_per_run+i).zfill(2)
            print("module", module)

            input_fname_list = []
            for run_number in run_list:
                input_fname = os.path.join(base_path,
                                           run_number,
                                           "gather",
                                           "{}-AGIPD{}-gathered.h5".format(run_number.upper(), module))
                input_fname_list.append(input_fname)

            target_dir = os.path.join(base_path, "dark")
            create_dir(target_dir)

            if use_xfel_format:
                fname = "dark_AGIPD{}_xfel_{}.h5".format(module, today)
            else:
                fname = "dark_AGIPD{}_agipd_{}.h5".format(module, today)

            output_fname = os.path.join(target_dir, fname)

            p = multiprocessing.Process(target=ProcessDark, args=(input_fname_list,
                                                                  output_fname,
                                                                  use_xfel_format))
            p.start()
            process_list.append(p)

        for p in process_list:
            p.join()
