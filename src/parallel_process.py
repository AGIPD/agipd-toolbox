from __future__ import print_function

from multiprocessing import Pool, TimeoutError
import numpy as np
import os
import time
import h5py
from process import ProcessDrscs, initiate_result, check_file_exists


def exec_process(asic, input_file, analog, digital, pixel_v_list, pixel_u_list, mem_cell_list):

    #plot_dir = "/gpfs/cfel/fsds/labs/agipd/calibration/processed/M314/temperature_m15C/drscs/plots/itestc150/manu_test/asic01_failed/"
    #plot_prefix = "M314_itestc150"
    #create_error_plots=True
    create_error_plots=False

    #cal = ProcessDrscs(asic, analog=analog, digital=digital)
    cal = ProcessDrscs(asic, input_file)#,
    #                   plot_prefix=plot_prefix, plot_dir=plot_dir, create_plots=False)
    cal.run(pixel_v_list, pixel_u_list, mem_cell_list, create_error_plots=create_error_plots)

    return cal.result, pixel_v_list, pixel_u_list, mem_cell_list


class ParallelProcess():
    def __init__(self, asic, input_fname, pixel_v_list, pixel_u_list,
                 mem_cell_list, n_processes, output_fname):
        self.asic = asic
        self.input_fname = input_fname

        self.pixel_v_list = pixel_v_list
        self.pixel_u_list = pixel_u_list
        self.mem_cell_list = mem_cell_list
        self.n_gain_stages = 3

        self.n_processes = n_processes
        self.process_lists = []

        self.output_fname = output_fname
        self.result = None

        self.digital_path = "/entry/instrument/detector/data_digital"
        self.analog_path = "/entry/instrument/detector/data"

        self.analog = None
        self.digital = None

        check_file_exists(self.output_fname)

        self.generate_process_lists()

        #self.load_data()

        self.run()

    def generate_process_lists(self):

        if len(self.pixel_v_list) <= self.n_processes:
            self.process_lists = [np.array([i]) for i in self.pixel_v_list]
        else:
            size = int(len(self.pixel_v_list) / self.n_processes)
            rest = len(self.pixel_v_list) % self.n_processes

            # distribute the workload
            self.process_lists = [self.pixel_v_list[i:i+size]
                for i in range(0, len(self.pixel_v_list) - size*(rest+1), size)]

            # if list to split is not a multiple of size, the rest is equaly
            # distributed over the remaining processes
            self.process_lists += [self.pixel_v_list[i:i+size+1]
                for i in range(len(self.process_lists)*size, len(self.pixel_v_list), size+1)]

        print("process_lists")
        for i in self.process_lists:
            print(i)

    def initiate_result(self, p_result):
        n_gain_stages = p_result["slope"]["mean"].shape[0]
        n_intervals = {
            "high": p_result["slope"]["individual"]["high"].shape[-1],
            "medium": p_result["slope"]["individual"]["medium"].shape[-1],
            "low": p_result["slope"]["individual"]["low"].shape[-1]
        }
        n_diff_changes_stored = p_result["collection"]["n_diff_changes_stored"]

        #print("n_gain_stages", n_gain_stages)
        #print("n_intervals", n_intervals)
        #print("n_bins", n_bins)

        self.result = initiate_result(self.pixel_v_list, self.pixel_u_list,
                                      self.mem_cell_list, n_gain_stages,
                                      n_intervals, n_diff_changes_stored)

    def load_data(self):

        source_file = h5py.File(self.input_fname, "r")

        # origin data is written as int16 which results in a integer overflow
        # when handling the scaling
        self.analog = source_file[self.analog_path][()].astype("int32")
        self.digital = source_file[self.digital_path][()].astype("int32")

        source_file.close()

    def run(self):
        # start 4 worker processes
        pool = Pool(processes=self.n_processes)

        print("\nStart process pool")
        t = time.time()
        try:
            result_list = []
            for pixel_v_sublist in self.process_lists:
                print("process pixel_v_sublist", pixel_v_sublist)
                result_list.append(
                    pool.apply_async(exec_process,
                                     (self.asic, self.input_fname,
                                      self.analog, self.digital,
                                      pixel_v_sublist, self.pixel_u_list,
                                      self.mem_cell_list)))

            for process_result in result_list:
                p_result, v_list, u_list, mem_cell_list = process_result.get()

                if self.result is None:
                    self.initiate_result(p_result)

                self.integrate_result(p_result, v_list, u_list, mem_cell_list)

        finally:
            pool.terminate()

        self.write_data()

        print("process pool took time:", time.time() - t)

    def integrate_result(self, p_result, v_list, u_list, mem_cell_list):
        v_start = v_list[0]
        v_stop = v_list[-1] + 1
        u_start = u_list[0]
        u_stop = u_list[-1] +1
        m_start = mem_cell_list[0]
        m_stop = mem_cell_list[-1] + 1

        for key in p_result["collection"]:
            # do not do this for array time entries because otherwise it would
            # overwrite self.results with a pointer to p_results
            if key not in ["diff_changes_idx", "len_diff_changes_idx"]:
                self.result["collection"][key] = p_result["collection"][key]

        # idx at start: individual, diff_changes_idx
        # subintervals
        idx = (slice(v_start, v_stop),
               slice(u_start, u_stop),
               slice(m_start, m_stop),
               Ellipsis)

        for key in ["slope", "offset", "residuals", "fit_error"]:
            for gain in ["high", "medium", "low"]:
                self.result[key]["individual"][gain][idx] = (
                    p_result[key]["individual"][gain][idx])

        for gain in ["high", "medium", "low"]:
            self.result["intervals"]["subintervals"][gain][idx] = (
                p_result["intervals"]["subintervals"][gain][idx])

        for key in ["diff_changes_idx"]:
            try:
                self.result["collection"][key][idx] = (
                    p_result["collection"][key][idx])
            except:
                print(key, idx)
                print(p_result["collection"][key][idx])
                print(self.result["collection"][key][idx])

        # idx at end: mean, medians, threshold
        idx = (Ellipsis,
               slice(v_start, v_stop),
               slice(u_start, u_stop),
               slice(m_start, m_stop))

        for key in ["slope", "offset", "residuals", "fit_error"]:
            self.result[key]["mean"][idx] = (
                p_result[key]["mean"][idx])
        self.result["medians"][idx] = (
            p_result["medians"][idx])
        self.result["thresholds"][idx] = (
            p_result["thresholds"][idx])

        # only idx: error_code, warning_code, len_diff_changes_idx
        idx = (Ellipsis,
               slice(v_start, v_stop),
               slice(u_start, u_stop),
               slice(m_start, m_stop))

        for key in ["error_code", "warning_code"]:
            self.result[key][idx] = p_result[key][idx]

        for key in ["len_diff_changes_idx"]:
            try:
                self.result["collection"][key][idx] = (
                    p_result["collection"][key][idx])
            except:
                print(key, idx)
                print(p_result["collection"][key][idx])
                print(self.result["collection"][key][idx])

        # special: gain_stages
        idx = (Ellipsis,
               slice(v_start, v_stop),
               slice(u_start, u_stop),
               slice(m_start, m_stop),
               slice(None))

        self.result["intervals"]["gain_stages"][idx] = (
            p_result["intervals"]["gain_stages"][idx])

    def write_data(self):
        if self.result is None:
            print("No results to write")
            return

        save_file = h5py.File(self.output_fname, "w", libver="latest")

        try:
            print("\nStart saving data")
            t = time.time()

            for key in self.result:
                if type(self.result[key]) != dict:
                    save_file.create_dataset("/{}".format(key), data=self.result[key])
                else:
                    for subkey in self.result[key]:
                        if type(self.result[key][subkey]) != dict:
                            save_file.create_dataset("/{}/{}".format(key, subkey),
                                                     data=self.result[key][subkey])
                        else:
                            for gain in ["high", "medium", "low"]:
                                save_file.create_dataset("/{}/{}/{}".format(key, subkey, gain),
                                                         data=self.result[key][subkey][gain])

            save_file.flush()
            print("took time: {}".format(time.time() - t))
        finally:
            save_file.close()

if __name__ == "__main__":

    base_dir = "/gpfs/cfel/fsds/labs/agipd/calibration/processed/"

    #asic = 2
    asic = 1
    module = "M314"
    temperature = "temperature_m15C"
    current = "itestc150"

    input_fname = os.path.join(base_dir, module, temperature, "drscs", current, "gather",
                              "{}_drscs_{}_asic{}.h5".format(module, current, str(asic).zfill(2)))
    output_fname = os.path.join(base_dir, module, temperature, "drscs", current, "process",
                               "{}_drscs_{}_asic{}_processed.h5".format(module, current, str(asic).zfill(2)))

    pixel_v_list = np.arange(64)
    pixel_u_list = np.arange(64)
    mem_cell_list = np.arange(352)
    #pixel_v_list = np.arange(0, 2)
    #pixel_u_list = np.arange(0, 1)
    #mem_cell_list = np.arange(0, 1)

    n_processes = 10

    proc = ParallelProcess(asic, input_fname, pixel_v_list, pixel_u_list,
                           mem_cell_list, n_processes, output_fname)
