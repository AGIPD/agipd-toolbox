import builtins
import copy
from multiprocessing import Pool
import os
import time

import h5py
import numpy as np

from .util import check_file_exists

debug_mode = False
def print(*args):
    if debug_mode:
        builtins.print(args)

def exec_process(asic, input_fname, pixel_v_list, pixel_u_list,
                 mem_cell_list, safety_factor, input_file_handle):

    cal = ProcessDrscs(asic,
                       input_fname=input_fname,
                       safety_factor=safety_factor,
                       input_handle=input_file_handle)

    cal.run(pixel_v_list, pixel_u_list, mem_cell_list)

    return cal.result, pixel_v_list, pixel_u_list, mem_cell_list


class ParallelProcess()i:
    def __init__(self, asic, input_fname, pixel_v_list, pixel_u_list,
                 mem_cell_list, n_processes, safety_factor, output_fname,
                 gather_file_handle, reuse_results):
        self.asic = asic
        self.input_fname = input_fname
        self.input_handle = gather_file_handle

        self.pixel_v_list = pixel_v_list
        self.pixel_u_list = pixel_u_list
        self.mem_cell_list = mem_cell_list
        self.n_gain_stages = 3

        self.n_processes = n_processes
        self.process_lists = []

        self.safety_factor = safety_factor

        self.output_fname = output_fname
        self.result = None

        self.digital_path = "/entry/instrument/detector/data_digital"
        self.analog_path = "/entry/instrument/detector/data"

        self.do_process = True
        if not reuse_results:
            check_file_exists(self.output_fname)
        else:
            if not os.path.exists(self.output_fname):
                print("Result file didn't exist, processing results")
            else:
                self.do_process = False

        if self.do_process:
            self.generate_process_lists()

    def generate_process_lists(self):

        if len(self.pixel_v_list) <= self.n_processes:
            self.process_lists = [np.array([i]) for i in self.pixel_v_list]
        else:
            size = int(len(self.pixel_v_list) / self.n_processes)
            rest = len(self.pixel_v_list) % self.n_processes

            # distribute the workload
            stop = len(self.pixel_v_list) - size*(rest+1)
            self.process_lists = [self.pixel_v_list[i:i + size]
                                  for i in range(0, stop, size)]

            # if list to split is not a multiple of size, the rest is equaly
            # distributed over the remaining processes
            start = len(self.process_lists)*size
            stop = len(self.pixel_v_list)
            self.process_lists += [self.pixel_v_list[i:i+size+1]
                                   for i in range(start, stop, size + 1)]

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

        # +1 because counting starts with zero
        self.dim_v = self.pixel_v_list.max() + 1
        self.dim_u = self.pixel_u_list.max() + 1
        self.dim_mem_cell = self.mem_cell_list.max() + 1

        self.result = initiate_result(self.dim_v, self.dim_u,
                                      self.dim_mem_cell, n_gain_stages,
                                      n_intervals, n_diff_changes_stored)

    def run(self):
        if self.do_process:
            # start worker processes
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
                                          pixel_v_sublist, self.pixel_u_list,
                                          self.mem_cell_list,
                                          self.safety_factor,
                                          self.input_handle)))

                for process_result in result_list:
                    p_result, v_list, u_list, mem_cell_list = process_result.get()

                    if self.result is None:
                        self.initiate_result(p_result)

                    self.integrate_result(p_result, v_list, u_list, mem_cell_list)

            finally:
                pool.terminate()

            if self.input_handle is None:
                self.write_data()

            print("process pool took time:", time.time() - t)
        else:
            self.read_data()

        return self.get_data()

    def integrate_result(self, p_result, v_list, u_list, mem_cell_list):
        v_start = v_list[0]
        v_stop = v_list[-1] + 1
        u_start = u_list[0]
        u_stop = u_list[-1] + 1
        m_start = mem_cell_list[0]
        m_stop = mem_cell_list[-1] + 1

        for key in p_result["collection"]:
            # do not do this for array time entries because otherwise it would
            # overwrite self.results with a pointer to p_results
            if key not in ["diff_changes_idx", "len_diff_changes_idx"]:
                self.result["collection"][key] = p_result["collection"][key]

        # idx at start: individual, subintervals, diff_changes_idx, saturation
        idx = (slice(v_start, v_stop),
               slice(u_start, u_stop),
               slice(m_start, m_stop),
               Ellipsis)

        for key in ["slope", "offset", "residual", "average_residual"]:
            for gain in ["high", "medium", "low"]:
                self.result[key]["individual"][gain][idx] = (
                    p_result[key]["individual"][gain][...])

        for gain in ["high", "medium", "low"]:
            self.result["intervals"]["subintervals"][gain][idx] = (
                p_result["intervals"]["subintervals"][gain][...])

        for key in ["diff_changes_idx"]:
            try:
                self.result["collection"][key][idx] = (
                    p_result["collection"][key][...])
            except:
                print(key, idx)
                print(p_result["collection"][key][...])
                print(self.result["collection"][key][idx])

        self.result["intervals"]["saturation"][idx] = (
            p_result["intervals"]["saturation"][...])

        # idx at end: mean, medians, threshold
        idx = (Ellipsis,
               slice(v_start, v_stop),
               slice(u_start, u_stop),
               slice(m_start, m_stop))

        for key in ["slope", "offset", "residual", "average_residual"]:
            self.result[key]["mean"][idx] = (
                p_result[key]["mean"][...])

        self.result["medians"][idx] = (
            p_result["medians"][...])
        self.result["thresholds"][idx] = (
            p_result["thresholds"][...])

        # only idx: error_code, warning_code, len_diff_changes_idx
        idx = (Ellipsis,
               slice(v_start, v_stop),
               slice(u_start, u_stop),
               slice(m_start, m_stop))

        for key in ["error_code", "warning_code"]:
            self.result[key][idx] = p_result[key][...]

        for key in ["len_diff_changes_idx"]:
            try:
                self.result["collection"][key][idx] = (
                    p_result["collection"][key][...])
            except:
                print(key, idx)
                print(p_result["collection"][key][...])
                print(self.result["collection"][key][idx])

        # special: gain_stages
        idx = (Ellipsis,
               slice(v_start, v_stop),
               slice(u_start, u_stop),
               slice(m_start, m_stop),
               slice(None))

        self.result["intervals"]["gain_stages"][idx] = (
            p_result["intervals"]["gain_stages"][...])

    def read_data(self):
        self.result = {}
        source_file = h5py.File(self.output_fname, "r")

        try:
            for key in source_file["/"].keys():
                source_key = "/" + key
                if isinstance(source_file[source_key], h5py.Dataset):
                    self.result[key] = source_file[source_key][()]
                else:
                    if key not in self.result: self.result[key] = dict()
                    for subkey in source_file[source_key].keys():
                        source_key2 = source_key + "/" + subkey
                        if isinstance(source_file[source_key2], h5py.Dataset):
                            self.result[key][subkey] = source_file[source_key2][()]
                        else:
                            if subkey not in self.result[key]: self.result[key][subkey] = dict()
                            for gain in source_file[source_key2].keys():
                                source_key3 = source_key2 + "/" + gain
                                self.result[key][subkey][gain] = source_file[source_key3][()]
        finally:
            source_file.close()

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
                    path = "/{}".format(key)
                    data = self.result[key]
                    save_file.create_dataset(path, data=data)
                else:
                    for subkey in self.result[key]:
                        if type(self.result[key][subkey]) != dict:
                            path = "/{}/{}".format(key, subkey)
                            data = self.result[key][subkey]
                            save_file.create_dataset(path, data=data)
                        else:
                            for gain in ["high", "medium", "low"]:
                                path = "/{}/{}/{}".format(key, subkey, gain)
                                data = self.result[key][subkey][gain]
                                save_file.create_dataset(path, data=data)

            save_file.flush()
            print("took time: {}".format(time.time() - t))
        except Exception as e:
            print("Failed writing results {}".format(e))
        finally:
            save_file.close()

    def get_data(self):
        return self.result
