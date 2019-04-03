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

#from __future__ import print_function

from multiprocessing import Pool, TimeoutError
import numpy as np
import os
import time
import h5py
from Process import ProcessDrscs, initiate_result, check_file_exists
#from functools import reduce  # forward compatibility for Python 3
import operator

def exec_process(in_fname, out_fname, pixel_v_list, pixel_u_list, mem_cell_list):

    create_error_plots=False


    #cal = ProcessDrscs(asic, analog=analog, digital=digital)
    cal = ProcessDrscs(input_file, out_fname, runs, run_name)
    cal.run(pixel_v_list, pixel_u_list, mem_cell_list)

    return cal.result, pixel_v_list, pixel_u_list, mem_cell_list


def map_to_dict(map_list, result):
    return reduce(operator.getitem, map_list, result)


def map_to_hdf5(map_list, result):
    return  np.array(result["/" + "/".join(map_list)])


def integrate_result(idx, result, source):

    if type(source) == h5py._hl.files.File:
        map_to_format = map_to_hdf5

        # these have to be treated differently because it contains ints/floats
        for key in source["collection"].keys():
            # do not do this for array time entries because otherwise it would
            # overwrite self.results with a pointer to p_results
            if key not in ["diff_changes_idx", "len_diff_changes_idx"]:
                result["collection"][key] = source["/collection/" + key][()]

    elif type(source) == dict():
        map_to_format = map_to_dict

        # these have to be treated differently because it contains ints/floats
        for key in source["collection"]:
            # do not do this for array time entries because otherwise it would
            # overwrite self.results with a pointer to p_results
            if key not in ["diff_changes_idx", "len_diff_changes_idx"]:
                result["collection"][key] = source["collection"][key]
    else:
        raise("Source to intergate of unsupported format")



    # idx at start: individual, subintervals, diff_changes_idx, saturation
    index = idx + (Ellipsis,)

    for key in ["slope", "offset", "residual", "average_residual"]:
        for gain in ["high", "medium", "low"]:
            source_key = [key, "individual", gain]
            map_to_dict(source_key, result)[index] = (
                map_to_format(source_key, source)[index])

    for gain in ["high", "medium", "low"]:
        source_key = ["intervals", "subintervals", gain]
        map_to_dict(source_key, result)[index] = (
            map_to_format(source_key, source)[index])

    for key in ["diff_changes_idx"]:
        source_key = ["collection", key]
        map_to_dict(source_key, result)[index] = (
            map_to_format(source_key, source)[index])

    source_key = ["intervals", "saturation"]
    map_to_dict(source_key, result)[index] = (
        map_to_format(source_key, source)[index])

    # idx at end: mean, medians, threshold
    index = (Ellipsis, ) + idx

    for key in ["slope", "offset", "residual", "average_residual"]:
        source_key = [key, "mean"]
        map_to_dict(source_key, result)[index] = (
            map_to_format(source_key, source)[index])

    for key in ["medians", "thresholds"]:
        source_key = [key]
        map_to_dict(source_key, result)[index] = (
            map_to_format(source_key, source)[index])

    # only idx: error_code, warning_code, len_diff_changes_idx
    index = (Ellipsis,) + idx

    for key in ["error_code", "warning_code"]:
        source_key = [key]
        map_to_dict(source_key, result)[index] = (
            map_to_format(source_key, source)[index])

    for key in ["len_diff_changes_idx"]:
        source_key = ["collection", key]
        map_to_dict(source_key, result)[index] = (
            map_to_format(source_key, source)[index])

    # special: gain_stages
    index = (Ellipsis,) + idx + (slice(None),)

    source_key = ["intervals", "gain_stages"]
    map_to_dict(source_key, result)[index] = (
        map_to_format(source_key, source)[index])



class ParallelProcess(object):
    def __init__(self, in_fname, out_fname, runs, run_name, n_processes):
#    def __init__(self,input_fname, pixel_v_list, pixel_u_list,
#                 mem_cell_list, n_processes, output_fname):

        self._out_fname = out_fname

        self.in_fname = in_fname

        self.runs = runs
        self.run_names = run_name
        self._row_location = None
        self._col_location = None
        self._memcell_location = None
        self._frame_location = None
        self._set_data_order()

        self._set_dims_and_metadata()

        self.n_gain_stages = 3

        self.n_processes = n_processes
        self.process_lists = []

        self.result = None


#        check_file_exists(self._out_fname)

        self.generate_process_lists()

        #self.load_data()

        self.run()

    def _set_data_order(self):
        """Set the locations where the data is stored

        This give the different process methods the posibility to act genericly
        to data reordering.

        """
        self._row_location = 0
        self._col_location = 1
        self._memcell_location = 2
        self._frame_location = 3

    def _set_dims_and_metadata(self):
        run_number = self.runs[0]
        run_name = self.run_names[0]

        in_fname = self.in_fname.format(run_number=run_number, run_name=run_name)
        with h5py.File(in_fname, "r") as f:
            shape = f['analog'].shape

            self.module = f['collection/module'][()]
            self.channel = f['collection/channel'][()]

        self.n_rows = shape[self._row_location]
        self.n_cols = shape[self._col_location]
        self.n_memcells = shape[self._memcell_location]

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

        self.result = initiate(self.pixel_v_list, self.pixel_u_list,
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
                                     (self.input_fname,
                                      self.analog, self.digital,
                                      pixel_v_sublist, self.pixel_u_list,
                                      self.mem_cell_list))

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

        # idx at start: individual, subintervals, diff_changes_idx, saturation
        idx = (slice(v_start, v_stop),
               slice(u_start, u_stop),
               slice(m_start, m_stop),
               Ellipsis)

        for key in ["slope", "offset", "residual", "average_residual"]:
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

        self.result["intervals"]["saturation"][idx] = (
            p_result["intervals"]["saturation"][idx])

        # idx at end: mean, medians, threshold
        idx = (Ellipsis,
               slice(v_start, v_stop),
               slice(u_start, u_stop),
               slice(m_start, m_stop))

        for key in ["slope", "offset", "residual", "average_residual"]:
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
