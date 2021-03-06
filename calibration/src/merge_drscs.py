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

from __future__ import print_function

import h5py
import time
import numpy as np
import os
from string import Template
from process import check_file_exists
from parallel_process import integrate_result
from multiprocessing import Pool


class ParallelMerge(object):
    def __init__(self, input_template, output_template, asic_list,
                 n_processes, current_list):
        self.input_template = input_template
        self.output_template = output_template

        self.asic_list = asic_list
        self.n_processes = n_processes
        self.current_list = current_list

        self.run()

    def run(self):
        # start worker processes
        pool = Pool(processes=self.n_processes)

        print("\nStart process pool")
        t = time.time()
        try:
            result_list = []
            for asic in self.asic_list:
                print("merge asic", asic)
#                exec_merge(self.input_template,
#                             self.output_template,
#                             asic,
#                             self.current_list)
                result_list.append(
                    pool.apply_async(exec_merge,
                                     (self.input_template,
                                      self.output_template,
                                      asic,
                                      self.current_list)))

            for pool_result in result_list:
                pool_result.get()

        finally:
            pool.terminate()
        print("process pool took time:", time.time() - t)


def exec_merge(input_template, output_template, asic, current_list):
    output_fname = output_template.substitute(a=str(asic).zfill(2))

    obj = None
    try:
        obj = MergeDrscs(input_template, output_fname, asic, current_list)
    except:
        pass

    # for some reason the process does not stop if MergeDrscs failes
    if obj is not None:
        obj.run()


class MergeDrscs(object):
    def __init__(self, cs_input_template, output_fname, asic, current_list):

        self.cs_input_template = cs_input_template

        self.output_fname = output_fname
        check_file_exists(self.output_fname)

        self.asic = str(asic).zfill(2)

        self.error_path = "/error_code"

        self.current_list = current_list
        self.fit_info = dict()
        self.sorted_fit_info = None

        self.recovered_fits = dict()
        for c in self.current_list:
            self.recovered_fits[c] = []

        self.result = dict()

    def run(self):
        self.get_failed_fits()

        self.load_least_failed()

        self.determine_current_for_fails()

        self.recover_failed_fits()

        self.write_data()

    def get_failed_fits(self):
        for current in self.current_list:
            input_fname = self.cs_input_template.substitute(c=current,
                                                            a=self.asic)
            source_file = h5py.File(input_fname, "r")

            try:
                self.fit_info[current] = dict()
                self.fit_info[current]["error_code"] = (
                    source_file[self.error_path][()])
            finally:
                source_file.close()

            condition = self.fit_info[current]["error_code"] != 0
            self.fit_info[current]["failed_fits"] = np.where(condition)
            self.fit_info[current]["n_failed_fits"] = (
                self.fit_info[current]["failed_fits"][0].size)

        # sort from the current with the most successful fits to the one
        # with the least
        self.sorted_fit_info = [i[0]
                                for i in sorted(
                                    self.fit_info.items(),
                                    key=lambda j: j[1]["n_failed_fits"])]

    def load_least_failed(self):
        current = self.sorted_fit_info[0]

        input_fname = self.cs_input_template.substitute(c=current, a=self.asic)
        source_file = h5py.File(input_fname, "r")

        try:
            for key in source_file["/"].keys():
                source_key = "/" + key
                if isinstance(source_file[source_key], h5py.Dataset):
                    self.result[key] = source_file[source_key][()]
                else:
                    if key not in self.result:
                        self.result[key] = dict()
                    for subkey in source_file[source_key].keys():
                        source_key2 = source_key + "/" + subkey
                        if isinstance(source_file[source_key2], h5py.Dataset):
                            self.result[key][subkey] = (
                                source_file[source_key2][()])
                        else:
                            if subkey not in self.result[key]:
                                self.result[key][subkey] = dict()
                            for gain in source_file[source_key2].keys():
                                source_key3 = source_key2 + "/" + gain
                                self.result[key][subkey][gain] = (
                                    source_file[source_key3][()])
        finally:
            source_file.close()

        current_as_number = self.convert_current_to_number(current)
        shape = self.result["error_code"].shape

        self.result["chosen_current"] = (
            current_as_number * np.ones(shape, dtype=np.int))
        # TODO set all failed ones to nan
        failed_fits = self.fit_info[current]["failed_fits"]
        self.result["chosen_current"][failed_fits] = 0

    def convert_current_to_number(self, current):
        return int(current[len("itestc"):])

    def determine_current_for_fails(self):
        current = self.sorted_fit_info[0]
        failed_fits = self.fit_info[current]["failed_fits"]

        for c in self.sorted_fit_info[1:]:
            condition = self.fit_info[c]["error_code"][failed_fits] == 0
            recoverable = np.where(condition)

            self.recovered_fits[c] = (failed_fits[0][recoverable],
                                      failed_fits[1][recoverable],
                                      failed_fits[2][recoverable])

    def recover_failed_fits(self):
        for current in self.recovered_fits:
            if (self.recovered_fits[current] and
                    self.recovered_fits[current][0].size != 0):
                input_fname = self.cs_input_template.substitute(c=current,
                                                                a=self.asic)
                source_file = h5py.File(input_fname, "r")

                try:
                    idx = self.recovered_fits[current]

                    integrate_result(idx, self.result, source_file)
                    self.result["chosen_current"][idx] = (
                        self.convert_current_to_number(current))
                finally:
                    source_file.close()

    def write_data(self):
        output_file = h5py.File(self.output_fname, "w", libver="latest")

        try:
            print("\nStart saving data")
            t = time.time()

            for key in self.result:
                if type(self.result[key]) != dict:
                    output_file.create_dataset(
                        "/{}".format(key),
                        data=self.result[key])
                else:
                    for subkey in self.result[key]:
                        if type(self.result[key][subkey]) != dict:
                            output_file.create_dataset(
                                "/{}/{}".format(key, subkey),
                                data=self.result[key][subkey])
                        else:
                            for gain in ["high", "medium", "low"]:
                                output_file.create_dataset(
                                    "/{}/{}/{}".format(key, subkey, gain),
                                    data=self.result[key][subkey][gain])

            output_file.flush()
            print("took time: {}".format(time.time() - t))
        finally:
            output_file.close()


if __name__ == "__main__":
    base_path = "/gpfs/cfel/fsds/labs/agipd/calibration/processed/"
    module = "M234"
    temperature = "temperature_m20C"
    asic_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    n_processes = 10

    input_path = os.path.join(base_path, module, temperature, "drscs")
    # substitute all except current and asic
    input_template = (
        Template("${p}/${c}/process/${m}_drscs_${c}_asic${a}_processed.h5")
        .safe_substitute(p=input_path, m=module)
    )
    # make a template out of this string to let Merge set current and asic
    input_template = Template(input_template)

    output_path = os.path.join(base_path, module, temperature, "drscs",
                               "merged")
    output_template = (
        Template("${p}/${m}_drscs_asic${a}_merges.h5")
        .safe_substitute(p=output_path, m=module, t=temperature)
    )
    output_template = Template(output_template)

    ParallelMerge(input_template, output_template, asic_list, n_processes)
