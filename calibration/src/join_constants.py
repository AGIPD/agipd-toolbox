#!/usr/bin/env python


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

import argparse
import glob
import h5py
import os

import utils


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir",
                        type=str,
                        required=True,
                        help="Input directory to load data from")
    parser.add_argument("--input_file",
                        type=str,
                        required=True,
                        help="Template for input files")
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="Output directory to write results to")
    parser.add_argument("--output_file",
                        type=str,
                        required=True,
                        help="Name of the output file")

    args = parser.parse_args()

    return args


class JoinConstants(object):
    def __init__(self, in_fname, out_fname, facility="xfel"):
        self._in_fname = in_fname
        self._out_fname = out_fname
        self._facility = facility

    def get_file_names(self):
        fname = self._in_fname.replace("AGIPD{:02}", "AGIPD[0-9][0-9]")

        if "asic" in self._in_fname:
            fname = fname.replace("asic{:02}", "asic[0-9][0-9]")
                
        file_list = glob.glob(fname)
        file_list.sort()
        
        return file_list

    def run(self):
        file_list = self.get_file_names()

        files = {}
        for fname in file_list:
            if self._facility == 'cfel':
                channel = 0
            else:
                channel = int(fname.split("AGIPD")[1][:2])

            if channel in files:
                files[channel].append(fname)
            else:
                files[channel] = [fname]

        data = {}
        # collect data and prepare for concatenation to module
        for channel, file_list in files.items():
            data[channel] = {}
            data_to_be_concatenated = {}
            for fname in file_list:
                split_res = fname.split("asic")
                if len(split_res) == 1:
                    asic = "all"
                else:
                    asic = int(fname.split("asic")[1][:2])

                print("channel{}: loading content of file {}"
                      .format(channel, fname))
                file_content = utils.load_file_content(fname)

                for key, value in file_content.items():
                    # metadata does not have to be concatenated
                    if key.startswith("collection"):
                        # entry is redundant for the other asics
                        if key not in data[channel]:
                            data[channel][key] = value

                    # mark to be concatenated
                    else:
                        if key not in data_to_be_concatenated:
                            data_to_be_concatenated[key] = {}

                        data_to_be_concatenated[key][asic] = value

            # rebuild the module from the asic data
            for key, value in data_to_be_concatenated.items():
                data[channel][key] = utils.build_module(value)

        # write
        with h5py.File(self._out_fname, "w") as f:
            for channel, data_channel in data.items():
                prefix = "channel{:02d}".format(channel)
                for key, value in data_channel.items():
                    f.create_dataset(prefix + "/" + key, data=value)
                    f.flush()


if __name__ == "__main__":
    args = get_arguments()

#    base_dir = "/gpfs/exfel/exp/SPB/201730/p900009/scratch/user/kuhnm/dark"
#    in_file_name = "dark_AGIPD{:02d}_agipd_2017-10-27.h5"

    in_dir = args.input_dir
    in_file_name = args.input_file
    in_fname = os.path.join(in_dir, in_file_name)

#    out_file_name = "dark_joined_constants_agipd.h5"
    out_file_name = args.output_file
    out_dir = args.output_dir
    out_fname = os.path.join(out_dir, out_file_name)

    obj = JoinConstants(in_fname, out_fname)
    obj.run()
