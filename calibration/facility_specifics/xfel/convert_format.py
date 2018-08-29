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

import os
import argparse
import h5py

import __init__
import utils


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_dir",
                        type=str,
                        required=True,
                        help="Base directory to work in")
    parser.add_argument("--input_file",
                        type=str,
                        required=True,
                        help="Name of the input files")
    parser.add_argument("--output_file",
                        type=str,
                        required=True,
                        help="Template of the output file")
    parser.add_argument("--channel",
                        type=int,
                        required=True,
                        help="Module to gather, e.g 1 (XFEL)")
    parser.add_argument("--output_format",
                        type=str,
                        required=True,
                        help="Format to convert the data to (xfel or agipd)")

    args = parser.parse_args()

    return args


class ConvertFormat(object):
    def __init__(self,
                 input_fname,
                 output_fname,
                 output_format,
                 channel,
                 key_list=None):

        self.input_fname = input_fname
        self.output_fname = output_fname
        self.output_format = output_format
        self.channel = channel

        if key_list is None:
            with h5py.File(self.input_fname, "r") as f:
                any_mod = list(f.keys())[0]
                self.n_mods = len(list(f.keys()))
                key_list = list(f[any_mod].keys())

            key_list.remove("collection")
            self.keys_to_convert = key_list

        else:
            self.keys_to_convert = key_list


    def run(self):
        print("Loading input_file from {}".format(self.input_fname))
        file_content = utils.load_file_content(self.input_fname)

        for ch in range(self.n_mods):
            prefix = "channel{:02}".format(ch)
            for key in self.keys_to_convert:
                file_content[prefix + "/" + key] = self.convert(file_content[prefix + "/" + key], ch)

        print("Writing output_file to {}".format(self.output_fname))
        utils.write_content(self.output_fname, file_content) 

    def convert(self, data, channel):
        if self.output_format == "xfel":
            return utils.convert_to_xfel_format(channel, data)
        elif self.output_format == "agipd":
            return utils.convert_to_agipd_format(channel, data)
        else:
            msg = "Format to which data should be converted is not supported."
            raise Exception(msg)


if __name__ == "__main__":
    args = get_arguments()

    base_dir = args.base_dir
    input_filename = args.input_file
    output_filename = args.output_file
    channel = args.channel
    output_format = args.output_format


    input_fname = os.path.join(base_dir, input_filename.format(channel))
    output_fname = os.path.join(base_dir,
                                output_filename.format(channel, output_format))

    obj = ConvertFormat(input_fname=input_fname,
                        output_fname=output_fname,
                        output_format=output_format,
                        channel=channel)
    obj.run()
