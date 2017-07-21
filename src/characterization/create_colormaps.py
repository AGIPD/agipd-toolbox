from __future__ import print_function

import os
import sys
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
import time
import h5py

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir",
                        type=str,
                        required=True,
                        help="Directory to get data from")
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="Base directory to write results to")
    parser.add_argument("--module",
                        type=str,
                        required=True,
                        help="Module to gather, e.g M310")
    parser.add_argument("--temperature",
                        type=str,
                        required=True,
                        help="temperature to gather, e.g. temperature_30C")
    parser.add_argument("--current",
                        type=str,
                        required=True,
                        help="Current to use, e.g. itestc20")

    args = parser.parse_args()

    return args


def create_dir(directory_name):
    if not os.path.exists(directory_name):
        try:
            os.makedirs(directory_name)
            print("Dir '{0}' does not exist. Create it.".format(directory_name))
        except IOError:
            if os.path.isdir(directory_name):
                pass


class CreateColormaps():
    def __init__(self, input_file_dir, output_file_dir, module, current,
                 plot_dir, pixel_v_list, pixel_u_list, mem_cell_list, mode):

        self.plot_dir = plot_dir
        self.pixel_v_list = pixel_v_list
        self.pixel_u_list = pixel_u_list
        self.mem_cell_list = mem_cell_list

        self.plot_ending = ".png"

        file_prefix = "{}_drscs_{}_asic".format(module, current)
        self.input_prefix = os.path.join(input_file_dir, file_prefix)
        self.output_prefix = os.path.join(output_file_dir, file_prefix)

        self.asics_in_upper_row = range(16,8,-1)
        self.asics_in_lower_row = range(1,9)

        if mode == "live":
            self.create_row = self.create_row_from_raw
        elif mode == "file":
            self.create_row = self.create_row_from_file
        else:
            raise Exception("mode not supported")

        self.colormap_matrix = {
            "slope": None,
            "offset": None
        }

    def run(self):
        for mem_cell in self.mem_cell_list:

            self.plot_prefix = "{}/{}_{}_{}".format(plot_dir, module, current,
                                                    str(mem_cell).zfill(3))

            print("\nStarted at", str(datetime.datetime.now()))
            t= time.time()
            self.get_data()
            self.create_plots()
            print("\nFinished at {} after {}"
                  .format(datetime.datetime.now(), time.time() - t))

    def get_data(self):

        colormap_matrix_upper_row = self.create_row(self.asics_in_upper_row)
        colormap_matrix_lower_row = self.create_row(self.asics_in_lower_row)

        for key in self.colormap_matrix:
            self.colormap_matrix[key] = np.concatenate(
                 (colormap_matrix_upper_row[key],
                  colormap_matrix_lower_row[key]), axis=0)

    def create_row(self, asic_list):
        pass

    def create_row_from_file(self, asic_list):
        print ("asic_list={}".format(asic_list))

        colormap_matrix = {
            "slope": None,
            "offset": None
        }

        colormap_matrix_per_asic = {
            "slope": None,
            "offset": None
        }

        for asic in asic_list:

            # the input files for processing are the output ones from gather
            input_fname = "{}{}_processed.h5".format(self.input_prefix, str(asic).zfill(2))
            print("input_file", input_fname)

            source_file = h5py.File(input_fname, "r")

            # origin data is written as int16 which results in a integer overflow
            # when handling the scaling
            colormap_matrix_per_asic["slope"] = source_file["/colormap_matrix/slope"][()]
            colormap_matrix_per_asic["offset"] = source_file["/colormap_matrix/offset"][()]

            source_file.close()

            if colormap_matrix["slope"] is None:
                colormap_matrix["slope"] = colormap_matrix_per_asic["slope"]
                colormap_matrix["offset"] = colormap_matrix_per_asic["offset"]

            else:
                colormap_matrix["slope"] = np.concatenate(
                    (colormap_matrix["slope"], colormap_matrix_per_asic["slope"]), axis=1)
                colormap_matrix["offset"] = np.concatenate(
                    (colormap_matrix["offset"], colormap_matrix_per_asic["offset"]), axis=1)

        return colormap_matrix

    def create_row_from_raw(self, asic_list):
        try:
            CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
        except:
            CURRENT_DIR = os.path.dirname(os.path.realpath('__file__'))

        SRC_PATH = os.path.dirname(CURRENT_DIR)

        if SRC_PATH not in sys.path:
            sys.path.append(SRC_PATH)

        from process_data_per_asic import ProcessDrscs

        print ("asic_list={}".format(asic_list))

        colormap_matrix = {
            "slope": None,
            "offset": None
        }

        for asic in asic_list:

            # the input files for processing are the output ones from gather
            input_file = "{}{}.h5".format(self.input_prefix, str(asic).zfill(2))
            print("input_file", input_file)

            output_file = "{}{}_processed.h5".format(self.output_prefix, str(asic).zfill(2))

            cal = ProcessDrscs(input_file, output_file, self.plot_dir, create_plots=False)

            cal.run(self.pixel_v_list, self.pixel_u_list, self.mem_cell_list,
                    create_error_plots, create_colormaps=["matrix"], write_data=False)

            if colormap_matrix["slope"] is None:# and colormap_matrix["offset"] is None:
                colormap_matrix["slope"] = cal.colormap_matrix_slope
                colormap_matrix["offset"] = cal.colormap_matrix_offset

            else:
                colormap_matrix["slope"] = np.concatenate(
                   (colormap_matrix["slope"], cal.colormap_matrix_slope), axis=1)
                colormap_matrix["offset"] = np.concatenate(
                    (colormap_matrix["offset"], cal.colormap_matrix_offset), axis=1)

        return colormap_matrix

    def create_plots(self):
        #fig = plt.figure(figsize=None)
        #zoom = 8
        #w, h = fig.get_size_inches()
        #plot_size = (w * zoom, h * zoom)
        #plt.close(fig)
        plot_size = (27, 7)

        for mem_cell in self.mem_cell_list:
            # fill number to be of 3 digits
            for key in self.colormap_matrix:
                fig = plt.figure(figsize=plot_size)
                plt.imshow(self.colormap_matrix[key][..., mem_cell])
                plt.colorbar()

                fig.savefig("{}_{}_{}{}".format(self.plot_prefix, "low", key, self.plot_ending),
                            bbox_inches='tight')
                fig.clf()
                plt.close(fig)


if __name__ == "__main__":

    args = get_arguments()

    input_base_dir = args.input_dir
    output_base_dir = args.output_dir
    module = args.module
    temperature = args.temperature
    current = args.current

    #input_base_dir = "/gpfs/cfel/fsds/labs/agipd/calibration/processed/"
    #output_base_dir = "/gpfs/cfel/fsds/labs/agipd/calibration/processed/"
    #module = "M314"
    #temperature = "temperature_m15C"
    #current = "itestc150"

    print("Configured parameter:")
    print("input_dir: ", input_base_dir)
    print("output_dir: ", output_base_dir)
    print("module: ", module)
    print("temperature: ", temperature)
    print("current: ", current)

    input_file_dir = os.path.join(input_base_dir, module, temperature, "drscs", current, "process_result")
    output_file_dir = os.path.join(output_base_dir, module, temperature, "drscs", current)

    plot_dir = os.path.join(output_base_dir, module, temperature, "drscs", "plots", current, "colormaps")
    print("plot_dir", plot_dir)
    create_dir(plot_dir)

    create_error_plots = False
    pixel_v_list = np.arange(64)
    pixel_u_list = np.arange(64)
    mem_cell_list = np.arange(1,2)

    #mode = "live"
    mode = "file"

    obj = CreateColormaps(input_file_dir, output_file_dir, module, current,
                          plot_dir, pixel_v_list, pixel_u_list, mem_cell_list, mode)

    obj.run()
