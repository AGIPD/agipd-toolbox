from __future__ import print_function

import os
import sys
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
import time
import h5py
from string import Template
from multiprocessing import Pool
from numpy.ma import masked_array
from matplotlib import cm

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
    parser.add_argument("--individual_plots",
                        help="Create plots per asic",
                        action="store_true")
#    parser.add_argument("--store_data",
#                        help="Store calculated matrix into file")
#                        action="store_true")

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


def generate_matrix(result):
    """
    generate map for low gain
    """
    gain = 2
    gain_name = "low"

    matrix = {
        "slope": None,
        "offset": None
    }

    #TODO generic (get from file)
    n_used_fits = 3

    # gain, pixel_v, pixel_u, mem_cell
    # get them from the slope entry (offset and residual are of the same shape)
    n_pixel_v, n_pixel_u, n_mem_cell = result["slope"]["mean"].shape[1:]

    for key in matrix:
        # create array
        matrix[key] = np.empty((n_pixel_v, n_pixel_u, n_mem_cell))
        # intiate with nan
        matrix[key][...] = np.NAN

    for v in np.arange(n_pixel_v):
        for u in np.arange(n_pixel_u):
            for mem_cell in np.arange(n_mem_cell):
                #print("pixel [{},{}], mem_cell {}".format(v, u, mem_cell))

                for key in ["slope", "offset"]:
                    mean = result[key]["mean"][gain, v, u, mem_cell]
                    values = result[key]["individual"][gain_name][v, u, mem_cell][1:-1]
                    #print(key)
                    #print("mean={}, values={}".format(mean, values))

                    try:
                        if mean != 0:
                            matrix[key][v, u, mem_cell] = (
                                np.linalg.norm(values - mean)/np.absolute(n_used_fits * mean))
                    except:
                        print("pixel {}, mem_cell {}".format(mem_cell, v, u))
                        print(key)
                        print("mean={}".format(mean))
                        print("values={}".format(values))

    return matrix


def create_individual_plots(input_fname, mem_cell_list, plot_prefix, plot_ending):
    try:
        colormap_matrix = create_matrix_individual(input_fname)

        create_plots(mem_cell_list, colormap_matrix, plot_prefix, plot_ending, splitted=False)
    except OSError:
        print("OSError:", input_fname)
        pass


def create_matrix_individual(input_fname):
    t = time.time()

    process_result = {
        "slope": {
            "mean": None,
            "individual": {
                "high" : None,
                "medium": None,
                "low": None
            }
        },
        "offset": {
            "mean": None,
            "individual": {
                "high" : None,
                "medium": None,
                "low": None
            }
        },
    }

    # read in data
    source_file = h5py.File(input_fname, "r")

    process_result["slope"]["mean"] = source_file["/slope/mean"][()]
    process_result["slope"]["individual"]["low"] = source_file["/slope/individual/low"][()]

    process_result["offset"]["mean"] = source_file["/offset/mean"][()]
    process_result["offset"]["individual"]["low"] = source_file["/offset/individual/low"][()]

    source_file.close()

    # calcurate matrix
    return generate_matrix(process_result)

def create_plots(mem_cell_list, colormap_matrix, plot_prefix, plot_ending, splitted=False):
    plot_size = (27, 7)

    for mem_cell in mem_cell_list:
        # fill number to be of 3 digits
        for key in colormap_matrix:
            m = colormap_matrix[key][..., mem_cell]

            if splitted and np.where(m >= 1)[0].size != 0:

                m_a = masked_array(m, m < 1)
                m_b = masked_array(m, m >= 1)

                fig, ax = plt.subplots(figsize=plot_size)

                p_a = ax.imshow(m_a, interpolation='nearest', cmap=cm.Reds)
                # [left, bottom, width, height]
                colorbar_ax = fig.add_axes([1.0, 0, 0.01, 0.9])
                fig.colorbar(p_a, cax=colorbar_ax)

                p_b = ax.imshow(m_b, interpolation='nearest', cmap=cm.viridis)
                # [left, bottom, width, height]
                colorbar_ax = fig.add_axes([1.05, 0, 0.01, 0.9])
                fig.colorbar(p_b, cax=colorbar_ax)

            else:
                fig = plt.figure(figsize=plot_size)
                plt.imshow(m)
                plt.colorbar()

            fig.savefig("{}_{}_{}{}".format(plot_prefix, "low", key, plot_ending),
                        bbox_inches='tight')
            fig.clf()
            plt.close(fig)


class CreateColormaps():
    def __init__(self, input_file_dir, output_file_dir, module, current,
                 plot_dir, pixel_v_list, pixel_u_list, mem_cell_list,
                 n_processes, individual_plots=False):

        self.plot_dir = plot_dir
        self.pixel_v_list = pixel_v_list
        self.pixel_u_list = pixel_u_list
        self.mem_cell_list = mem_cell_list

        self.plot_ending = ".png"

        file_prefix = "{}_drscs_{}_asic".format(module, current)
        self.input_prefix = os.path.join(input_file_dir, file_prefix)
        self.output_prefix = os.path.join(output_file_dir, file_prefix)

        self.asics_in_upper_row = np.arange(16,8,-1)
        self.asics_in_lower_row = np.arange(1,9)

        self.asic_list = np.concatenate((self.asics_in_upper_row,
                                         self.asics_in_lower_row))
        #self.asic_list = [2]
        self.asic_list = np.arange(1,17)

        self.individual_plots = individual_plots

        self.n_processes = n_processes

        self.pool = Pool(processes=self.n_processes)

        self.colormap_matrix = {
            "slope": None,
            "offset": None
        }

    def run(self):
        for mem_cell in self.mem_cell_list:

            print("\nStarted at", str(datetime.datetime.now()))
            t= time.time()

            if self.individual_plots:


                # substitute all except asic
                self.plot_template = Template(
                    "${p}/individual/${m}_${c}_asic${a}_${mc}").safe_substitute(
                        p=plot_dir, m=module, c=current, mc=str(mem_cell).zfill(3))
                # make a template out of this string
                self.plot_template = Template(self.plot_template)

                self.get_data_individual()
            else:
                self.plot_prefix = "{}/{}_{}_{}".format(plot_dir,
                                                        module,
                                                        current,
                                                        str(mem_cell).zfill(3))

                self.get_data()
                create_plots(self.mem_cell_list, self.colormap_matrix,
                             self.plot_prefix, self.plot_ending, splitted=True)

            print("\nFinished at {} after {}"
                  .format(datetime.datetime.now(), time.time() - t))

    def get_data_individual(self):

        result_list = []
        for asic in self.asic_list:
            plot_prefix = self.plot_template.substitute(a=asic)

            # the input files for processing are the output ones from gather
            input_fname = "{}{}_processed.h5".format(self.input_prefix, str(asic).zfill(2))
            print("input_file", input_fname)

            result_list.append(
                self.pool.apply_async(
                    create_individual_plots,
                    (input_fname,
                     self.mem_cell_list,
                     plot_prefix,
                     self.plot_ending)))

        for process_result in result_list:
            process_result.get()

    def get_data(self):

        colormap_matrix_upper_row = self.create_row(self.asics_in_upper_row)
        colormap_matrix_lower_row = self.create_row(self.asics_in_lower_row)

        for key in self.colormap_matrix:
            self.colormap_matrix[key] = np.concatenate(
                 (colormap_matrix_upper_row[key],
                  colormap_matrix_lower_row[key]), axis=0)

    def create_row(self, asic_list):
        print ("asic_list={}".format(asic_list))

        generated_matrix = [[] for asic in asic_list]

        matrix = {
            "slope": None,
            "offset": None
        }
        result_list = []

        for asic in asic_list:

            # the input files for processing are the output ones from gather
            input_fname = "{}{}_processed.h5".format(self.input_prefix, str(asic).zfill(2))
            print("input_file", input_fname)

            # calcurate matrix
            result_list.append(
                self.pool.apply_async(
                     create_matrix_individual, (input_fname,)))

        # build matrix for whole module
        for i in range(asic_list.size):

            generated_matrix = result_list[i].get()

            if matrix["slope"] is None:
                matrix["slope"] = generated_matrix["slope"]
                matrix["offset"] = generated_matrix["offset"]

            else:
                matrix["slope"] = np.concatenate(
                    (matrix["slope"], generated_matrix["slope"]), axis=1)
                matrix["offset"] = np.concatenate(
                    (matrix["offset"], generated_matrix["offset"]), axis=1)

        return matrix


if __name__ == "__main__":

    args = get_arguments()

    input_base_dir = args.input_dir
    output_base_dir = args.output_dir
    module = args.module
    temperature = args.temperature
    current = args.current
    individual_plots = args.individual_plots

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
    print("individual_plots: ", individual_plots)

    input_file_dir = os.path.join(input_base_dir, module, temperature, "drscs", current, "process")
    output_file_dir = os.path.join(output_base_dir, module, temperature, "drscs", current)

    plot_dir = os.path.join(output_base_dir, module, temperature, "drscs", "plots", current, "colormaps")
    print("plot_dir", plot_dir)
    create_dir(plot_dir)

    if individual_plots:
        create_dir(os.path.join(plot_dir, "individual"))

    create_error_plots = False
    pixel_v_list = np.arange(29, 30)
    pixel_u_list = np.arange(1, 2)
    mem_cell_list = np.arange(1,2)

    n_processes = 8

    obj = CreateColormaps(input_file_dir, output_file_dir, module, current,
                          plot_dir, pixel_v_list, pixel_u_list, mem_cell_list,
                          n_processes, individual_plots)

    obj.run()
