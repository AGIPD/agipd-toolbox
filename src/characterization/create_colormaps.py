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
                        default="/gpfs/cfel/fsds/labs/agipd/calibration/processed",
                        help="Directory to get data from")
    parser.add_argument("--output_dir",
                        type=str,
                        default="/gpfs/cfel/fsds/labs/agipd/calibration/processed",
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
                        help="Current to use (e.g. itestc20) or combined")
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

def generate_matrix(result, gain_name, quality):
    """
    generate map for low gain
    """
    gain_d = {
        "high": 0,
        "medium": 1,
        "low": 2
    }
    gain = gain_d[gain_name]

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

    if quality:
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
                        except Exception as e:
                            print("Error was: ", e)
                            print("pixel {}, mem_cell {}".format(mem_cell, v, u))
                            print(key)
                            print("mean={}".format(mean))
                            print("values={}".format(values))
    else:
        for key in ["slope", "offset"]:
            matrix[key] = result[key]["mean"][gain, ...]

            idx = np.where(result["error_code"] != 0)
            matrix[key][idx] = np.NAN

    return matrix


def create_individual_plots(input_fname, mem_cell_list, plot_prefix, plot_ending, gain_name, quality):
    try:
        colormap_matrix = create_matrix_individual(input_fname, gain_name, quality)

        create_plots(mem_cell_list, colormap_matrix, plot_prefix, plot_ending,
                     gain_name, quality, splitted=False)
    except OSError:
        print("OSError:", input_fname)
        pass


def create_matrix_individual(input_fname, gain_name, quality):
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

    process_result["error_code"] = source_file["/error_code"][()]
    source_file.close()

    # calcurate matrix
    return generate_matrix(process_result, gain_name, quality)

def create_plots(mem_cell_list, colormap_matrix, plot_file_prefix, plot_ending,
                 gain_name, quality, splitted=False):
    plot_size = (27, 7)
                # [left, bottom, width, height]
    ax_location = [0.91, 0.11, 0.01, 0.75]
    ax2_location = [0.94, 0.11, 0.01, 0.75]

    for mem_cell in mem_cell_list:
        # fill number to be of 3 digits
        for key in colormap_matrix:
            m = colormap_matrix[key][..., mem_cell]

            if splitted and quality and np.where(m >= 1)[0].size != 0:

                m_a = masked_array(m, m < 1)
                m_b = masked_array(m, m >= 1)

                fig, ax = plt.subplots(figsize=plot_size)

                p_a = ax.imshow(m_a, interpolation='nearest', cmap=cm.Reds)
                colorbar_ax = fig.add_axes(ax_location)
                fig.colorbar(p_a, cax=colorbar_ax)

                p_b = ax.imshow(m_b, interpolation='nearest', cmap=cm.viridis)
                colorbar_ax = fig.add_axes(ax2_location)
                fig.colorbar(p_b, cax=colorbar_ax)

            else:
                fig = plt.figure(figsize=plot_size)
                plt.imshow(m)
                colorbar_ax = fig.add_axes(ax_location)
                plt.colorbar(cax=colorbar_ax)

            title_prefix = plot_file_prefix.rsplit("/", 1)[1]
            plt.suptitle("{}_{} {}".format(title_prefix, gain_name, key), fontsize=24)

            fig.savefig("{}_{}_{}{}".format(plot_file_prefix, gain_name, key, plot_ending),
                        bbox_inches='tight')
            fig.clf()
            plt.close(fig)


class CreateColormaps():
    def __init__(self, input_template, output_template, module, current,
                 plot_dir, pixel_v_list, pixel_u_list, mem_cell_list,
                 n_processes, gain_name="low", quality=True, individual_plots=False):

        self.plot_dir = plot_dir
        self.pixel_v_list = pixel_v_list
        self.pixel_u_list = pixel_u_list
        self.mem_cell_list = mem_cell_list

        self.gain_name = gain_name
        self.quality = quality

        self.plot_ending = ".png"

        self.input_template = input_template
        self.output_template = output_template

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


                if quality:
                    # substitute all except asic
                    self.plot_template = Template(
                        "${p}/individual/${m}_${c}_asic${a}_${mc}_quality").safe_substitute(
                            p=plot_dir, m=module, c=current, mc=str(mem_cell).zfill(3))
                    # make a template out of this string
                    self.plot_template = Template(self.plot_template)
                else:
                    # substitute all except asic
                    self.plot_template = Template(
                        "${p}/individual/${m}_${c}_asic${a}_${mc}").safe_substitute(
                            p=plot_dir, m=module, c=current, mc=str(mem_cell).zfill(3))
                    # make a template out of this string
                    self.plot_template = Template(self.plot_template)

                self.get_data_individual()
            else:
                if quality:
                    self.plot_prefix = ("{}/{}_{}_{}_quality"
                                        .format(plot_dir,
                                                module,
                                                current,
                                                str(mem_cell).zfill(3)))
                else:
                    self.plot_prefix = "{}/{}_{}_{}".format(plot_dir,
                                                            module,
                                                            current,
                                                            str(mem_cell).zfill(3))

                self.get_data()
                create_plots(self.mem_cell_list, self.colormap_matrix,
                             self.plot_prefix, self.plot_ending, self.gain_name,
                             self.quality, splitted=True)

            print("\nFinished at {} after {}"
                  .format(datetime.datetime.now(), time.time() - t))

    def get_data_individual(self):

        result_list = []
        for asic in self.asic_list:
            plot_prefix = self.plot_template.substitute(a=asic)

            # the input files for processing are the output ones from gather
            input_fname = self.input_template.substitute(a=str(asic).zfill(2))
            print("input_file", input_fname)

            result_list.append(
                self.pool.apply_async(
                    create_individual_plots,
                    (input_fname,
                     self.mem_cell_list,
                     plot_prefix,
                     self.plot_ending,
                     self.gain_name,
                     self.quality)))

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
            input_fname = self.input_template.substitute(a=str(asic).zfill(2))
            print("input_file", input_fname)

            # calcurate matrix
            result_list.append(
                self.pool.apply_async(
                     create_matrix_individual, (input_fname, self.gain_name, self.quality)))

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

    if current == "combined":
        input_path = os.path.join(input_base_dir, module, temperature, "drscs", "combined")
        # substitute all except current and asic
        input_template = Template("${p}/${m}_drscs_asic${a}_combined.h5").safe_substitute(p=input_path, m=module)

        plot_dir = os.path.join(output_base_dir, module, temperature, "drscs", "plots", "combined")
        print("plot_dir", plot_dir)

    else:
        input_path = os.path.join(input_base_dir, module, temperature, "drscs", current, "process")
        # substitute all except current and asic
        input_template = Template("${p}/${m}_drscs_${c}_asic${a}_processed.h5").safe_substitute(p=input_path, m=module, c=current)

        plot_dir = os.path.join(output_base_dir, module, temperature, "drscs", "plots", current, "colormaps")
        print("plot_dir", plot_dir)

    # make a template out of this string to let Combine set current and asic
    input_template = Template(input_template)

    output_path = os.path.join(output_base_dir, module, temperature, "drscs", current)
    # substitute all except current and asic
    output_template = Template("${m}_drscs_${c}_asic${a}_colormap.h5").safe_substitute(m=module, c=current)
    # make a template out of this string to let Combine set current and asic
    output_template = Template(output_template)

    create_dir(plot_dir)

    if individual_plots:
        create_dir(os.path.join(plot_dir, "individual"))

    create_error_plots = False
    pixel_v_list = np.arange(29, 30)
    pixel_u_list = np.arange(1, 2)
    mem_cell_list = np.arange(1,2)

    n_processes = 8

    # generate gain map for all gain stages
    quality = False
    for gain_name in ["high", "medium", "low"]:
        obj = CreateColormaps(input_template, output_template, module, current,
                              plot_dir, pixel_v_list, pixel_u_list, mem_cell_list,
                              n_processes, gain_name, quality, individual_plots)

        obj.run()

    # generate quality map
    gain_name = "low"
    quality = True

    obj = CreateColormaps(input_template, output_template, module, current,
                          plot_dir, pixel_v_list, pixel_u_list, mem_cell_list,
                          n_processes, gain_name, quality, individual_plots)

    obj.run()
