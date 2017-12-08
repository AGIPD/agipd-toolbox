from __future__ import print_function

import os
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
                        default="/gpfs/cfel/fsds/labs/agipd/calibration/"
                                "processed",
                        help="Directory to get data from")
    parser.add_argument("--output_dir",
                        type=str,
                        default="/gpfs/cfel/fsds/labs/agipd/calibration/"
                                "processed",
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
                        help="Current to use (e.g. itestc20) or merged")
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
            print("Dir '{0}' does not exist. Create it."
                  .format(directory_name))
        except IOError:
            if os.path.isdir(directory_name):
                pass


def generate_matrix(result, gain_name, matrix_type):
    gain_d = {
        "high": 0,
        "medium": 1,
        "low": 2
    }
    gain = gain_d[gain_name]

    matrix = {
        "slope": None,
        "offset": None,
        "average_residual": None,
    }

    # TODO generic (get from file)
    n_used_fits = 3

    # gain, pixel_v, pixel_u, mem_cell
    # get them from the slope entry (offset and residual are of the same shape)
    n_pixel_v, n_pixel_u, n_mem_cell = result["slope"]["mean"].shape[1:]

    for key in matrix:
        # create array
        matrix[key] = np.empty((n_pixel_v, n_pixel_u, n_mem_cell))
        # intiate with nan
        matrix[key][...] = np.NAN

    if matrix_type == "quality":
        for v in np.arange(n_pixel_v):
            for u in np.arange(n_pixel_u):
                for mem_cell in np.arange(n_mem_cell):

                    for key in ["slope", "offset"]:
                        mean = result[key]["mean"]
                        indv = result[key]["individual"][gain_name]

                        mean = mean[gain, v, u, mem_cell]
                        values = indv[v, u, mem_cell][1:-1]

                        try:
                            if mean != 0:
                                diff = values - mean
                                scale = n_used_fits * mean

                                matrix[key][v, u, mem_cell] = (
                                    np.linalg.norm(diff) / np.absolute(scale))
                        except Exception as e:
                            print("Error was: ", e)
                            print("pixel {}, mem_cell {}"
                                  .format(mem_cell, v, u))
                            print(key)
                            print("mean={}".format(mean))
                            print("values={}".format(values))
    else:
        for key in ["slope", "offset", "average_residual"]:
            matrix[key] = result[key]["mean"][gain, ...]

            idx = np.where(result["error_code"] != 0)
            matrix[key][idx] = np.NAN

    return matrix


def create_individual_plots(input_fname, mem_cell_list, plot_prefix,
                            plot_ending, gain_name, matrix_type):
    try:
        colormap_matrix, safety_factor = create_matrix_individual(input_fname,
                                                                  gain_name,
                                                                  matrix_type)

        create_plots(mem_cell_list,
                     colormap_matrix,
                     plot_prefix,
                     plot_ending,
                     gain_name,
                     matrix_type,
                     safety_factor,
                     splitted=True)
    except OSError:
        print("OSError:", input_fname)
        pass


def create_matrix_individual(input_fname, gain_name, matrix_type):
    process_result = {
        "slope": {
            "mean": None,
            "individual": {
                "high": None,
                "medium": None,
                "low": None
            }
        },
        "offset": {
            "mean": None,
            "individual": {
                "high": None,
                "medium": None,
                "low": None
            }
        },
        "average_residual": {
            "mean": None,
            "individual": {
                "high": None,
                "medium": None,
                "low": None
            }
        },
    }

    # read in data
    source_file = h5py.File(input_fname, "r")

    for key in process_result.keys():
        r = process_result[key]
        low_path = "/{}/individual/low".format(key)

        r["mean"] = source_file["/{}/mean".format(key)][()]
        r["individual"]["low"] = source_file[low_path][()]

    process_result["error_code"] = source_file["/error_code"][()]
    safety_factor = source_file["/collection/safety_factor"][()]
    source_file.close()

    # calcurate matrix
    return (generate_matrix(process_result, gain_name, matrix_type),
            safety_factor)


def create_plots(mem_cell_list, colormap_matrix, plot_file_prefix, plot_ending,
                 gain_name, matrix_type, safety_factor, splitted=False):
    plot_size = (27, 7)
    #             [left, bottom, width, height]
    ax_location = [0.91, 0.11, 0.01, 0.75]
    ax2_location = [0.94, 0.11, 0.01, 0.75]

    for mem_cell in mem_cell_list:
        # fill number to be of 3 digits
        for key in colormap_matrix:
            m = colormap_matrix[key][..., mem_cell]

            if splitted and matrix_type == "quality" \
                    and np.where(m >= 1)[0].size != 0:

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
            plt.suptitle("{}_{} {} sf{}".format(title_prefix,
                                                gain_name,
                                                key,
                                                safety_factor),
                         fontsize=24)

            fig.savefig("{}_{}_{}_sf{}{}".format(plot_file_prefix,
                                                 gain_name, key,
                                                 safety_factor,
                                                 plot_ending),
                        bbox_inches='tight')
            fig.clf()
            plt.close(fig)


class CreateColormaps():
    def __init__(self, input_template, output_template, module, current,
                 plot_dir, n_processes, gain_name, matrix_type, mem_cell_list,
                 individual_plots=False):

        self.plot_dir = plot_dir
        self.gain_name = gain_name
        self.matrix_type = matrix_type
        self.mem_cell_list = mem_cell_list

        self.module = module
        self.current = current

        self.plot_ending = ".png"

        self.input_template = input_template
        self.output_template = output_template

        self.asics_in_upper_row = np.arange(16, 8, -1)
        self.asics_in_lower_row = np.arange(1, 9)

        self.asic_list = np.concatenate((self.asics_in_upper_row,
                                         self.asics_in_lower_row))
        self.asic_list = np.arange(1, 17)

        self.individual_plots = individual_plots

        self.n_processes = n_processes

        self.pool = Pool(processes=self.n_processes)

        if matrix_type == "quality":
            self.colormap_matrix = {
                "slope": None,
                "offset": None,
            }
        else:
            self.colormap_matrix = {
                "slope": None,
                "offset": None,
                "average_residual": None
            }

    def run(self):
        for mem_cell in self.mem_cell_list:

            print("\nStarted at", str(datetime.datetime.now()))
            t = time.time()

            if self.individual_plots:
                # substitute all except asic
                self.plot_template = (
                    Template("${p}/individual/${m}_${c}_asic${a}_${mc}_${mt}")
                    .safe_substitute(p=plot_dir,
                                     m=self.module,
                                     c=self.current,
                                     mc=str(mem_cell).zfill(3),
                                     mt=self.matrix_type))
                # make a template out of this string
                self.plot_template = Template(self.plot_template)
                self.get_data_individual()
            else:
                self.plot_prefix = ("{}/{}_{}_{}_{}"
                                    .format(plot_dir,
                                            self.module,
                                            self.current,
                                            str(mem_cell).zfill(3),
                                            self.matrix_type))

                self.get_data()
                create_plots(self.mem_cell_list,
                             self.colormap_matrix,
                             self.plot_prefix,
                             self.plot_ending,
                             self.gain_name,
                             self.matrix_type,
                             self.safety_factor,
                             splitted=True)

            print("\nFinished at {} after {}"
                  .format(datetime.datetime.now(), time.time() - t))

    def get_data_individual(self):

        result_list = []

        print("matrix_type {}".format(matrix_type))
        for asic in self.asic_list:
            plot_prefix = self.plot_template.substitute(a=asic)

            # the input files for processing are the output ones from gather
            input_fname = self.input_template.substitute(a=str(asic).zfill(2))
            print("input_file {}".format(input_fname))

            result_list.append(
                self.pool.apply_async(
                    create_individual_plots,
                    (input_fname,
                     self.mem_cell_list,
                     plot_prefix,
                     self.plot_ending,
                     self.gain_name,
                     self.matrix_type)))

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
        print("asic_list={}".format(asic_list))

        generated_matrix = [[] for asic in asic_list]

        matrix = {
            "slope": None,
            "offset": None,
            "average_residual": None
        }
        result_list = []

        print("matrix_type {}".format(matrix_type))
        for asic in asic_list:

            # the input files for processing are the output ones from gather
            input_fname = self.input_template.substitute(a=str(asic).zfill(2))
            print("input_file {}".format(input_fname))

            # calcurate matrix
            result_list.append(
                self.pool.apply_async(create_matrix_individual,
                                      (input_fname,
                                       self.gain_name,
                                       self.matrix_type)))

        # build matrix for whole module
        for i in range(asic_list.size):

            generated_matrix, self.safety_factor = result_list[i].get()

            if matrix["slope"] is None:
                for key in matrix.keys():
                    matrix[key] = generated_matrix[key]

            else:
                for key in matrix.keys():
                    matrix[key] = np.concatenate(
                        (matrix[key], generated_matrix[key]), axis=1)

        return matrix


if __name__ == "__main__":

    args = get_arguments()

    input_base_dir = args.input_dir
    output_base_dir = args.output_dir
    module = args.module
    temperature = args.temperature
    current = args.current
    individual_plots = args.individual_plots

    print("Configured parameter:")
    print("input_dir: ", input_base_dir)
    print("output_dir: ", output_base_dir)
    print("module: ", module)
    print("temperature: ", temperature)
    print("current: ", current)
    print("individual_plots: ", individual_plots)

    if current == "merged":
        input_path = os.path.join(input_base_dir,
                                  module,
                                  temperature,
                                  "drscs",
                                  "merged")
        # substitute all except current and asic
        input_template = (
            Template("${p}/${m}_drscs_asic${a}_merged.h5")
            .safe_substitute(p=input_path, m=module)
        )

    else:
        input_path = os.path.join(input_base_dir,
                                  module,
                                  temperature,
                                  "drscs",
                                  current,
                                  "process")
        # substitute all except current and asic
        input_template = (
            Template("${p}/${m}_drscs_${c}_asic${a}_processed.h5")
            .safe_substitute(p=input_path, m=module, c=current)
        )

    plot_dir = os.path.join(output_base_dir,
                            module,
                            temperature,
                            "drscs",
                            "plots",
                            current,
                            "colormaps")
    print("plot_dir", plot_dir)

    # make a template out of this string to let Combine set current and asic
    input_template = Template(input_template)

    output_path = os.path.join(output_base_dir,
                               module,
                               temperature,
                               "drscs",
                               current)
    # substitute all except current and asic
    output_template = (
        Template("${m}_drscs_${c}_asic${a}_colormap.h5")
        .safe_substitute(m=module, c=current)
    )
    # make a template out of this string to let Combine set current and asic
    output_template = Template(output_template)

    create_dir(plot_dir)

    if individual_plots:
        create_dir(os.path.join(plot_dir, "individual"))

    create_error_plots = False
    mem_cell_list = np.arange(1, 2)

    n_processes = 8

    # generate gain map for all gain stages
    matrix_type = "gain"
    for gain_name in ["high", "medium", "low"]:
        obj = CreateColormaps(input_template, output_template, module, current,
                              plot_dir, n_processes, gain_name, matrix_type,
                              mem_cell_list, individual_plots)

        obj.run()

    # generate quality map
    gain_name = "low"
    matrix_type = "quality"

    obj = CreateColormaps(input_template, output_template, module, current,
                          plot_dir, n_processes, gain_name, matrix_type,
                          mem_cell_list, individual_plots)

    obj.run()
