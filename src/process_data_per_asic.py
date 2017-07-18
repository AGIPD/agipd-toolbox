import matplotlib.pyplot as plt
import numpy as np
import h5py
from string import Template
import os
import sys
import time

# from Joe Kington's answer here https://stackoverflow.com/a/4495197/3751373
def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""

    # Find the indicies of changes in "condition"
    d = np.diff(condition, n=1)
#    print("d", d)
    idx = d.nonzero()[0]
#    print("idx", idx)

    # output of np.nonzero is tuple -> convert to np.array
    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right. -JK
    idx += 1
#    print("idx", idx)

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]
#    print("start of cond: idx", idx)

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit
#    print("end of cond: idx", idx)

    # Reshape the result into two columns
    idx.shape = (-1,2)
#    print("idx", idx)
    return idx

class ProcessDrscs():
    def __init__(self, file_name, plot_dir=None, create_plots=False):

        self.file_name = file_name

        self.digital_path = "/entry/instrument/detector/data_digital"
        self.analog_path = "/entry/instrument/detector/data"

        self.generate_plots_flag = create_plots
        self.plot_dir = plot_dir

        self.digital = None
        self.analog = None

        self.percent = 10
        self.nbins = 30

        # threshold for regions which are concidered to containing no gain information
        self.thold_for_zero = 4

        self.scaling_point = 200
        self.scaling_factor = 10

        self.scaled_x_axis = None
        self.hist = None
        self.bin = None
        self.coefficient_matrix = None
        self.constant_terms = {
            "high" : None,
            "medium": None,
            "low": None
        }
        self.data_to_fit = {
            "high" : None,
            "medium": None,
            "low": None
        }
        self.slope = {
            "high" : None,
            "medium": None,
            "low": None
        }
        self.offset = {
            "high" : None,
            "medium": None,
            "low": None
        }

        plot_prefix = "M314_itestc150_asic1"
        if plot_dir is not None:
            plot_prefix = os.path.join(self.plot_dir, plot_prefix)

        self.origin_data_plot_name = Template(plot_prefix + "_${px}_${mc}_data")
        self.fitting_plot_name = Template(plot_prefix + "_${px}_${mc}_fit")
        self.combined_plot_name = Template(plot_prefix + "_${px}_${mc}_combined")
        self.plot_ending = ".png"

        self.load_data()
        self.scale_x_axis()

    def load_data(self):

        source_file = h5py.File(self.file_name, "r")

        # origin data is written as int16 which results in a integer overflow
        # when handling the scaling
        self.analog = source_file[self.analog_path][()].astype("int32")
        self.digital = source_file[self.digital_path][()].astype("int32")

        source_file.close()

    def scale_x_axis(self):
        lower  = np.arange(self.scaling_point)
        upper = (np.arange(self.scaling_point, self.digital.shape[3])
                 * self.scaling_factor
                 - self.scaling_point * self.scaling_factor
                 + self.scaling_point)

        self.scaled_x_axis = np.concatenate((lower, upper))


    def run(self, pixel_v_list, pixel_u_list, mem_cell_list, create_error_plots):
        for pixel_v in pixel_v_list:
            for pixel_u in pixel_u_list:
                for mem_cell in mem_cell_list:
                    pixel = [pixel_v, pixel_u]
                    try:
                        self.process_data_point(pixel, mem_cell)
                    except KeyboardInterrupt:
                        sys.exit(1)
                    except Exception as e:
                        print("Failed to run for pixel {} and mem_cell {}"
                              .format(pixel, mem_cell))
                        print("Error was: {}".format(e))

                        if create_error_plots:
                            try:
                                self.generate_data_plot()
                            except:
                                print("Failed to generate plot")

                        #raise

    def process_data_point(self, pixel, mem_cell):
        self.pixel = pixel
        self.mem_cell = mem_cell

        self.hist, self.bins = np.histogram(self.digital[self.pixel[0], self.pixel[1],
                                                         self.mem_cell, :],
                                            bins=self.nbins)
        #print("hist={}".format(self.hist))
        #print("bins={}".format(self.bins))

        self.calc_thresholds()

#        print("\nfitting high gain with thresholds ({}, {})".format(0, self.threshold[0]))
        self.fit_data(0, self.threshold[0], "high")

        #print("\nfitting medium gain")
        self.fit_data(self.threshold[0], self.threshold[1], "medium")

#        print("\nfitting low gain with threshholds ({}, {})".format(self.threshold[1], None))
        self.fit_data(self.threshold[1], None, "low")

        if self.generate_plots_flag:
            if self.generate_plots_flag == "data":
                self.generate_data_plot()

            elif self.generate_plots_flag == "fit":
                self.generate_fit_plot("high")
                self.generate_fit_plot("medium")
                self.generate_fit_plot("low")

            elif self.generate_plots_flag == "combined":
                self.generate_combined_plot()

            else:
                self.generate_all_plots()


    def calc_thresholds(self):
        idx = contiguous_regions(self.hist < self.thold_for_zero)

        # remove the first intervall found if it starts with the first bin
        if idx[0][0] == 0:
            idx = idx [1:]
            #print("Pixel {}, mem cell {}: Removed first intervall".format(self.pixel, self.mem_cell))

        # remove the last intervall found if it ends with the last bin
        if idx[-1][1] == self.nbins:
            idx = idx [1:]
            #print("Pixel {}, mem cell {}: Removed last intervall".format(self.pixel, self.mem_cell))

        if len(idx) < 2:
            print("thold_for_zero={}".format(self.thold_for_zero))
            print("idx={}".format(idx))
            raise Exception("Too few intervalls")
        if len(idx) > 2:
            print("thold_for_zero={}".format(self.thold_for_zero))
            print("idx={}".format(idx))
            raise Exception("Too many intervalls")

        mean_no_gain = np.mean(idx, axis=1).astype(int)
        #print("mean_no_gain={}".format(mean_no_gain))

        self.threshold = self.bins[mean_no_gain]
        #print("theshold={}".format(self.threshold))

    def fit_data(self, threshold_l, threshold_u, gain):
        data_d = self.digital[self.pixel[0], self.pixel[1], self.mem_cell, :]

        # determine condition
        if threshold_l is not None and threshold_u is not None:
            # the & will give you an elementwise and (the parentheses are necessary).
            condition = (threshold_l < data_d) & (data_d < threshold_u)
            #print("condition = ({} < data_d) & (data_d < {})".format(threshold_l, threshold_u))
        elif threshold_l is not None:
            condition = threshold_l < data_d
            #print("condition = {} < data_d".format(threshold_l))
        elif threshold_u is not None:
            condition = data_d < threshold_u
            #print("condition = data_d < {}".format(threshold_u))

        #print("condition value={}".format(condition))
        idx = contiguous_regions(condition)[0]

        # find the inner points (cut off the top and bottom part
        tmp = np.arange(idx[0], idx[1])
        lower_border = int(tmp[0] + len(tmp) * self.percent/100)
        upper_border = int(tmp[0] +  len(tmp) * (1 - self.percent/100))
        #print("lower_border = {}".format(lower_border))
        #print("upper_border = {}".format(upper_border))

        # transform the problem y = mx + c
        # into the form y = Ap, where A = [[x 1]] and p = [[m], [c]]
        # meaning  constant_terms = coefficient_matrix * [slope, offset]
        self.data_to_fit[gain] = self.analog[self.pixel[0], self.pixel[1],
                                             self.mem_cell,
                                             lower_border:upper_border]

        self.constant_terms[gain] = np.arange(lower_border, upper_border)

        # note scaling
        x = self.constant_terms[gain][np.where(self.constant_terms[gain] > self.scaling_point)]
        self.constant_terms[gain][np.where(self.constant_terms[gain] > self.scaling_point)] = (
                (x - self.scaling_point) * self.scaling_factor + self.scaling_point)

        self.coefficient_matrix = np.vstack([self.constant_terms[gain],
                                             np.ones(len(self.constant_terms[gain]))]).T

        # fit the data
        self.slope[gain], self.offset[gain] = np.linalg.lstsq(self.coefficient_matrix,
                                                              self.data_to_fit[gain])[0]
        #print("found slope: {}".format(self.slope[gain]))
        #print("found offset: {}".format(self.offset[gain]))

    def write_data(self):
        pass

    def generate_data_plot(self):

        width = 0.7 * (self.bins[1] - self.bins[0])
        center = (self.bins[:-1] + self.bins[1:]) / 2

        # plot data
        fig = plt.figure(figsize=(13, 5))
        ax = fig.add_subplot(121)
        ax.bar(center, self.hist, align="center", width=width, label="hist digital")
        ax.legend()

        ax = fig.add_subplot(122)
        ax.plot(self.scaled_x_axis,
                 self.analog[self.pixel[0], self.pixel[1], self.mem_cell, :],
                 ".", markersize=0.5, label="analog")
        ax.plot(self.scaled_x_axis,
                 self.digital[self.pixel[0], self.pixel[1], self.mem_cell, :],
                 ".", markersize=0.5, label="digital")

        ax.legend()
        prefix = self.origin_data_plot_name.substitute(px=self.pixel,
                                                       # fill number to be of 3 digits
                                                       mc=str(self.mem_cell).zfill(3))
        fig.savefig("{}{}".format(prefix, self.plot_ending))
        fig.clf()
        plt.close(fig)


    def generate_fit_plot(self, gain):
        # plot fitting
        fig = plt.figure(figsize=None)
        plt.plot(self.constant_terms[gain],
                 self.data_to_fit[gain],
                 'o', label="Original data", markersize=1)

        plt.plot(self.constant_terms[gain],
                 self.slope[gain] * self.constant_terms[gain] + self.offset[gain],
                 "r", label="Fitted line high")

        plt.legend()
        prefix = self.fitting_plot_name.substitute(px=self.pixel,
                                                   # fill number to be of 3 digits
                                                   mc=str(self.mem_cell).zfill(3))
        fig.savefig("{}_{}{}".format(prefix, gain, self.plot_ending))
        fig.clf()
        plt.close(fig)

    def generate_combined_plot(self):
        # plot combined
        fig = plt.figure(figsize=None)
        plt.plot(self.scaled_x_axis,
                 self.analog[self.pixel[0], self.pixel[1], self.mem_cell, :],
                 ".", markersize=0.5, label="analog")

        plt.plot(self.scaled_x_axis,
                 self.digital[self.pixel[0], self.pixel[1], self.mem_cell, :],
                 ".", markersize=0.5, label="digital")

        plt.plot(self.constant_terms["high"],
                 self.slope["high"] * self.constant_terms["high"] + self.offset["high"],
                 "r", label="Fitted line high")

        plt.plot(self.constant_terms["medium"],
                 self.slope["medium"] * self.constant_terms["medium"] + self.offset["medium"],
                 "r", label="Fitted line medium")

        plt.plot(self.constant_terms["low"],
                 self.slope["low"] * self.constant_terms["low"] + self.offset["low"],
                 "r", label="Fitted line low")

        plt.legend()
        prefix = self.combined_plot_name.substitute(px=self.pixel,
                                                    # fill number to be of 3 digits
                                                    mc=str(self.mem_cell).zfill(3))
        fig.savefig("{}{}".format(prefix, self.plot_ending))
        fig.clf()
        plt.close(fig)

    def generate_all_plots(self):

        self.generate_data_plot()

        self.generate_fit_plot("high")
        self.generate_fit_plot("medium")
        self.generate_fit_plot("low")

        self.generate_combined_plot()

if __name__ == "__main__":

    file_name = "/gpfs/cfel/fsds/labs/processed/calibration/processed/M314/temperature_m15C/drscs/itestc150/M314_drscs_itestc150_asic1.h5"
    plot_dir = "/gpfs/cfel/fsds/labs/processed/calibration/processed/M314/temperature_m15C/drscs/plots/itestc150"

    create_error_plots = True
    pixel_v_list = np.arange(64)
    pixel_u_list = np.arange(64)
    mem_cell_list = np.arange(352)
    #pixel_v_list = np.arange(1)
    #pixel_u_list = np.arange(2, 3)
    #mem_cell_list = np.arange(60, 61)

    cal = ProcessDrscs(file_name, plot_dir=plot_dir, create_plots="all")

    cal.run(pixel_v_list, pixel_u_list, mem_cell_list, create_error_plots)
