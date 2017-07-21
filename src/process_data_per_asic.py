import matplotlib.pyplot as plt
import numpy as np
import h5py
from string import Template
import os
import sys
import time
import traceback

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
    def __init__(self, asic, input_file_name, output_file_name,
                 plot_prefix=None, plot_dir=None, create_plots=False):

        self.input_fname = input_file_name
        self.output_fname = output_file_name

        self.digital_path = "/entry/instrument/detector/data_digital"
        self.analog_path = "/entry/instrument/detector/data"

        self.create_plot_type = create_plots
        self.plot_dir = plot_dir

        self.digital = None
        self.analog = None

        self.percent = 10
        self.nbins = 30

        # threshold for regions which are concidered to containing no gain information
        self.thold_for_zero = 4

        self.scaling_point = 200
        self.scaling_factor = 10

        self.n_gain_stages = 3
        # in how many the base interval should be split
        self.n_intervals = {
            "high": 1,
            "medium": 1,
            "low": 5
        }
        self.n_zero_region_stored = 5
        self.fit_cutoff_left = 1
        self.fit_cutoff_right = 1

        self.pixel = None
        self.mem_cell = None
        self.gain_idx = None
        self.scaled_x_values = None
        self.hist = None
        self.bin = None
        self.coefficient_matrix = None
        self.x_values = {
            "high" : [[] for _ in np.arange(self.n_intervals["high"])],
            "medium": [[] for _ in np.arange(self.n_intervals["medium"])],
            "low": [[] for _ in np.arange(self.n_intervals["low"])]
        }
        self.data_to_fit = {
            "high" : [[] for _ in np.arange(self.n_intervals["high"])],
            "medium": [[] for _ in np.arange(self.n_intervals["medium"])],
            "low": [[] for _ in np.arange(self.n_intervals["low"])]
        }

        # temporarily which can not be stored in the result directly
        self.slope = {
            "high" : [[] for _ in np.arange(self.n_intervals["high"])],
            "medium": [[] for _ in np.arange(self.n_intervals["medium"])],
            "low": [[] for _ in np.arange(self.n_intervals["low"])]
        }
        self.residuals = {
            "high" : [[] for _ in np.arange(self.n_intervals["high"])],
            "medium": [[] for _ in np.arange(self.n_intervals["medium"])],
            "low": [[] for _ in np.arange(self.n_intervals["low"])]
        }
        self.offset = {
            "high" : [[] for _ in np.arange(self.n_intervals["high"])],
            "medium": [[] for _ in np.arange(self.n_intervals["medium"])],
            "low": [[] for _ in np.arange(self.n_intervals["low"])]
        }
        self.medians = {
            "high" : None,
            "medium": None,
            "low": None
        }
        # the information is stored in pixel and memory cell independent
        # objects because np.array needs the size fixed but the number of
        # intervals is not fixed
        self.intervals = {
            "zero_regions": None,
            "gain_stages": {
                "high" : None,
                "medium": None,
                "low": None
            },
            "subintervals": {
                "high" : None,
                "medium": None,
                "low": None
            }
        }
        self.thresholds = None

        self.result = {
            "slope": {
                "mean": None,
                "individual": None
            },
            "offset": {
                "mean": None,
                "individual": None
            },
            "residuals": {
                "mean": None,
                "individual": None
            },
            "intervals": {
                "zero_regions": None,
                "gain_stages": None,
                "subintervals": {
                    "high" : None,
                    "medium": None,
                    "low": None
                }
            },
            "medians": None,
            "thresholds": None,
            "error_code": None,
            "warning_code": None
        }

        ###
        # error_codes:
        # 1    Uunknown error
        # 2    Zero region: too few intervals
        # 3    Zero region: too many intervals
        #
        # warning_codes:
        #
        ###


        self.colormap_matrix = {
            "slope": None,
            "offset": None
        }

        if plot_prefix:
            self.plot_prefix = "{}{}".format(plot_prefix, str(asic).zfill(2))
            if plot_dir is not None:
                plot_prefix = os.path.join(self.plot_dir, plot_prefix)

            self.plot_name = {
                "origin_data": Template(plot_prefix + "_${px}_${mc}_data"),
                "fit": Template(plot_prefix + "_${px}_${mc}_fit"),
                "combined": Template(plot_prefix + "_${px}_${mc}_combined"),
                "colormap": Template(plot_prefix + "_${mc}_colormap"),
            }
            self.plot_ending = ".png"
        else:
            self.plot_name = None
            self.plot_ending = None


        print("Load data")
        t = time.time()
        self.load_data()
        print("took time: {}".format(time.time() - t))
        self.scale_full_x_axis()

    def load_data(self):

        source_file = h5py.File(self.input_fname, "r")

        # origin data is written as int16 which results in a integer overflow
        # when handling the scaling
        self.analog = source_file[self.analog_path][()].astype("int32")
        self.digital = source_file[self.digital_path][()].astype("int32")

        source_file.close()

    def run(self, pixel_v_list, pixel_u_list, mem_cell_list,
            create_error_plots, create_colormaps, write_data=True):

        if len(pixel_v_list) == 0 or len(pixel_u_list) == 0:
            print("pixel_v_list", pixel_v_list)
            print("pixel_u_list", pixel_u_list)
            raise Exception("No proper pixel specified")
        if len(mem_cell_list) == 0:
            print("mem_cell_list", mem_cell_list)
            raise Exception("No proper memory cell specified")

        if (create_error_plots or create_colormaps) and self.plot_name is None:
            raise Exception("Plotting was not defined on initiation. Quitting.")

        self.pixel_v_list = pixel_v_list
        self.pixel_u_list = pixel_u_list
        self.mem_cell_list = mem_cell_list

        # initiate
        # +1 because counting starts with zero
        shape_tmp = (pixel_v_list.max() + 1, pixel_u_list.max() + 1, mem_cell_list.max() + 1)

        shape = (self.n_gain_stages, ) + shape_tmp
        print("result shape: {}".format(shape))

        # initiate fit results
        for key in ["slope", "offset", "residuals"]:
            self.result[key]["mean"] = np.zeros(shape, np.float32)
            self.result[key]["individual"] = {
                "high": np.zeros(shape_tmp + (self.n_intervals["high"],)),
                "medium": np.zeros(shape_tmp + (self.n_intervals["low"],)),
                "low": np.zeros(shape_tmp + (self.n_intervals["low"],)),
            }
        self.result["medians"] = np.zeros(shape, np.float32)
        self.result["error_code"] = np.zeros(shape_tmp, np.int16)
        self.result["warning_code"] = np.zeros(shape_tmp, np.int16)

        # intiate intervals
        # interval consists of start and end point -> x2
        self.result["intervals"]["gain_stages"] = self.init_with_nan(shape + (2,))
        self.result["intervals"]["zero_regions"] = (
            self.init_with_nan(shape_tmp + (self.n_zero_region_stored, 2)))
        for key in ["high", "medium", "low"]:
            self.result["intervals"]["subintervals"][key] = (
                self.init_with_nan(shape_tmp + (self.n_intervals[key], 2)))

        # initiate thresholds
        # thresholds between to distinguish between the gain stages
        threshold_shape = (self.n_gain_stages - 1,) + shape_tmp
        print("threshold shape: {}".format(threshold_shape))
        self.result["thresholds"] = np.zeros(threshold_shape, np.float32)

        for pixel_v in self.pixel_v_list:
            for pixel_u in self.pixel_u_list:
                for mem_cell in self.mem_cell_list:
                    try:
                        self.pixel = [pixel_v, pixel_u]
                        self.mem_cell = mem_cell
                        self.current_idx = (pixel_v, pixel_u, mem_cell)
                        gain_idx = {
                            "high": (0, ) + self.current_idx,
                            "medium": (1, ) + self.current_idx,
                            "low": (2, ) + self.current_idx
                        }

                        self.process_data_point(self.pixel, self.mem_cell)

                        try:
                            self.result["intervals"]["zero_regions"][self.current_idx] = self.intervals["zero_regions"]
                        except ValueError:
                            l = len(self.intervals["zero_regions"])
                            zero_regions = self.result["intervals"]["zero_regions"][self.current_idx]
                            if l < self.n_zero_region_stored:
                                zero_regions[:l] = self.intervals["zero_regions"]
                            else:
                                zero_regions = self.intervals["zero_regions"][:n_zero_region_stored]

                        for gain in gain_idx:
                            self.result["slope"]["individual"][gain][self.current_idx] = self.slope[gain]
                            self.result["offset"]["individual"][gain][self.current_idx] = self.offset[gain]
                            self.result["residuals"]["individual"][gain][self.current_idx] = self.residuals[gain]

                            self.result["intervals"]["gain_stages"][gain_idx[gain]] = self.intervals["gain_stages"][gain]
                            self.result["intervals"]["subintervals"][gain][self.current_idx] = self.intervals["subintervals"][gain]

                            self.result["medians"][gain_idx[gain]] = self.medians[gain]

                            l = self.fit_cutoff_left
                            r = self.fit_cutoff_right
                            if len(self.slope[gain]) >= l + r + 1:
                                self.result["slope"]["mean"][gain_idx[gain]] = np.mean(self.slope[gain][l:-r])
                                self.result["offset"]["mean"][gain_idx[gain]] = np.mean(self.offset[gain][l:-r])
                                self.result["residuals"]["mean"][gain_idx[gain]] = np.mean(self.residuals[gain][l:-r])
                            else:
                                self.result["slope"]["mean"][gain_idx[gain]] = np.mean(self.slope[gain])
                                self.result["residuals"]["mean"][gain_idx[gain]] = np.mean(self.residuals[gain])


                        # store the thresholds
                        for i in np.arange(len(self.thresholds)):
                            self.result["thresholds"][(i, ) + self.current_idx] = self.thresholds[i]

                    except KeyboardInterrupt:
                        sys.exit(1)
                    except Exception as e:
                        print("Failed to run for pixel {} and mem_cell {}"
                              .format(self.pixel, self.mem_cell))
                        print(traceback.format_exc())

                        if not self.result["error_code"][self.current_idx]:
                            self.result["error_code"][self.current_idx] = 1

                        if create_error_plots:
                            try:
                                self.generate_data_plot()
                            except:
                                print("Failed to generate plot")
                                raise

                        #raise

        if create_colormaps:
            if type(create_colormaps) == list and "matrix" in create_colormaps:
                self.generate_colormap_matrix()
            else:
                self.generate_colormap_matrix()
                self.plot_colormap()

        if write_data:
            self.write_data()

    def init_with_nan(self, shape):
        # create array
        obj = np.empty(shape)
        # intiate with nan
        obj[...] = np.NAN

        return obj

    def process_data_point(self, pixel, mem_cell):
        self.hist, self.bins = np.histogram(self.digital[self.pixel[0], self.pixel[1],
                                                         self.mem_cell, :],
                                            bins=self.nbins)
        #print("hist={}".format(self.hist))
        #print("bins={}".format(self.bins))

        self.calc_thresholds()

        self.intervals["gain_stages"]["high"], self.intervals["subintervals"]["high"] = (
            self.fit_gain("high", 0, self.thresholds[0], cut_off_ends=False))

        self.intervals["gain_stages"]["medium"], self.intervals["subintervals"]["medium"] = (
            self.fit_gain("medium", self.thresholds[0], self.thresholds[1], cut_off_ends=False))

        self.intervals["gain_stages"]["low"], self.intervals["subintervals"]["low"] = (
            self.fit_gain("low", self.thresholds[1], None, cut_off_ends=False))

        self.calc_gain_median()

        if self.create_plot_type:
            print("\nGenerate plots")
            t = time.time()

            if "data" in self.create_plot_type:
                self.generate_data_plot()

            elif "fit" in self.create_plot_type:
                self.generate_fit_plot("high")
                self.generate_fit_plot("medium")
                self.generate_fit_plot("low")

            elif "combined" in self.create_plot_type:
                self.generate_combined_plot()

            else:
                self.generate_all_plots()

            print("took time: {}".format(time.time() - t))

    def calc_thresholds(self):
        self.intervals["zero_regions"] = contiguous_regions(self.hist < self.thold_for_zero)

        # remove the first interval found if it starts with the first bin
        if self.intervals["zero_regions"][0][0] == 0:
            self.intervals["zero_regions"] = self.intervals["zero_regions"][1:]
            #print("Pixel {}, mem cell {}: Removed first interval".format(self.pixel, self.mem_cell))

        # remove the last interval found if it ends with the last bin
        if self.intervals["zero_regions"][-1][1] == self.nbins:
            self.intervals["zero_regions"] = self.intervals["zero_regions"][1:]
            #print("Pixel {}, mem cell {}: Removed last interval".format(self.pixel, self.mem_cell))


        if len(self.intervals["zero_regions"]) < 2:
            print("thold_for_zero={}".format(self.thold_for_zero))
            print("intervals={}".format(self.intervals["zero_regions"]))
            self.result["error_code"][self.current_idx] = 2
            raise Exception("Too few intervals")
        if len(self.intervals["zero_regions"]) > 2:
            print("thold_for_zero={}".format(self.thold_for_zero))
            print("intervals={}".format(self.intervals["zero_regions"]))
            self.result["error_code"][self.current_idx] = 3
            raise Exception("Too many intervals")

        #print("zero_regions", self.intervals["zero_regions"])

        mean_zero_region = np.mean(self.intervals["zero_regions"], axis=1).astype(int)
        #print("mean_zero_region={}".format(mean_zero_region))

        self.thresholds = self.bins[mean_zero_region]
        #print("thesholds={}".format(self.thresholds))

    def determine_fit_interval(self, threshold_l, threshold_u):
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
        return contiguous_regions(condition)

    def split_interval(self, interval, nsplits):
        """
        split given interval into equaly sized sub-intervals
        """
        #print("nsplits", nsplits)
        #print("interval={}".format(interval))

        step = int((interval[-1] - interval[0]) / nsplits)

        splitted_intervals = []
        for i in np.arange(nsplits):
            splitted_intervals.append(
                [interval[0] + i * step, interval[0] + (i + 1) * step])

        # due to rounding or int convertion the last calculated interval ends
        # before the end of the given interval
        splitted_intervals[-1][1] = interval[1]

        #print("splitted_intervals={}".format(splitted_intervals))

        return splitted_intervals

    def calc_gain_median(self):
        for gain in ["high", "medium", "low"]:
            start = self.intervals["gain_stages"][gain][0]
            stop = self.intervals["gain_stages"][gain][1]

            self.medians[gain] = np.median(self.digital[self.pixel[0], self.pixel[1],
                                                        self.mem_cell, start:stop])

    def fit_gain(self, gain, threshold_l, threshold_u, cut_off_ends):
#        print("\nfitting {} gain with threshholds ({}, {})".format(gain, threshold_l, threshold_u))
        interval_tmp = self.determine_fit_interval(threshold_l, threshold_u)
        #TODO: verify if this is really the right approach
        # sometimes there is more than one interval returned, e.g. [[ 370,  372],[ 373, 1300]]
        # combine it
        interval = [interval_tmp[0][0], interval_tmp[-1][-1]]

        # the interval is splitted into sub-intervals
        intervals = self.split_interval(interval, self.n_intervals[gain])

        for i in np.arange(len(intervals)):
            self.fit_data(gain, intervals[i], i, cut_off_ends=cut_off_ends)

        return interval, intervals

    def fit_data(self, gain, interval, interval_idx, cut_off_ends=False):

        # find the inner points (cut off the top and bottom part
        if cut_off_ends:
            tmp = np.arange(interval[0], interval[1])
            # 100.0 is needed because else it casts it as ints, i.e. 10/100=>0
            lower_border = int(tmp[0] + len(tmp) * self.percent/100.0)
            upper_border = int(tmp[0] + len(tmp) * (1 - self.percent/100.0))
            #print("lower_border = {}".format(lower_border))
            #print("upper_border = {}".format(upper_border))
        else:
            lower_border = interval[0]
            upper_border = interval[1]

        # transform the problem y = mx + c
        # into the form y = Ap, where A = [[x 1]] and p = [[m], [c]]
        # meaning  data_to_fit = coefficient_matrix * [slope, offset]
        self.data_to_fit[gain][interval_idx] = self.analog[self.pixel[0], self.pixel[1],
                                                           self.mem_cell,
                                                           lower_border:upper_border]

        self.x_values[gain][interval_idx] = np.arange(lower_border, upper_border)

        # scaling
        self.scale_x_interval(gain, interval_idx)

        # .T means transposed
        self.coefficient_matrix = np.vstack([self.x_values[gain][interval_idx],
                                             np.ones(len(self.x_values[gain][interval_idx]))]).T

        # fit the data
        # reason to use numpy lstsq:
        # https://stackoverflow.com/questions/29372559/what-is-the-difference-between-numpy-linalg-lstsq-and-scipy-linalg-lstsq
        #lstsq returns: Least-squares solution (i.e. slope and offset), residuals, rank, singular values
        try:

            res = np.linalg.lstsq(self.coefficient_matrix, self.data_to_fit[gain][interval_idx])
            self.slope[gain][interval_idx], self.offset[gain][interval_idx] = res[0]
            self.residuals[gain][interval_idx] = res[1]
        except:
            #print("interval\n{}".format(interval))
            #print("self.coefficient_matrix\n{}".format(self.coefficient_matrix))
            #print("self.data_to_fit[{}][{}]\n{}"
            #      .format(gain, interval_idx, self.data_to_fit[gain][interval_idx]))
            raise

        #print("found slope: {}".format(self.slope[gain]))
        #print("found offset: {}".format(self.offset[gain]))

    def scale_full_x_axis(self):
        lower  = np.arange(self.scaling_point)
        upper = (np.arange(self.scaling_point, self.digital.shape[3])
                 * self.scaling_factor
                 - self.scaling_point * self.scaling_factor
                 + self.scaling_point)

        self.scaled_x_values = np.concatenate((lower, upper))

    def scale_x_interval(self, gain, interval_idx):
        # tmp variables to improve reading
        indices_to_scale = np.where(self.x_values[gain][interval_idx] > self.scaling_point)
        x = self.x_values[gain][interval_idx][indices_to_scale]
        # shift x value to root, scale, shift back
        # e.g. x_new = (x_old - 200) * 10 + 200
        self.x_values[gain][interval_idx][indices_to_scale] = (
            (x - self.scaling_point) * self.scaling_factor + self.scaling_point)

    def write_data(self):
        save_file = h5py.File(self.output_fname, "w", libver="latest")

        try:
            print("\nStart saving data")
            t = time.time()

            for key in self.result:
                if type(self.result[key]) != dict:
                    save_file.create_dataset("/{}".format(key), data=self.result[key])
                else:
                    for subkey in self.result[key]:
                        if type(self.result[key][subkey]) != dict:
                            save_file.create_dataset("/{}/{}".format(key, subkey),
                                                     data=self.result[key][subkey])
                        else:
                            for gain in ["high", "medium", "low"]:
                                save_file.create_dataset("/{}/{}/{}".format(key, subkey, gain),
                                                         data=self.result[key][subkey][gain])


            for key in self.colormap_matrix:
                if self.colormap_matrix[key] is not None:
                    save_file.create_dataset("/colormap_matrix/{}".format(key),
                                             data=self.colormap_matrix[key])

            save_file.flush()
            print("took time: {}".format(time.time() - t))
        finally:
            save_file.close()

    ################
    ### Plotting ###
    ################

    def generate_data_plot(self):

        width = 0.7 * (self.bins[1] - self.bins[0])
        center = (self.bins[:-1] + self.bins[1:]) / 2

        # plot data
        fig = plt.figure(figsize=(13, 5))
        ax = fig.add_subplot(121)
        ax.bar(center, self.hist, align="center", width=width, label="hist digital")
        ax.legend()

        ax = fig.add_subplot(122)
        ax.plot(self.scaled_x_values,
                self.analog[self.pixel[0], self.pixel[1], self.mem_cell, :],
                ".", markersize=0.5, label="analog")
        ax.plot(self.scaled_x_values,
                self.digital[self.pixel[0], self.pixel[1], self.mem_cell, :],
                ".", markersize=0.5, label="digital")

        ax.legend()
        prefix = self.plot_name["origin_data"].substitute(px=self.pixel,
                                                       # fill number to be of 3 digits
                                                       mc=str(self.mem_cell).zfill(3))
        fig.savefig("{}{}".format(prefix, self.plot_ending))
        fig.clf()
        plt.close(fig)

    def remove_legend_dubplicates(self):
        # Remove duplicates in legend
        # https://stackoverflow.com/questions/26337493/pyplot-combine-multiple-line-labels-in-legend
        handles, labels = plt.gca().get_legend_handles_labels()
        i =1
        while i<len(labels):
            if labels[i] in labels[:i]:
                del(labels[i])
                del(handles[i])
            else:
                i +=1

        return handles, labels

    def generate_fit_plot(self, gain):
        # plot fitting
        fig = plt.figure(figsize=None)
        for i in np.arange(len(self.x_values[gain])):
            plt.plot(self.x_values[gain][i],
                     self.data_to_fit[gain][i],
                     'o', color="C0", label="Original data", markersize=1)

            plt.plot(self.x_values[gain][i],
                     self.slope[gain][i] * self.x_values[gain][i] + self.offset[gain][i],
                     "r", label="Fitted line ({} intervals)".format(self.n_intervals[gain]))

        handles, labels = self.remove_legend_dubplicates()
        plt.legend(handles, labels)

        prefix = self.plot_name["fit"].substitute(px=self.pixel,
                                                   # fill number to be of 3 digits
                                                   mc=str(self.mem_cell).zfill(3))
        fig.savefig("{}_{}{}".format(prefix, gain, self.plot_ending))
        fig.clf()
        plt.close(fig)

    def generate_combined_plot(self):
        # plot combined
        fig = plt.figure(figsize=None)
        plt.plot(self.scaled_x_values,
                 self.analog[self.pixel[0], self.pixel[1], self.mem_cell, :],
                 ".", markersize=0.5, label="analog")

        plt.plot(self.scaled_x_values,
                 self.digital[self.pixel[0], self.pixel[1], self.mem_cell, :],
                 ".", markersize=0.5, label="digital")

        for gain in ["high", "medium", "low"]:
            # if there are multiple sub-fits some are cut off
            if len(self.x_values[gain]) >= self.fit_cutoff_left + self.fit_cutoff_right + 1:
                # display the cut off fits on the left side
                for i in np.arange(self.fit_cutoff_left):
                    plt.plot(self.x_values[gain][i],
                             self.slope[gain][i] * self.x_values[gain][i] + self.offset[gain][i],
                             "r", alpha=0.3, label="Fitted line (unused)")
                # display the used fits
                for i in np.arange(self.fit_cutoff_left, len(self.x_values[gain]) - self.fit_cutoff_right):
                    plt.plot(self.x_values[gain][i],
                             self.slope[gain][i] * self.x_values[gain][i] + self.offset[gain][i],
                             "r", label="Fitted line")
                # display the cut off fits on the right side
                for i in np.arange(len(self.x_values[gain]) - self.fit_cutoff_right,
                                   len(self.x_values[gain])):
                    plt.plot(self.x_values[gain][i],
                             self.slope[gain][i] * self.x_values[gain][i] + self.offset[gain][i],
                             "r", alpha=0.3, label="Fitted line (unused)")
            else:
                # display all fits
                for i in np.arange(len(self.x_values[gain])):
                    plt.plot(self.x_values[gain][i],
                             self.slope[gain][i] * self.x_values[gain][i] + self.offset[gain][i],
                             "r", label="Fitted line")

        handles, labels = self.remove_legend_dubplicates()
        plt.legend(handles, labels)

        prefix = self.plot_name["combined"].substitute(px=self.pixel,
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

    def generate_colormap_matrix(self):
        """
        generate map for low gain
        """
        gain = 2
        gain_name = "low"

        l = self.fit_cutoff_left
        r = self.fit_cutoff_right
        n_used_fits = self.n_intervals[gain_name] - l - r

        # gain, pixel_v, pixel_u, mem_cell
        n_pixel_v, n_pixel_u, n_mem_cell = self.result["slope"]["mean"].shape[1:]

        for key in self.colormap_matrix:
            # create array
            self.colormap_matrix[key] = np.empty((n_pixel_v, n_pixel_u, n_mem_cell))
            # intiate with nan
            self.colormap_matrix[key][:, :, :] = np.NAN

        for mem_cell in self.mem_cell_list:
            for v in self.pixel_v_list:
                for u in pixel_u_list:
                    #print("pixel [{},{}], mem_cell {}".format(v, u, mem_cell))

                    mean = self.result["slope"]["mean"][gain, v, u, mem_cell]
                    values = self.result["slope"]["individual"][gain_name][v, u, mem_cell][l:-r]
                    #print("slope")
                    #print("mean={}, values={}".format(mean, values))

                    try:
                        if mean != 0:
                            self.colormap_matrix["slope"][v, u, mem_cell] = (
                                np.linalg.norm(values - mean)/(n_used_fits * mean))
                    except:
                        print("pixel {}, mem_cell {}".format(mem_cell, v, u))
                        print("slope")
                        print("mean={}".format(mean))
                        print("values={}".format(values))


                    mean = self.result["slope"]["mean"][gain, v, u, mem_cell]
                    values = self.result["offset"]["individual"][gain_name][v, u, mem_cell][l:-r]

                    try:
                        if mean != 0:
                            self.colormap_matrix["offset"][v, u, mem_cell] = (
                                np.linalg.norm(values - mean)/(n_used_fits * mean))
                    except:
                        print("pixel {}, mem_cell {}".format(v, u, mem_cell))
                        print("slope")
                        print("mean={}".format(mean))
                        print("values={}".format(values))

                    #print("colormap_matrix_slopes={}".format(self.colormap_matrix_slopes[v, u]))

        #print("colormap")
        #print(colormap_matrix_slopes)

    def plot_colormap(self, mem_cell_list=False):
        if not mem_cell_list:
            mem_cell_list = self.mem_cell_list

        for mem_cell in mem_cell_list:
            # fill number to be of 3 digits
            prefix = self.plot_name["colormap"].substitute(mc=str(self.mem_cell).zfill(3))

            for key in self.colormap_matrix:
                fig = plt.figure(figsize=None)
                plt.imshow(self.colormap_matrix[key][..., mem_cell])
                plt.colorbar()

                fig.savefig("{}_{}_{}{}".format(prefix, "low", key, self.plot_ending))
                fig.clf()
                plt.close(fig)


if __name__ == "__main__":

    base_dir = "/gpfs/cfel/fsds/labs/agipd/calibration/processed/"

    asic = 1
    module = "M314"
    temperature = "temperature_m15C"
    current = "itestc150"

    input_file = os.path.join(base_dir, module, temperature, "drscs", current,
                              "{}_drscs_{}_asic{}.h5".format(module, current, str(asic).zfill(2)))
    output_file = os.path.join(base_dir, module, temperature, "drscs", current, "process_result",
                               "{}_drscs_{}_asic{}_processed.h5".format(module, current, str(asic).zfill(2)))
    #plot_dir = os.path.join(base_dir, "M314/temperature_m15C/drscs/plots/itestc150/manu_test")
    plot_dir = os.path.join(base_dir, module, temperature, "drscs", "plots", current, "manu_test")
    plot_prefix = "{}_{}_asic".format(module, current)

    pixel_v_list = np.arange(64)
    pixel_u_list = np.arange(64)
    mem_cell_list = np.arange(1, 2)
    #pixel_v_list = np.arange(1)
    #pixel_u_list = np.arange(1, 2)
    #mem_cell_list = np.arange(1, 2)

    #create_plots can be set to False, "data", "fit", "combined" or "all"
    cal = ProcessDrscs(asic, input_file, output_file, plot_prefix, plot_dir=plot_dir, create_plots=False)

    print("\nRun processing")
    t = time.time()
    cal.run(pixel_v_list, pixel_u_list, mem_cell_list, create_error_plots=False, create_colormaps=False)
    print("Processing took time: {}".format(time.time() - t))
