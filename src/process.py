from __future__ import print_function

from multiprocessing import Pool, TimeoutError
import matplotlib.pyplot as plt
import numpy as np
import h5py
from string import Template
import os
import sys
import time
import traceback


class IntervalError(Exception):
    pass

class FitError(Exception):
    pass

class IntervalSplitError(Exception):
    pass

def check_file_exist(file_name):
    print("save_file = {}".format(file_name))
    if os.path.exists(file_name):
        print("Output file already exists")
        sys.exit(1)
    else:
        print("Output file: ok")

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

# not defined inside the class to let other classes reuse this
def initiate_result(pixel_v_list, pixel_u_list, mem_cell_list, n_gain_stages,
                    n_intervals, n_zero_region_stored, nbins):

    result = {
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
        "residuals": {
            "mean": None,
            "individual": {
                "high" : None,
                "medium": None,
                "low": None
            }
        },
        "intervals": {
            "found_zero_regions": None,
            "used_zero_regions": None,
            "gain_stages": None,
            "subintervals": {
                "high" : None,
                "medium": None,
                "low": None
            }
        },
        "medians": {
            "high" : None,
            "medium": None,
            "low": None
        },
        "thresholds": None,
        "error_code": None,
        "warning_code": None,
        "collection": {
            "nbins": None,
            "thold_for_zero": None,
            "scaling_point": None,
            "scaling_factor": None,
            "n_gain_stages": None,
            "fit_cutoff_left": None,
            "fit_cutoff_right": None,
            "spread": None,
            "used_nbins": None
        }
    }

    # +1 because counting starts with zero
    shape_tmp = (pixel_v_list.max() + 1,
                 pixel_u_list.max() + 1,
                 mem_cell_list.max() + 1)

    shape = (n_gain_stages, ) + shape_tmp
    print("result shape: {}".format(shape))

    # thresholds between to distinguish between the gain stages
    threshold_shape = (n_gain_stages - 1,) + shape_tmp
    print("threshold shape: {}".format(threshold_shape))

    # initiate fit results
    for key in ["slope", "offset", "residuals"]:
        result[key]["mean"] = np.zeros(shape, np.float32)
        result[key]["individual"] = {
            "high": np.zeros(shape_tmp + (n_intervals["high"],)),
            "medium": np.zeros(shape_tmp + (n_intervals["medium"],)),
            "low": np.zeros(shape_tmp + (n_intervals["low"],)),
        }
    result["medians"] = np.zeros(shape, np.float32)
    # initiated with -1 to distinguish between code failures and catched errors
    result["error_code"] = -1 * np.ones(shape_tmp, np.int16)
    result["warning_code"] = np.zeros(shape_tmp, np.int16)
    result["collection"]["spread"] = np.zeros(shape_tmp)
    result["collection"]["used_nbins"] = nbins * np.ones(shape_tmp)

    # intiate intervals
    # interval consists of start and end point -> x2
    result["intervals"]["gain_stages"] = init_with_nan(shape + (2,))
    result["intervals"]["used_zero_regions"] = init_with_nan(shape_tmp + (2,2))
    result["intervals"]["found_zero_regions"] = (
        init_with_nan(shape_tmp + (n_zero_region_stored, 2)))
    for key in ["high", "medium", "low"]:
        result["intervals"]["subintervals"][key] = (
            init_with_nan(shape_tmp + (n_intervals[key], 2)))

    # initiate thresholds
    result["thresholds"] = np.zeros(threshold_shape, np.float32)

    return result

def init_with_nan(shape):
    # create array
    obj = np.empty(shape)
    # intiate with nan
    obj[...] = np.NAN

    return obj

# gives back to N biggest entries of a list
# source: https://stackoverflow.com/questions/12787650/finding-the-index-of-n-biggest-elements-in-python-array-list-efficiently
def biggest_entries(list_to_check, N):
    return np.argsort(list_to_check)[::-1][:N]

class ProcessDrscs():
    def __init__(self, asic, input_fname, output_fname=False,
                 plot_prefix=None, plot_dir=None, create_plots=False):

        self.input_fname = input_fname
        if output_fname:
            self.output_fname = output_fname
        else:
            self.output_fname = None

        self.digital_path = "/entry/instrument/detector/data_digital"
        self.analog_path = "/entry/instrument/detector/data"

        self.create_plot_type = create_plots
        self.plot_dir = plot_dir

        self.digital = None
        self.analog = None

        self.percent = 10
        self.nbins = 30

        self.hist = None
        self.bins = None

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

        # temporarily results which can not be stored in the result directly
        self.res = {
            "found_zero_regions": None,
            "thresholds": None
        }

        ###
        # error_codes:
        # -1   Not handled (something went wrong)
        # 1    Unknown error
        # 2    Zero region: too few intervals
        # 3    Zero region: too many intervals
        #
        # warning_codes:
        # 1    Use lesser nbins
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

            title_prefix = plot_prefix.rsplit("/", 1)[1]
            self.plot_title = {
                "origin_data": Template(title_prefix + "_${px}_${mc} data"),
                "fit": Template(title_prefix + "_${px}_${mc} fit ${g}"),
                "combined": Template(title_prefix + "_${px}_${mc} combined"),
                "colormap": Template(title_prefix + "_${mc} colormap"),
            }
            self.plot_ending = ".png"
        else:
            self.plot_name = None
            self.plot_ending = None

        if self.output_fname is not None:
            check_file_exist(self.output_fname)

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
            create_error_plots=False, create_colormaps=False):

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
        self.result = initiate_result(self.pixel_v_list, self.pixel_u_list,
                                      self.mem_cell_list, self.n_gain_stages,
                                      self.n_intervals,
                                      self.n_zero_region_stored,
                                      self.nbins)

        self.result["collection"]["nbins"] = self.nbins
        self.result["collection"]["thold_for_zero"] = self.thold_for_zero
        self.result["collection"]["scaling_point"] = self.scaling_point
        self.result["collection"]["scaling_factor"] = self.scaling_factor
        self.result["collection"]["n_gain_stages"] = self.n_gain_stages
        self.result["collection"]["fit_cutoff_left"] = self.fit_cutoff_left
        self.result["collection"]["fit_cutoff_right"] = self.fit_cutoff_right
        #spread and used_nbin is intiated in initiate_result

        for pixel_v in self.pixel_v_list:
            for pixel_u in self.pixel_u_list:
                #t = time.time()
                for mem_cell in self.mem_cell_list:
                    try:
                        self.current_idx = (pixel_v, pixel_u, mem_cell)
                        self.gain_idx = {
                            "high": (0, ) + self.current_idx,
                            "medium": (1, ) + self.current_idx,
                            "low": (2, ) + self.current_idx
                        }

                        self.process_data_point(self.current_idx)

                        # store current_idx dependent results in the final matrices
                        try:
                            self.result["intervals"]["found_zero_regions"][self.current_idx] = (
                                self.res["found_zero_regions"])
                        except ValueError:
                            # it is not fixed how many found_zero_regions are found
                            # but the size of np.arrays has to be fixed
                            l = len(self.res["found_zero_regions"])
                            f_zero_regions = (
                                self.result["intervals"]["found_zero_regions"][self.current_idx])

                            # fill up with found ones, rest of the array stays NAN
                            if l < self.n_zero_region_stored:
                                f_zero_regions[:l] = self.res["found_zero_regions"]
                            # more regions found than array can hold, only store the first part
                            else:
                                f_zero_regions = self.res["found_zero_regions"][:n_zero_region_stored]

                        for gain in self.gain_idx:
                            l = self.fit_cutoff_left
                            r = self.fit_cutoff_right
                            if len(self.result["slope"]["individual"][gain][self.current_idx]) >= l + r + 1:
                                for t in ["slope", "offset", "residuals"]:
                                    self.result[t]["mean"][self.gain_idx[gain]] = (
                                        np.mean(self.result[t]["individual"][gain][self.current_idx][l:-r]))
                            else:
                                for t in ["slope", "offset", "residuals"]:
                                    self.result[t]["mean"][self.gain_idx[gain]] = (
                                        np.mean(self.result[t]["individual"][gain][self.current_idx]))

                        # store the thresholds
                        for i in np.arange(len(self.res["thresholds"])):
                            self.result["thresholds"][(i, ) + self.current_idx] = self.res["thresholds"][i]

                        if self.result["error_code"][self.current_idx] < 0:
                            self.result["error_code"][self.current_idx] = 0

                    except KeyboardInterrupt:
                        sys.exit(1)
                    except Exception as e:
                        if type(e) == IntervalError:
                            #print("IntervalError")
                            pass
                        elif type(e) == IntervalSplitError:
                            #print("IntervalSplitError")
                            pass
                        elif type(e) == IntervalSplitError:
                            #print("IntervalSplitError")
                            pass
                        else:
                            print("Failed to run for pixel [{}, {}] and mem_cell {}"
                                  .format(self.current_idx[0], self.current_idx[1], self.current_idx[2]))
                            print(traceback.format_exc())

                            if self.result["error_code"][self.current_idx] <= 0:
                                self.result["error_code"][self.current_idx] = 1

                            print("hist={}".format(self.hist))
                            print("bins={}".format(self.bins))
                            print("used_zero_regions",
                                  self.result["intervals"]["used_zero_regions"][self.current_idx])
                            print("found_zero_regions", self.res["found_zero_regions"])
                            print("thesholds={}".format(self.res["thresholds"]))

                        if create_error_plots:
                            try:
                                self.generate_data_plot()
                            except:
                                print("Failed to generate plot")
                                raise

                        #raise
                #print("Process pixel {} took time: {}".format([pixel_v, pixel_u], time.time() - t))

        if create_colormaps:
            if type(create_colormaps) == list and "matrix" in create_colormaps:
                self.generate_colormap_matrix()
            else:
                self.generate_colormap_matrix()
                self.plot_colormap()

        if self.output_fname is not None:
            self.write_data()

    def process_data_point(self, current_idx):

        data = self.digital[self.current_idx[0], self.current_idx[1], self.current_idx[2], :]

        spread = int(np.linalg.norm(np.max(data) - np.min(data)))

        if 0 < spread and spread < self.nbins:
            nbins = spread
            self.result["warning_code"][self.current_idx] = 1
            print("[{}, {}], {}: Spread to lower than number of bins. Adjusting (new nbins={}, spread={})"
                  .format(self.current_idx[0], self.current_idx[1], self.current_idx[2], nbins, spread))
        else:
            nbins = self.nbins

        self.result["collection"]["spread"][self.current_idx] = spread
        self.result["collection"]["used_nbins"][self.current_idx] = nbins

        self.hist, self.bins = np.histogram(data, bins=nbins)
        #print("hist={}".format(self.hist))
        #print("bins={}".format(self.bins))

        self.calc_thresholds()

        (self.result["intervals"]["gain_stages"][self.gain_idx["high"]],
         self.result["intervals"]["subintervals"]["high"][self.current_idx])= (
            self.fit_gain("high",
                          0,
                          self.res["thresholds"][0],
                          cut_off_ends=False))

        (self.result["intervals"]["gain_stages"][self.gain_idx["medium"]],
         self.result["intervals"]["subintervals"]["medium"][self.current_idx]) = (
            self.fit_gain("medium",
                          self.res["thresholds"][0],
                          self.res["thresholds"][1],
                          cut_off_ends=False))

        (self.result["intervals"]["gain_stages"][self.gain_idx["low"]],
         self.result["intervals"]["subintervals"]["low"][self.current_idx]) = (
            self.fit_gain("low",
                          self.res["thresholds"][1],
                          None,
                          cut_off_ends=False))

        self.calc_gain_median()

        if self.create_plot_type:
            print("\nGenerate plots")
            t = time.time()

            if type(self.create_plot_type) == list:
                print("plot type", self.create_plot_type)

                if "all" in self.create_plot_type:
                    self.generate_all_plots()
                else:
                    if "data" in self.create_plot_type:
                        print("data plot")
                        self.generate_data_plot()

                    if "fit" in self.create_plot_type:
                        self.generate_fit_plot("high")
                        self.generate_fit_plot("medium")
                        self.generate_fit_plot("low")

                    if "combined" in self.create_plot_type:
                        print("combined plot")
                        self.generate_combined_plot()
            else:
                self.generate_all_plots()

            print("took time: {}".format(time.time() - t))

    def calc_thresholds(self):
        self.res["found_zero_regions"] = (
            contiguous_regions(self.hist < self.thold_for_zero))
        #print("found_zero_regions raw: {}".format(self.res["found_zero_regions"]))

        # remove the first interval found if it starts with the first bin
        if (self.res["found_zero_regions"].size != 0
                and self.res["found_zero_regions"][0][0] == 0):
            self.res["found_zero_regions"] = self.res["found_zero_regions"][1:]

        # remove the last interval found if it ends with the last bin
        if (self.res["found_zero_regions"].size != 0
                and self.res["found_zero_regions"][-1][1] == self.nbins):
            self.res["found_zero_regions"] = self.res["found_zero_regions"][:-1]

        #print("found_zero_regions ends removed: {}".format(self.res["found_zero_regions"]))

        if len(self.res["found_zero_regions"]) < 2:
            #print("thold_for_zero={}".format(self.thold_for_zero))
            #print("intervals={}".format(self.res["found_zero_regions"]))
            self.result["error_code"][self.current_idx] = 2
            raise IntervalError("Too few intervals")
        if len(self.res["found_zero_regions"]) > 2:
            #print("thold_for_zero={}".format(self.thold_for_zero))
            #print("intervals={}".format(self.res["found_zero_regions"]))
            self.result["error_code"][self.current_idx] = 3
            raise IntervalError("Too many intervals")

        #print("found_zero_regions", self.res["found_zero_regions"])
        used_zero_regions = self.res["found_zero_regions"]
        self.result["intervals"]["used_zero_regions"][self.current_idx] = used_zero_regions


        mean_zero_region = np.mean(used_zero_regions, axis=1).astype(int)
        #print("mean_zero_region={}".format(mean_zero_region))

        self.res["thresholds"] = self.bins[mean_zero_region]
        #print("thesholds={}".format(self.res["thresholds"]))

    def rechoosing_fit_intervals(self):
        #if multiple intervals were found, choose the biggest ones
        interval_length = [region[1] - region[0]
                           for region in self.res["found_zero_regions"]]

        return biggest_entries(interval_length, 2)

    def determine_fit_interval(self, threshold_l, threshold_u):
        data_d = self.digital[self.current_idx[0], self.current_idx[1],
                              self.current_idx[2], :]

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

        return contiguous_regions(condition)

    def split_interval(self, interval, nsplits):
        """
        split given interval into equaly sized sub-intervals
        """
        #print("nsplits", nsplits)
        #print("interval={}".format(interval))

        step = int((interval[-1] - interval[0]) / nsplits)
        #print("step", step)

        if step == 0:
            print("nsplits", nsplits)
            print("interval={}".format(interval))
            raise IntervalSplitError("Interval has not enough points to split")

        splitted_intervals = [
            [interval[0] + i * step, interval[0] + (i + 1) * step]
                for i in np.arange(nsplits)]

        # due to rounding or int convertion the last calculated interval ends
        # before the end of the given interval
        splitted_intervals[-1][1] = interval[1]

        #print("splitted_intervals={}".format(splitted_intervals))

        return splitted_intervals

    def calc_gain_median(self):
        for gain in ["high", "medium", "low"]:
            idx = (self.current_idx[0],
                   self.current_idx[1],
                   self.current_idx[2],
                   slice(self.result["intervals"]["gain_stages"][self.gain_idx[gain]][0],
                         self.result["intervals"]["gain_stages"][self.gain_idx[gain]][1]))

            self.result["medians"][self.gain_idx[gain]] = np.median(self.digital[idx])

    def fit_gain(self, gain, threshold_l, threshold_u, cut_off_ends):
        #print("\nfitting {} gain with threshholds ({}, {})".format(gain, threshold_l, threshold_u))
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
        self.data_to_fit[gain][interval_idx] = self.analog[self.current_idx[0], self.current_idx[1],
                                                           self.current_idx[2],
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
        array_idx = self.current_idx + (interval_idx,)
        res = None
        try:
            res = np.linalg.lstsq(self.coefficient_matrix, self.data_to_fit[gain][interval_idx])
            self.result["slope"]["individual"][gain][array_idx] = res[0][0]
            self.result["offset"]["individual"][gain][array_idx] = res[0][1]
            self.result["residuals"]["individual"][gain][array_idx] = res[1]
        except:
            if res is None:
                print("interval\n{}".format(interval))
                print("self.coefficient_matrix\n{}".format(self.coefficient_matrix))
                print("self.data_to_fit[{}][{}]\n{}"
                      .format(gain, interval_idx, self.data_to_fit[gain][interval_idx]))

                raise

            if res[0].size != 2:
                raise FitError("Failed to calculate slope and offset")
            elif res[1].size != 1:
                raise FitError("Failed to calculate residual")
            else:
                print("interval\n{}".format(interval))
                print("self.coefficient_matrix\n{}".format(self.coefficient_matrix))
                print("self.data_to_fit[{}][{}]\n{}"
                      .format(gain, interval_idx, self.data_to_fit[gain][interval_idx]))
                print("res", res)

                raise

        #print("found slope: {}".format(self.result["slope"]["individual"][gain]))
        #print("found offset: {}".format(self.result["offset"]["individual"][gain]))

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
                self.analog[self.current_idx[0], self.current_idx[1],
                            self.current_idx[2], :],
                ".", markersize=0.5, label="analog")
        ax.plot(self.scaled_x_values,
                self.digital[self.current_idx[0], self.current_idx[1], self.current_idx[2], :],
                ".", markersize=0.5, label="digital")

        ax.legend()
        prefix = self.plot_name["origin_data"].substitute(px=[self.current_idx[0], self.current_idx[1]],
                                                          # fill number to be of 3 digits
                                                          mc=str(self.current_idx[2]).zfill(3))
        title = self.plot_title["origin_data"].substitute(px=[self.current_idx[0], self.current_idx[1]],
                                                          # fill number to be of 3 digits
                                                          mc=str(self.current_idx[2]).zfill(3))
        fig.suptitle(title)
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

        fig = plt.figure(figsize=None)
        for i in np.arange(len(self.x_values[gain])):
            array_idx = self.current_idx + (i,)

            slope = self.result["slope"]["individual"][gain][array_idx]
            x_values = self.x_values[gain][i]
            offset = self.result["offset"]["individual"][gain][array_idx]

            plt.plot(x_values, self.data_to_fit[gain][i],
                     'o', color="C0", label="Original data", markersize=1)

            plt.plot(x_values, slope * x_values + offset,
                     "r", label="Fitted line ({} intervals)".format(self.n_intervals[gain]))

        handles, labels = self.remove_legend_dubplicates()
        plt.legend(handles, labels)

        prefix = self.plot_name["fit"].substitute(px=[self.current_idx[0], self.current_idx[1]],
                                                  # fill number to be of 3 digits
                                                  mc=str(self.current_idx[2]).zfill(3))
        title = self.plot_title["fit"].substitute(px=[self.current_idx[0], self.current_idx[1]],
                                                  # fill number to be of 3 digits
                                                  mc=str(self.current_idx[2]).zfill(3),
                                                  g=gain)
        fig.suptitle(title)
        fig.savefig("{}_{}{}".format(prefix, gain, self.plot_ending))
        fig.clf()
        plt.close(fig)

    def generate_combined_plot(self):
        idx = (self.current_idx[0],
               self.current_idx[1],
               self.current_idx[2],
               slice(None))

        fig = plt.figure(figsize=None)
        plt.plot(self.scaled_x_values,
                 self.analog[idx],
                 ".", markersize=0.5, label="analog")

        plt.plot(self.scaled_x_values,
                 self.digital[idx],
                 ".", markersize=0.5, label="digital")

        for gain in ["high", "medium", "low"]:
            # if there are multiple sub-fits some are cut off
            if len(self.x_values[gain]) >= self.fit_cutoff_left + self.fit_cutoff_right + 1:
                # display the cut off fits on the left side
                for i in np.arange(self.fit_cutoff_left):
                    array_idx = self.current_idx + (i,)

                    slope = self.result["slope"]["individual"][gain][array_idx]
                    x_values = self.x_values[gain][i]
                    offset = self.result["offset"]["individual"][gain][array_idx]

                    plt.plot(x_values, slope * x_values + offset,
                             "r", alpha=0.3, label="Fitted line (unused)")
                # display the used fits
                for i in np.arange(self.fit_cutoff_left, len(self.x_values[gain]) - self.fit_cutoff_right):
                    array_idx = self.current_idx + (i,)

                    slope = self.result["slope"]["individual"][gain][array_idx]
                    x_values = self.x_values[gain][i]
                    offset = self.result["offset"]["individual"][gain][array_idx]

                    plt.plot(x_values, slope * x_values + offset,
                             "r", label="Fitted line")
                # display the cut off fits on the right side
                for i in np.arange(len(self.x_values[gain]) - self.fit_cutoff_right,
                                   len(self.x_values[gain])):
                    array_idx = self.current_idx + (i,)

                    slope = self.result["slope"]["individual"][gain][array_idx]
                    x_values = self.x_values[gain][i]
                    offset = self.result["offset"]["individual"][gain][array_idx]

                    plt.plot(x_values, slope * x_values + offset,
                             "r", alpha=0.3, label="Fitted line (unused)")
            else:
                # display all fits
                for i in np.arange(len(self.x_values[gain])):
                    array_idx = self.current_idx + (i,)

                    slope = self.result["slope"]["individual"][gain][array_idx]
                    x_values = self.x_values[gain][i]
                    offset = self.result["offset"]["individual"][gain][array_idx]

                    plt.plot(x_values, slope * x_values + offset,
                             "r", label="Fitted line")

        handles, labels = self.remove_legend_dubplicates()
        plt.legend(handles, labels)

        prefix = self.plot_name["combined"].substitute(px=[self.current_idx[0], self.current_idx[1]],
                                                       # fill number to be of 3 digits
                                                       mc=str(self.current_idx[2]).zfill(3))
        title = self.plot_title["combined"].substitute(px=[self.current_idx[0], self.current_idx[1]],
                                                       # fill number to be of 3 digits
                                                       mc=str(self.current_idx[2]).zfill(3))
        fig.suptitle(title)
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

        for v in self.pixel_v_list:
            for u in pixel_u_list:
                for mem_cell in self.mem_cell_list:
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
            prefix = self.plot_name["colormap"].substitute(mc=str(mem_cell).zfill(3))

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
    #asic = 11
    module = "M314"
    temperature = "temperature_m15C"
    current = "itestc150"

    input_fname = os.path.join(base_dir, module, temperature, "drscs", current, "gather",
                              "{}_drscs_{}_asic{}.h5".format(module, current, str(asic).zfill(2)))
    output_fname = os.path.join(base_dir, module, temperature, "drscs", current, "process",
                               "{}_drscs_{}_asic{}_processed.h5".format(module, current, str(asic).zfill(2)))
    #plot_dir = os.path.join(base_dir, "M314/temperature_m15C/drscs/plots/itestc150/manu_test")
    plot_dir = os.path.join(base_dir, module, temperature, "drscs", "plots", current, "manu_test")
    plot_prefix = "{}_{}_asic{}".format(module, current, asic)

    #pixel_v_list = np.arange(0, 64)
    #pixel_u_list = np.arange(0, 64)
    #mem_cell_list = np.arange(0, 352)
    #pixel_v_list = np.arange(29, 30)
    #pixel_u_list = np.arange(1, 2)
    #mem_cell_list = np.arange(1, 2)
    pixel_v_list = np.arange(10, 11)
    pixel_u_list = np.arange(11, 12)
    mem_cell_list = np.arange(30, 31)

    output_fname = False
    #create_plots can be set to False, "data", "fit", "combined" or "all"
    create_plots = False
    create_error_plots=True
    #create_plots=["data", "combined"]

    cal = ProcessDrscs(asic, input_fname, output_fname=output_fname,
                       plot_prefix=plot_prefix, plot_dir=plot_dir,
                       create_plots=create_plots)

    print("\nRun processing")
    t = time.time()
    cal.run(pixel_v_list, pixel_u_list, mem_cell_list,
            create_error_plots=create_error_plots,
            create_colormaps=False)
    print("Processing took time: {}".format(time.time() - t))
