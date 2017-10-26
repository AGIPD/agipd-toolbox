import builtins
import os
import time

import h5py
import numpy as np
import traceback

from .util import check_file_exists

debug_mode = False


def print(*args):
    if debug_mode:
        builtins.print(args)


class FitError(Exception):
    pass


class IntervalSplitError(Exception):
    pass


class DataQualityError(Exception):
    pass


class GainStageNumberError(Exception):
    pass


class ThresholdNumberError(Exception):
    pass


class InfiniteLoopNumberError(Exception):
    pass


# not defined inside the class to let other classes reuse this
def initiate_result(dim_v, dim_u, dim_mem_cell, n_gain_stages,
                    n_intervals, n_diff_changes_stored):

    result = {
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
        "residual": {
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
        "intervals": {
            "gain_stages": None,
            "subintervals": {
                "high": None,
                "medium": None,
                "low": None
            }
        },
        "medians": {
            "high": None,
            "medium": None,
            "low": None
        },
        "thresholds": None,
        "error_code": None,
        "warning_code": None,
        "collection": {
            "diff_threshold": None,
            "region_range_in_percent": None,
            "region_range": None,
            "safety_factor": None,
            "diff_changes_idx": None,
            "len_diff_changes_idx": None,
            "saturation_threshold": None,
            "saturation_threshold_diff": None,
            "scaling_point": None,
            "scaling_factor": None,
            "n_gain_stages": None,
            "fit_cutoff_left": None,
            "fit_cutoff_right": None,
        }
    }

    shape_tmp = (dim_v, dim_u, dim_mem_cell)

    shape = (n_gain_stages, ) + shape_tmp
    print("result shape: {}".format(shape))

    # thresholds between to distinguish between the gain stages
    threshold_shape = (n_gain_stages - 1,) + shape_tmp
    print("threshold shape: {}".format(threshold_shape))

    # initiate fit results
    for key in ["slope", "offset", "residual", "average_residual"]:
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

    # intiate intervals
    # interval consists of start and end point -> x2
    result["intervals"]["gain_stages"] = init_with_nan(shape + (2,), np.int)
    for key in ["high", "medium", "low"]:
        result["intervals"]["subintervals"][key] = (
            init_with_nan(shape_tmp + (n_intervals[key], 2), np.int))
    result["intervals"]["saturation"] = init_with_nan(shape_tmp + (2,), np.int)

    # initiate thresholds
    result["thresholds"] = np.zeros(threshold_shape, np.float32)

    result["collection"]["diff_changes_idx"] = init_with_nan(
        shape_tmp + (n_diff_changes_stored,), np.int)
    result["collection"]["len_diff_changes_idx"] = np.zeros(shape_tmp, np.int)

    return result


def init_with_nan(shape, dtype):
    # create array
    obj = np.empty(shape, dtype=dtype)
    # intiate with nan
    obj[...] = np.NAN

    return obj


# gives back to N biggest entries of a list
# modification of: https://stackoverflow.com/questions/12787650/finding-the-index-of-n-biggest-elements-in-python-array-list-efficiently  # noqa E501
def biggest_entries(list_to_check, N):
    return np.argsort(list_to_check)[-N:]


class ProcessDrscs():
    def __init__(self, asic, input_fname=None, output_fname=None,
                 safety_factor=1000, plot_prefix=None, plot_dir=None,
                 create_plots=False, log_level="info", input_handle=None):

        self.use_debug = False

        self.input_fname = input_fname
        self.input_handle = input_handle
        self.output_fname = output_fname

        self.analog = None
        self.digital = None

        self.digital_path = "/entry/instrument/detector/data_digital"
        self.analog_path = "/entry/instrument/detector/data"

        self.create_plot_type = create_plots
        self.plot_dir = plot_dir

        self.percent = 10

        self.safety_factor = safety_factor
        self.n_diff_changes_stored = 10
        self.saturation_threshold = 14000
        self.saturation_threshold_diff = 30

        self.diff_threshold = -100
        self.region_range_in_percent = 2
        self.region_range = 10
        self.max_diff_changes = 150

        self.scaling_point = 200
        self.scaling_factor = 10

        self.n_gain_stages = 3
        # in how many the base interval should be split
        self.n_intervals = {
            "high": 1,
            "medium": 1,
            "low": 5
        }
        self.fit_cutoff_left = 1
        self.fit_cutoff_right = 1

        self.use_digital = False

        self.current_idx = None
        self.out_idx = None

        self.gain_idx = None
        self.scaled_x_values = None
        self.coefficient_matrix = None
        self.x_values = {
            "high": [[] for _ in np.arange(self.n_intervals["high"])],
            "medium": [[] for _ in np.arange(self.n_intervals["medium"])],
            "low": [[] for _ in np.arange(self.n_intervals["low"])]
        }
        self.data_to_fit = {
            "high": [[] for _ in np.arange(self.n_intervals["high"])],
            "medium": [[] for _ in np.arange(self.n_intervals["medium"])],
            "low": [[] for _ in np.arange(self.n_intervals["low"])]
        }

        ###
        # error_code:
        # -1   Not handled (something went wrong)
        # 1    Unknown error
        # 2    Too many gain stages found
        # 3    Not enough gain stages found
        # 4    Breaking infinite loop
        # 5    Failed to calculate slope and offset
        # 6    Not enough point to fit {} gain
        # 7    Interval has not enough points to split
        # 8    Too many fluctuations in the data found
        #
        # warning_code:
        # 1    Failed to calculate residual
        ###

        if plot_prefix:
            print("plot_prefix {}".format(plot_prefix))
            self.plot_file_prefix = "{}_asic{}".format(plot_prefix,
                                                       str(asic).zfill(2))
            if plot_dir is not None:
                self.plot_file_prefix = os.path.join(self.plot_dir,
                                                     self.plot_file_prefix)

            self.plot_title_prefix = self.plot_file_prefix.rsplit("/", 1)[1]
            print("plot_file_prefix {}".format(self.plot_file_prefix))
            print("plot_title_prefix {}".format(self.plot_title_prefix))

            self.plot_ending = ".png"
        else:
            self.plot_file_prefix = None
            self.plot_title_prefix = None
            self.plot_ending = None

        if self.output_fname is not None:
            check_file_exists(self.output_fname)

    def run(self, pixel_v_list, pixel_u_list, mem_cell_list,
            create_error_plots=False):

        if len(pixel_v_list) == 0 or len(pixel_u_list) == 0:
            print("pixel_v_list: {}".format(pixel_v_list))
            print("pixel_u_list: {}".format(pixel_u_list))
            raise Exception("No proper pixel specified")
        if len(mem_cell_list) == 0:
            print("mem_cell_list: {}".format(mem_cell_list))
            raise Exception("No proper memory cell specified")

        if create_error_plots and self.plot_title_prefix is None:
            raise Exception("Plotting was not defined on initiation. "
                            "Quitting.")

        self.pixel_v_list = pixel_v_list
        self.pixel_u_list = pixel_u_list
        self.mem_cell_list = mem_cell_list

        self.check_lists_continous()

        if self.input_fname is not None:
            print("Load data")
            t = time.time()
            self.load_data()
            print("took time: {}".format(time.time() - t))

        self.scale_full_x_axis()

        # +1 because counting starts with zero
        self.dim_v = self.pixel_v_list.max() - self.pixel_v_list[0] + 1
        self.dim_u = self.pixel_u_list.max() - self.pixel_u_list[0] + 1
        self.dim_mem_cell = (self.mem_cell_list.max() -
                             self.mem_cell_list[0] + 1)

        # initiate
        self.result = initiate_result(self.dim_v,
                                      self.dim_u,
                                      self.dim_mem_cell,
                                      self.n_gain_stages,
                                      self.n_intervals,
                                      self.n_diff_changes_stored)

        c = self.result["collection"]
        c["diff_threshold"] = self.diff_threshold
        c["region_range_in_percent"] = self.region_range_in_percent
        c["region_range"] = self.region_range
        c["safety_factor"] = self.safety_factor
        c["n_diff_changes_stored"] = self.n_diff_changes_stored
        c["scaling_point"] = self.scaling_point
        c["scaling_factor"] = self.scaling_factor
        c["n_gain_stages"] = self.n_gain_stages
        c["fit_cutoff_left"] = self.fit_cutoff_left
        c["fit_cutoff_right"] = self.fit_cutoff_right
        c["saturation_threshold"] = self.saturation_threshold
        c["saturation_threshold_diff"] = self.saturation_threshold_diff

        for pixel_v in self.pixel_v_list:

            for pixel_u in self.pixel_u_list:
                for mem_cell in self.mem_cell_list:
                    try:
                        # location in the input data
                        self.current_idx = (pixel_v - self.pixel_v_list[0],
                                            pixel_u - self.pixel_u_list[0],
                                            mem_cell - self.mem_cell_list[0])
                        # location where to write the data to
                        self.out_idx = (pixel_v, pixel_u, mem_cell)

                        self.gain_idx = {
                            "high": (0, ) + self.current_idx,
                            "medium": (1, ) + self.current_idx,
                            "low": (2, ) + self.current_idx
                        }

                        self.process_data_point()

                        for gain in self.gain_idx:
                            l = self.fit_cutoff_left
                            r = self.fit_cutoff_right

                            intvs = self.result["slope"]["individual"][gain]
                            n_intvs = len(intvs[self.current_idx])
                            if n_intvs >= l + r + 1:
                                for t in ["slope", "offset", "residual",
                                          "average_residual"]:
                                    mean = self.result[t]["mean"]
                                    indv = self.result[t]["individual"][gain]
                                    mean[self.gain_idx[gain]] = np.mean(
                                        indv[self.current_idx][l:-r])
                            else:
                                for t in ["slope", "offset", "residual",
                                          "average_residual"]:
                                    mean = self.result[t]["mean"]
                                    indv = self.result[t]["individual"][gain]

                                    mean[self.gain_idx[gain]] = np.mean(
                                        indv[self.current_idx])

                        if self.result["error_code"][self.current_idx] < 0:
                            self.result["error_code"][self.current_idx] = 0

                    except Exception as e:
                        if type(e) == IntervalSplitError:
                            if self.use_debug:
                                self.log.debug("IntervalSplitError")
                        elif type(e) == DataQualityError:
                            if self.use_debug:
                                self.log.debug("DataQualityError")
                        elif type(e) == GainStageNumberError:
                            if self.use_debug:
                                self.log.debug("{}: GainStageNumberError"
                                               .format(self.out_idx))
                                self.log.debug(
                                    "found number of diff_changes_idx={}"
                                    .format(len(self.diff_changes_idx)))
                        elif type(e) == FitError:
                            if self.use_debug:
                                self.log.debug("FitError")
                        else:
                            self.log.error("Failed to run for pixel [{}, {}] "
                                           "and mem_cell {}"
                                           .format(self.out_idx[0],
                                                   self.out_idx[1],
                                                   self.out_idx[2]))
                            self.log.error(traceback.format_exc())

                            error_code = self.result["error_code"]

                            if error_code[self.current_idx] <= 0:
                                error_code[self.current_idx] = 1

    def check_lists_continous(self):

        v_list = np.arange(self.pixel_v_list[0], self.pixel_v_list[-1] + 1)
        u_list = np.arange(self.pixel_u_list[0], self.pixel_u_list[-1] + 1)
        start = self.mem_cell_list[0]
        stop = self.mem_cell_list[-1] + 1
        mem_cell_list = np.arange(start, stop)

        if not (np.array_equal(self.pixel_v_list, v_list) or
                not np.array_equal(self.pixel_u_list, u_list) or
                not np.array_equal(self.mem_cell_list, mem_cell_list)):
            print("Input is not a continuous list")
            print("pixel_v_list {}".format(self.pixel_v_list))
            print("pixel_u_list {}".format(self.pixel_u_list))
            print("mem_cell_list {}".format(self.mem_cell_list))

    def load_data(self):
        source_file = None
        if self.input_handle is None:
            source_file = h5py.File(self.input_fname, "r")
        else:
            source_file = self.input_handle

        idx = (slice(self.pixel_v_list[0], self.pixel_v_list[-1] + 1),
               slice(self.pixel_u_list[0], self.pixel_u_list[-1] + 1),
               slice(self.mem_cell_list[0], self.mem_cell_list[-1] + 1),
               slice(None))

        try:
            # origin data is written as int16 which results in a integer
            # overflow when handling the scaling
            self.analog = source_file[self.analog_path][idx].astype(np.int32)
            self.digital = source_file[self.digital_path][idx].astype(np.int32)
        finally:
            if self.input_handle is None:
                source_file.close()

    def process_data_point(self):

        self.calc_gain_regions()

        gain_stages = self.result["intervals"]["gain_stages"]
        subintervals = self.result["intervals"]["subintervals"]

        subintervals["high"][self.current_idx] = (
            self.fit_gain("high", gain_stages[self.gain_idx["high"]],
                          cut_off_ends=False))

        subintervals["medium"][self.current_idx] = (
            self.fit_gain("medium", gain_stages[self.gain_idx["medium"]],
                          cut_off_ends=False))

        subintervals["low"][self.current_idx] = (
            self.fit_gain("low", gain_stages[self.gain_idx["low"]],
                          cut_off_ends=False))

        self.calc_gain_median()
        self.calc_thresholds()

        if self.create_plot_type:
            self.log.info("\nGenerate plots")
            t = time.time()

            self.create_plots()

            self.log.info("took time: {}".format(time.time() - t))

    def calc_gain_regions(self):
        data_a = self.analog[self.current_idx[0],
                             self.current_idx[1],
                             self.current_idx[2], :].astype(np.float)

        # find out if the col was effected by frame loss
        lost_frames = np.where(data_a == 0)
        data_a[lost_frames] = np.nan

        missing = np.isnan(data_a)
        # remember the original indices
        origin_idx = np.arange(data_a.size)[~missing]
        data_a = data_a[~missing]

        # calculates the difference between neighboring elements
        self.diff = np.diff(data_a)

        # an additional threshold is needed to catch cases like this:
        # safety factor : 450
        # 257 : [9600 9239]
        # 258 : [9239 8873]
        # 259 : [8873 8436]
        # roi_before (prev_stop: 257):  [9496 9493 9542 9558 9576]
        # roi_after (pot_start: 258): [9239 8873 8436 8390 8395 8389 8388
        #                              8413 8425 8417]
        # roi_before (prev_stop: 257): [9496 9493 9542 9558 9576]
        # roi_after (pot_start: 259): [8436 8390 8395 8389 8388 8413 8425
        #                              8417 8416 8435]
        # here the safety factor would need needed to be decreased too far

        # other reason: all of the pixels with shadow lines would be passing
        # the tests

        self.diff_changes_idx = np.where((self.diff < self.diff_threshold) |
                                         (self.diff > self.safety_factor))[0]

        error_code = self.result["error_code"]
        if self.diff_changes_idx.size >= self.max_diff_changes:
            msg = "Too many fluctuations in the data found"

            if self.use_debug:
                print(msg)
                print(self.diff_changes_idx)

            error_code[self.current_idx] = 8
            raise DataQualityError(msg)

        if self.use_debug:
            print("diff_changes_idx")
            for i in self.diff_changes_idx:
                print("{} : {}".format(i, data_a[i:i + 2]))

        gain_intervals = [[0, 0]]

        i = 0
        prev_stop = 0
        prev_stop_idx = 0
        pot_start = 0
        set_stop_flag = True
        iteration_check = 0
        last_iteration_borders = []
        region_range_before = 0
        while i < self.diff_changes_idx.size:

            if set_stop_flag:
                prev_stop = self.diff_changes_idx[i]
                prev_stop_idx = i
            # exclude the found point
            pot_start = self.diff_changes_idx[i] + 1

            l = prev_stop - gain_intervals[-1][0]
            range_len_tmp = int(np.ceil(l * self.region_range_in_percent /
                                        100))
            if range_len_tmp != 0:
                region_range_before = range_len_tmp
            # the region before would be empty
            else:
                i += 1
                continue

            # determine the region before the potention gain stage change
            start_before = int(prev_stop - region_range_before)
            if start_before < 0:
                start_before = 0

            roi_before = data_a[start_before:prev_stop]

            # determine the region after the potention gain stage change
            stop_after = pot_start + self.region_range
            if stop_after > self.diff.size:
                stop_after = self.diff.size

            roi_after = data_a[pot_start:stop_after]

            # check if the following idx is contained in roi_before
            condition = start_before < self.diff_changes_idx[:prev_stop_idx]
            near_matches_before = np.where(condition)
            # np.where returns a tuple (array, type)
            near_matches_before = near_matches_before[0]

            # check if the following idx is contained in roi_after
            condition = self.diff_changes_idx[i + 1:] < stop_after
            near_matches_after = np.where(condition)
            # np.where returns a tuple (array, type)
            near_matches_after = near_matches_after[0]
            if roi_before.size == 0:
                mean_before = data_a[prev_stop]
            elif np.all(np.isnan(roi_before)):
                roi_before = np.array([])
                mean_before = np.nan
            else:
                mean_before = np.nanmean(roi_before)

            if roi_after.size == 0:
                mean_after = data_a[pot_start]
            elif np.all(np.isnan(roi_after)):
                roi_after = np.array([])
                mean_after = np.nan
            else:
                mean_after = np.nanmean(roi_after)

            if self.use_debug:
                print("\n")
                print("gain intervals {}".format(gain_intervals))
                print("prev_stop: {}".format(prev_stop))
                print("pot_start: {}".format(pot_start))

                print("region_range_before: {}".format(region_range_before))
                print("roi_before {}".format(roi_before))
                print("roi_after: {}".format(roi_after))

                diff_idx = self.diff_changes_idx[:prev_stop_idx]
                print("near match before {} {}"
                      .format(near_matches_before,
                              diff_idx[near_matches_before]))

                diff_idx = self.diff_changes_idx[i + 1:]
                print("near match after {} {}"
                      .format(near_matches_after,
                              diff_idx[near_matches_after]))

                print("mean_before {}".format(mean_before))
                print("mean_after {}".format(mean_after))

            if near_matches_before.size == 0 and near_matches_after.size == 0:

                # if the slope is too high the mean is falsified by this
                condition = (np.max(roi_before) - np.min(roi_before) >
                             self.safety_factor * 2)
                if (roi_before.size != 0 and condition):
                    # cut the region of interest into half
                    new_region = roi_before[roi_before.size // 2:]
                    mean_before = np.nanmean(new_region)

                    if self.use_debug:
                        print("mean_before after cut down of "
                              "region of interest: {}"
                              .format(mean_before))

                if mean_before > mean_after + self.safety_factor:
                    # a stage change was found
                    gain_intervals[-1][1] = prev_stop
                    gain_intervals.append([pot_start, 0])

                i += 1
                set_stop_flag = True

            else:
                if gain_intervals[-1][1] == 0:
                    if self.use_debug:
                        print("gain intervals last entry == 0")

                    if near_matches_before.size != 0:
                        diff_idx = self.diff_changes_idx[:prev_stop_idx]
                        region_start = diff_idx[near_matches_before[-1]] + 1

                        roi_before = data_a[region_start:prev_stop]

                        if roi_before.size == 0:
                            mean_before = data_a[prev_stop]
                        elif np.all(np.isnan(roi_before)):
                            mean_before = np.nan
                        else:
                            mean_before = np.nanmean(roi_before)

                        if self.use_debug:
                            print("near_matches_before is not empty")
                            print("region_start {}".format(region_start))
                            print("roi_before {}".format(roi_before))
                            print("mean_before {}".format(mean_before))

                    if near_matches_before.size == 0:
                        if self.use_debug:
                            print("near_matches_before.size == 0")

                        if mean_before > mean_after + self.safety_factor:
                            if self.use_debug:
                                print("mean check")

                            gain_intervals[-1][1] = prev_stop

                            # prevent an infinite loop
                            intv = [prev_stop, pot_start]
                            if last_iteration_borders == intv:
                                if iteration_check >= 10:
                                    error_code[self.current_idx] = 4
                                    msg = "Breaking infinite loop"
                                    raise InfiniteLoopNumberError(msg)
                                iteration_check += 1
                            else:
                                last_iteration_borders = intv
                                iteration_check = 0

                            i += near_matches_after[-1]
                            set_stop_flag = False
                            continue
                        # for diff changes where there is a jump right after
                        # (down, up, stay, stay, jump,...)
                        # cut off the area after the jump and try again
                        elif near_matches_after.size != 0:
                            diff_idx = self.diff_changes_idx[i + 1:]
                            region_start = diff_idx[near_matches_after[-1]]
                            roi_after = data_a[region_start:stop_after]

                            if roi_after.size == 0:
                                mean_after = data_a[pot_start]
                            elif np.all(np.isnan(roi_after)):
                                mean_after = np.nan
                            else:
                                mean_after = np.nanmean(roi_after)

                            if self.use_debug:
                                print("near_matches_after is not empty")
                                print("region_start {}".format(region_start))
                                print("roi_after {}".format(roi_after))
                                print("mean_after {}".format(mean_after))

                            if mean_before > mean_after + self.safety_factor:
                                gain_intervals[-1][1] = prev_stop

                                # prevent an infinite loop
                                intv = [prev_stop, pot_start]
                                if last_iteration_borders == intv:
                                    if iteration_check >= 10:
                                        error_code[self.current_idx] = 4
                                        msg = "Breaking infinite loop"
                                        raise InfiniteLoopNumberError(msg)
                                    iteration_check += 1
                                else:
                                    last_iteration_borders = intv
                                    iteration_check = 0

                                i += near_matches_after[-1]
                                set_stop_flag = False
                                continue
                            # if there is an outlier right before an jump but
                            # the slope is so steep that not considering the
                            # region between the outlier and the jump would
                            # falsify the jump detection
                            elif mean_before + self.safety_factor < mean_after:
                                if self.use_debug:
                                    print("mean before is much bigger than "
                                          "mean after")

                                set_stop_flag = True
                            else:
                                set_stop_flag = False
                    else:
                        if mean_before > mean_after + self.safety_factor:
                            gain_intervals[-1][1] = prev_stop

                            # prevent an infinite loop
                            intv = [prev_stop, pot_start]
                            if last_iteration_borders == intv:
                                if iteration_check >= 10:
                                    error_code[self.current_idx] = 4
                                    msg = "Breaking infinite loop"
                                    raise InfiniteLoopNumberError(msg)
                                iteration_check += 1
                            else:
                                last_iteration_borders = intv
                                iteration_check = 0

                            if near_matches_after.size != 0:
                                # do not add +1 here (like it is done if the
                                # interval is closed) because this will result
                                # in one jump in the idx to many
                                i += near_matches_after[-1]
                                set_stop_flag = False

                            # do not increase i
                            continue

                else:
                    if near_matches_after.size != 0:
                        # because near_matches_after starts with 0
                        # -> infinite loop without the +1
                        i += 1 + near_matches_after[-1]
                        set_stop_flag = False
                        continue

                    elif mean_before > mean_after + self.safety_factor:
                        if not set_stop_flag:
                            set_stop_flag = True

                        gain_intervals.append([pot_start, 0])
                i += 1

        if gain_intervals[-1][1] == 0:
            gain_intervals[-1][1] = self.diff.size + 1
        else:
            gain_intervals.append([pot_start, self.diff.size + 1])

        # map the found indices back to the origin data
        mapped_gain_intervals = []
        for idx in np.arange(len(gain_intervals) - 1):
            start = origin_idx[gain_intervals[idx][0]]
            stop = origin_idx[gain_intervals[idx][1]]
            mapped_gain_intervals.append([start, stop])

        start = origin_idx[gain_intervals[-1][0]]
        stop = origin_idx[-1]
        mapped_gain_intervals.append([start, stop])

        gain_intervals = mapped_gain_intervals

        if self.use_debug:
            print("{}, found gain intervals {}"
                  .format(self.current_idx, gain_intervals))
            print("len gain intervals {}".format(len(gain_intervals)))

        if len(gain_intervals) > 3:
            error_code[self.current_idx] = 2
            raise GainStageNumberError("Too many gain stages found: {}"
                                       .format(gain_intervals))

        if len(gain_intervals) < 3:
            error_code[self.current_idx] = 3
            raise GainStageNumberError("Not enough gain stages found: {}"
                                       .format(gain_intervals))

        # store current_idx dependent results in the final matrices
        try:
            self.result["collection"]["diff_changes_idx"][self.current_idx] = (
                self.diff_changes_idx)
        except ValueError:
            # it is not fixed how many diff_changes indices are found
            # but the size of np.arrays has to be fixed
            l = len(self.diff_changes_idx)
            i = self.result["collection"]["diff_changes_idx"][self.current_idx]

            # fill up with found ones, rest of the array stays NAN
            if l < self.n_diff_changes_stored:
                i[:l] = self.diff_changes_idx
            # more regions found than array can hold, only store the first part
            else:
                i = self.diff_changes_idx[:self.n_diff_changes_stored]

        self.result["collection"]["len_diff_changes_idx"][self.current_idx] = (
            len(self.diff_changes_idx))

        gain_stages = self.result["intervals"]["gain_stages"]

        gain_stages[self.gain_idx["high"]] = gain_intervals[0]
        gain_stages[self.gain_idx["medium"]] = gain_intervals[1]

        new_gain_stage = self.detect_saturation(gain_intervals[2])
        gain_stages[self.gain_idx["low"]] = new_gain_stage

    def detect_saturation(self, interval):

        idx = (self.current_idx[0],
               self.current_idx[1],
               self.current_idx[2],
               slice(interval[0], interval[1]))

        sat_index = np.where(self.analog[idx] >= self.saturation_threshold)[0]

        # check if the result of where was empty
        if sat_index.size == 0:
            sat_index = interval[1]
        else:
            sat_index = interval[0] + sat_index[0]

        new_interval = [interval[0], sat_index]

        # second iteration for saturation detection
        new_gain_interval, new_sat_interval = (
            self.detect_saturation_with_diff(new_interval))

        self.result["intervals"]["saturation"][self.current_idx] = (
            [new_sat_interval[0], interval[1]])

        return new_gain_interval

    def detect_saturation_with_diff(self, interval):
        diff_interval = self.diff[interval[0]:interval[1]]

        condition = np.absolute(diff_interval) < self.saturation_threshold_diff
        sat_indices = np.where(condition)[0]

        if sat_indices.size != 0:
            j = sat_indices[-1]
            for i in sat_indices[::-1]:
                if i == j:
                    j -= 1
                else:
                    break

            saturation_interval = [interval[0] + i, interval[1]]
            new_gain_interval = [interval[0] + sat_indices[0], interval[0] + i]

            return new_gain_interval, saturation_interval
        else:
            return interval, [interval[1], interval[1]]

    def fit_gain(self, gain, interval, cut_off_ends):

        # the interval is splitted into sub-intervals
        intervals = self.split_interval(interval, self.n_intervals[gain])

        for i in np.arange(len(intervals)):
            self.fit_data(gain, intervals[i], i, cut_off_ends=cut_off_ends)

        return intervals

    def split_interval(self, interval, nsplits):
        """
        split given interval into equaly sized sub-intervals
        """

        step = int((interval[-1] - interval[0]) / nsplits)

        if step == 0:
            if self.use_debug:
                print("current_idx {}".format(self.current_idx))
                print("nsplits {}".format(nsplits))
                print("interval={}".format(interval))
            self.result["error_code"][self.current_idx] = 7
            raise IntervalSplitError("Interval has not enough points to split")

        splitted_intervals = []
        for i in np.arange(nsplits):
            start = interval[0] + i * step
            stop = interval[0] + (i + 1) * step
            splitted_intervals.append([start, stop])

        # due to rounding or int convertion the last calculated interval ends
        # before the end of the given interval
        splitted_intervals[-1][1] = interval[1]

        return splitted_intervals

    def fit_data(self, gain, interval, interval_idx, cut_off_ends=False):

        # find the inner points (cut off the top and bottom part
        if cut_off_ends:
            tmp = np.arange(interval[0], interval[1])
            # 100.0 is needed because else it casts it as ints, i.e. 10/100=>0
            lower_border = int(tmp[0] + len(tmp) * self.percent / 100.0)
            upper_border = int(tmp[0] + len(tmp) * (1 - self.percent / 100.0))
        else:
            lower_border = interval[0]
            upper_border = interval[1]

        # transform the problem y = mx + c
        # into the form y = Ap, where A = [[x 1]] and p = [[m], [c]]
        # meaning  data_to_fit = A * [slope, offset]
        y = self.analog[self.current_idx[0],
                        self.current_idx[1],
                        self.current_idx[2],
                        lower_border:upper_border].astype(np.float)

        self.x_values[gain][interval_idx] = np.arange(lower_border,
                                                      upper_border)
        # TODO check that size of inteval is not to small, i.e. < 3

        # scaling
        self.scale_x_interval(gain, interval_idx)

        # find out if the col was effected by frame loss
        lost_frames = np.where(y == 0)
        y[lost_frames] = np.NAN

        # remove the ones with frameloss
        missing = np.isnan(y)
        self.data_to_fit[gain][interval_idx] = y[~missing]
        x = self.x_values[gain][interval_idx][~missing]

        number_of_points = len(x)

        # .T means transposed
        A = np.vstack([x, np.ones(number_of_points)]).T

        if A.size <= 1 or self.data_to_fit[gain][interval_idx].size <= 1:

            self.result["error_code"][self.current_idx] = 6

            msg = ("{}: Not enough point to fit {} gain "
                   "(number of point available: {})"
                   .format(self.current_idx, gain,
                           self.data_to_fit[gain][interval_idx].size))
            if self.use_debug:
                self.log.debug(msg)

            raise FitError(msg)

        # fit the data
        # reason to use numpy lstsq:
        # https://stackoverflow.com/questions/29372559/what-is-the-difference-between-numpy-linalg-lstsq-and-scipy-linalg-lstsq  # noqa E501
        # lstsq returns: Least-squares solution (i.e. slope and offset),
        #                residuals,
        #                rank,
        #                singular values
        array_idx = self.current_idx + (interval_idx,)
        res = None
        try:
            res = np.linalg.lstsq(A, self.data_to_fit[gain][interval_idx])

            self.result["slope"]["individual"][gain][array_idx] = res[0][0]
            self.result["offset"]["individual"][gain][array_idx] = res[0][1]
            self.result["residual"]["individual"][gain][array_idx] = res[1]
            self.result["average_residual"]["individual"][gain][array_idx] = (
                np.sqrt(res[1] / number_of_points))

        except:
            if res is None:
                print("interval\n{}".format(interval))
                print("A\n{}".format(A))
                print("self.data_to_fit[{}][{}]\n{}"
                      .format(gain, interval_idx,
                              self.data_to_fit[gain][interval_idx]))
                raise

            if res[0].size != 2:
                self.result["error_code"][self.current_idx] = 5
                msg = "Failed to calculate slope and offset"

                if self.use_debug:
                    print(msg)

                raise FitError(msg)

            elif res[1].size != 1:
                self.result["warning_code"][self.current_idx] = 1
                msg = "Failed to calculate residual"

                if self.use_debug:
                    self.log.debug("interval {}".format(interval))
                    self.log.debug(res)
                    self.log.debug(
                        "self.data_to_fit[{}][{}] {}"
                        .format(gain,
                                interval_idx,
                                self.data_to_fit[gain][interval_idx]))
                    self.log.debug(msg)

                raise FitError(msg)

            else:
                print("interval\n{}".format(interval))
                print("A\n{}".format(A))
                print("self.data_to_fit[{}][{}]\n{}"
                      .format(gain, interval_idx,
                              self.data_to_fit[gain][interval_idx]))
                print("res {}".format(res))

                raise

    def calc_gain_median(self):
        for gain in ["high", "medium", "low"]:
            gain_stages = self.result["intervals"]["gain_stages"]
            start = gain_stages[self.gain_idx[gain]][0]
            stop = gain_stages[self.gain_idx[gain]][1]
            idx = (self.current_idx[0],
                   self.current_idx[1],
                   self.current_idx[2],
                   slice(start, stop))

            self.result["medians"][self.gain_idx[gain]] = (
                np.median(self.digital[idx]))

    def calc_thresholds(self):
        # threshold between high and medium
        high = self.result["medians"][self.gain_idx["high"]]
        medium = self.result["medians"][self.gain_idx["medium"]]
        low = self.result["medians"][self.gain_idx["low"]]

        self.result["thresholds"][(0, ) + self.current_idx] = (
            np.mean([high, medium]))

        # threshold between medium and low
        self.result["thresholds"][(1, ) + self.current_idx] = (
            np.mean([medium, low]))

    def scale_full_x_axis(self):
        lower = np.arange(self.scaling_point)
        upper = (np.arange(self.scaling_point, self.digital.shape[3])
                 * self.scaling_factor
                 - self.scaling_point * self.scaling_factor
                 + self.scaling_point)

        self.scaled_x_values = np.concatenate((lower, upper))

    def scale_x_interval(self, gain, interval_idx):
        # tmp variables to improve reading
        condition = self.x_values[gain][interval_idx] > self.scaling_point
        indices_to_scale = np.where(condition)
        x = self.x_values[gain][interval_idx][indices_to_scale]
        # shift x value to root, scale, shift back
        # e.g. x_new = (x_old - 200) * 10 + 200
        scaled = ((x - self.scaling_point) * self.scaling_factor +
                  self.scaling_point)
        self.x_values[gain][interval_idx][indices_to_scale] = scaled
