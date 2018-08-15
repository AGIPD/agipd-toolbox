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

from collections import namedtuple

from process_base import ProcessBase

import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


class ProcessXray(ProcessBase):
    Res = namedtuple("res", ["photon_spacing",
                             "spacing_error",
                             "peak_stddev",
                             "peak_error",
                             "quality"])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initiate(self):

        n_offsets = len(self.runs)

        self.shapes = {
            "photon_spacing": (self.n_rows,
                               self.n_cols,
                               self.n_memcells,
                               1),

            "stddev":         (self.n_rows,
                               self.n_cols,
                               self.n_memcells,
                               2)
        }

        self.transpose_order = (3,
                                self._memcell_location,
                                self._row_location,
                                self._col_location)

        self.result = {
            "photon_spacing": {
                "data": np.empty(self.shapes["photon_spacing"]),
                "path": "photon_spacing",
                "type": np.int16
            },
            "spacing_error": {
                "data": np.empty(self.shapes["photon_spacing"]),
                "path": "spacing_error",
                "type": np.float
            },
            "peak_stddev": {
                "data": np.empty(self.shapes["stddev"]),
                "path": "peak_stddev",
                "type": np.float
            },
            "peak_error": {
                "data": np.empty(self.shapes["stddev"]),
                "path": "peak_error",
                "type": np.float
            },
            "quality": {
                "data": np.empty(self.shapes["photon_spacing"]),
                "path": "quality",
                "type": np.int16
            }
        }


    def gauss(self, x, *p):
        a, b, c = p
        y = a*np.exp(-np.power((x - b), 2.)/(2. * c**2.))

        return y


    def fit_peaks(self, hist, bins, npeaks, initial):

        bin_width = bins[1] - bins[0]
        window_bins = int(25/bin_width)
        fit_window = np.empty(npeaks, dtype='int')

        popt = []
        pcov = []
        # loop over peaks, do fits
        for i in range(0, npeaks):

            p_initial = [initial[0][i], initial[1][i], 10]
            fit_window[i] = np.where(bins==initial[1][i])[0]

            x_fit = bins[fit_window[i]-window_bins:fit_window[i]+window_bins]

            h_fit = hist[fit_window[i]-window_bins:fit_window[i]+window_bins]
            res = curve_fit(self.gauss, x_fit, h_fit, p0=p_initial)
            popt.append(res[0])
            pcov.append(res[1])


        return popt, pcov


    def get_photon_spacing(self, analog, row, col, mc):
        #for one pixel one memcell

        failed = ProcessXray.Res(photon_spacing=0,
                                 spacing_error=0,
                                 peak_stddev=(0,0),
                                 peak_error=(0,0),
                                 quality=0)

        bins = np.arange(np.min(analog), np.max(analog), 1, dtype='int16')
        (hist, _) = np.histogram(analog, bins)
        smooth_window = 11
        hist_smooth = signal.convolve(hist, np.ones((smooth_window,)), mode='same')

        # find starting peak locations, heights
        peak_loc_bins = signal.find_peaks_cwt(hist_smooth, np.arange(10,70))
        peak_loc_bins = np.array(peak_loc_bins)

        if len(peak_loc_bins)==0:
            print("ERROR: No peaks found!!")
            return failed

        peak_locations = bins[peak_loc_bins]
        peak_sizes = hist_smooth[peak_loc_bins]

        # find_peaks_cwt also finds many spurious peaks, filter these out
        # define minimum peak height
        min_height = 30
        peak_loc_bins_filtered = peak_loc_bins[np.where(peak_sizes > min_height)]
        peak_sizes_filtered = peak_sizes[np.where(peak_sizes > min_height)]
        peak_locations_filtered = peak_locations[np.where(peak_sizes > min_height)]

        npeaks = len(peak_locations_filtered)
        if npeaks < 2:
            #print("ERROR: Fewer than 2 peaks found!", row, col, mc)
            return failed

        # fit 0- and 1-photon peaks with gaussian
        initial = [peak_sizes_filtered, peak_locations_filtered]
        params, pcov = self.fit_peaks(hist_smooth, bins, 2, initial)
        results = list(zip(params[0], params[1]))
        param_errors = [np.sqrt(np.diag(p)) for p in pcov]
        errors = list(zip(param_errors[0], param_errors[1]))

        photon_spacing = np.abs(results[1][1] - results[1][0])
        spacing_error = np.sqrt(np.power(errors[1][1], 2) + np.power(errors[1][0], 2))
        fit_results = ProcessXray.Res

        if photon_spacing <= 10:
            return failed

        quality =  results[0][0] - np.min(hist_smooth[peak_loc_bins_filtered[0]:peak_loc_bins_filtered[1]])

        return ProcessXray.Res(photon_spacing=photon_spacing,
                               spacing_error=spacing_error,
                               peak_stddev=results[2],
                               peak_error=errors[2],
                               quality=quality)


    def calculate(self):
        for i, run_number in enumerate(self.runs):
            run_name = self.run_names[i]
            in_fname = self.in_fname.format(run_number=run_number, run_name=run_name)

            print("Start loading data from {} ... ".format(in_fname),
                  end="", flush=True)
            analog, digital = self.load_data(in_fname)
            print("Done.")

            print("Start masking problems ... ", end="", flush=True)
            m_analog, m_digital = self.mask_out_problems(analog=analog,
                                                         digital=digital)
            print("Done.")

            print("Start computing photon spacing ... ",
                  end="", flush=True)

            failed_count = 0
            for row in range(self.n_rows):
                print("row ", row)

                for col in range(self.n_cols):

                    for mc in range(self.n_memcells):

                        idx = (row, col, mc)
                        if np.all(np.ma.getmask(m_analog)[row, col, mc, :]):
                            self.result["photon_spacing"]["data"][idx] = 0
                            self.result["spacing_error"]["data"][idx] = 0
                            self.result["peak_stddev"]["data"][idx] = 0
                            self.result["peak_error"]["data"][idx] = 0
                            self.result["quality"]["data"][idx] = 0
                        else:
                            try:
                                fit_result = self.get_photon_spacing(m_analog[row, col, mc, :], row, col, mc)
                                self.result["photon_spacing"]["data"][idx] = fit_result.photon_spacing
                                self.result["spacing_error"]["data"][idx] = fit_result.spacing_error
                                self.result["peak_stddev"]["data"][idx] = fit_result.peak_stddev
                                self.result["peak_error"]["data"][idx] = fit_result.peak_error
                                self.result["quality"]["data"][idx] = fit_result.quality


                            except:
                                print("ERROR: Fit failed!")
                                print("memcell, row, col", mc, row, col)
                                print("analog.shape", analog.shape)
                                self.result["photon_spacing"]["data"][idx] = 0
                                self.result["spacing_error"]["data"][idx] = 0
                                self.result["peak_stddev"]["data"][idx] = 0
                                self.result["peak_error"]["data"][idx] = 0
                                self.result["quality"]["data"][idx] = 0
                                #raise


                        if self.result["photon_spacing"]["data"][idx] == 0:
                            failed_count = failed_count + 1

            self.result["photon_spacing"]["data"] = self.result["photon_spacing"]["data"].transpose(self.transpose_order)
            self.result["spacing_error"]["data"] = self.result["spacing_error"]["data"].transpose(self.transpose_order)
            self.result["peak_stddev"]["data"] = self.result["peak_stddev"]["data"].transpose(self.transpose_order)
            self.result["peak_error"]["data"] = self.result["peak_error"]["data"].transpose(self.transpose_order)
            self.result["quality"]["data"] = self.result["quality"]["data"].transpose(self.transpose_order)

            print("Done.")
            total_fits = self.n_rows * self.n_cols * self.n_memcells
            print("Failed fits: ", failed_count, " = ", (failed_count/total_fits)*100, "%")
