import h5py
import sys
import numpy as np
import time
import os
from datetime import date
import peakutils

try:
    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
except:
    CURRENT_DIR = os.path.dirname(os.path.realpath('__file__'))

BASE_PATH = os.path.dirname(os.path.dirname(CURRENT_DIR))
SRC_PATH = os.path.join(BASE_PATH, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import utils  # noqa E402
from _version import __version__


class ProcessBase(object):
    def __init__(self, in_fname, out_fname, runs):

        self._out_fname = out_fname

        # public attributes for use in inherited classes
        self.in_fname = in_fname

        self.runs = runs

        self._row_location = None
        self._col_location = None
        self._memcell_location = None
        self._frame_location = None
        self._set_data_order()

        self._set_dims_and_metadata()

        self.shapes = {}
        self.result = {}

        print("\n\n\nStart process")
        print("in_fname:", self.in_fname)
        print("out_fname:", self._out_fname)
        print("module, channel:", self.module, self.channel)
        print()

        self.run()

    def _set_data_order(self):
        """Set the locations where the data is stored

        This give the different process methods the posibility to act genericly
        to data reordering.

        """
        self._row_location = 0
        self._col_location = 1
        self._memcell_location = 2
        self._frame_location = 3

    def _set_dims_and_metadata(self):
        in_fname = self.in_fname.format(run_number=self.runs[0])
        with h5py.File(in_fname, "r") as f:
            shape = f['analog'].shape

            self.module = f['collection/module'][()]
            self.channel = f['collection/channel'][()]

        self.n_rows = shape[self._row_location]
        self.n_cols = shape[self._col_location]
        self.n_memcells = shape[self._memcell_location]

    def load_data(self, in_fname):
        with h5py.File(in_fname, "r") as f:
            analog = f['analog'][()]
            digital = f['digital'][()]

        return analog, digital

    def initiate(self):
        pass

    def run(self):

        total_time = time.time()

        self.initiate()

        self.calculate()

        print("Start saving results at {} ... ".format(self._out_fname),
              end='')
        self.write_data()
        print("Done.")

        print("Process took time: {}\n\n".format(time.time() - total_time))

    def get_mask(self, analog, digital):

        # find out if the col was effected by frame loss
        return (analog == 0)

    def mask_out_problems(self, analog, digital, mask=None):

        if mask is None:
            mask = self.get_mask(analog, digital)

        # remove the ones with frameloss
        m_analog = np.ma.masked_array(data=analog, mask=mask)
        m_digital = np.ma.masked_array(data=digital, mask=mask)

        return m_analog, m_digital

    def calculate(self):
        pass

    def fit_linear(self, x, y, mask=None):
        if mask is None:
            y_masked = y
            x_masked = x
        else:
            y_masked = y[~mask]
            x_masked = x[~mask]

        number_of_points = len(x_masked)
        try:
            A = np.vstack([x_masked, np.ones(number_of_points)]).T
        except:
            print("number_of_points", number_of_points)
            print("x (after masking)", x_masked)
            print("y (after masking)", y_masked)
            print("len y_masked", len(y_masked))
            raise

        # lstsq returns: Least-squares solution (i.e. slope and offset),
        #                residuals,
        #                rank,
        #                singular values
        res = np.linalg.lstsq(A, y_masked)

        return res

    def fit_linear_old(self, x, y):
        # find out if the col was effected by frame loss
        lost_frames = np.where(y == 0)
        y[lost_frames] = np.NAN

        # remove the ones with frameloss
        missing = np.isnan(y)
        y = y[~missing]
        x = x[~missing]

        number_of_points = len(x)
        A = np.vstack([x, np.ones(number_of_points)]).T

        # lstsq returns: Least-squares solution (i.e. slope and offset),
        #                residuals,
        #                rank,
        #                singular values
        res = np.linalg.lstsq(A, y)

        return res

    def write_data(self):

        # convert into unicode
        if type(self.runs[0]) == str:
            used_run_numbers = [run.encode('utf8') for run in self.runs]
        else:
            used_run_numbers = ["r{:04d}".format(run).encode('utf8')
                                for run in self.runs]

        collection = {
            "run_number": used_run_numbers,
            "creation_date": str(date.today()),
            "version": __version__
        }

        with h5py.File(self._out_fname, "w", libver='latest') as f:
            for key, dset in self.result.items():
                f.create_dataset(dset['path'],
                                 data=dset['data'],
                                 dtype=dset['type'])

            prefix = "collection"
            for key, value in collection.items():
                name = "{}/{}".format(prefix, key)
                f.create_dataset(name, data=value)

            f.flush()


    def fit_gaussian(self, x, y):
        initial = [np.max(y), x[0], (x[1] - x[0]) * 5]
        try:
            params, pcov = curve_fit(peakutils.peak.gaussian, x, y, initial)
        except:
            return (0, 0)

        return (params[1], params[2])


    def orderFilterCourse(self, data, window_radius, order, lowpass_points):
        if data.ndim != 1:
            raise ValueError('data must be one dimensional')

        if window_radius%2 == 0:
            window_radius -= 1

        sample_positions = np.round(np.linspace(window_radius, data.size - window_radius, lowpass_points)).astype(int)
        sample_values = np.zeros((sample_positions.size,))
        for i in np.arange(sample_positions.size):
            neighbourhood = data[(sample_positions[i] - window_radius):(sample_positions[i] + window_radius + 1)]
            sample_values[i] = np.sort(neighbourhood)[order]

        lowpass_data = np.interp(np.arange(data.size), sample_positions, sample_values, left = sample_values[0], right = sample_values[-1])
        return lowpass_data


    def lowpass_correction(self, analog, locality_radius=800, lowpass_points=1000):
        data = analog

        local_data = data[2 * locality_radius:6 * locality_radius]
        local_bin_edges = np.arange(np.min(local_data), np.max(local_data))
        
        (local_hist, _) = np.histogram(local_data, local_bin_edges)
        smooth_window_size = 11
        local_hist_smooth = convolve(local_histogram, np.ones((smooth_window_size,)), mode='same')
        
        most_frequent_value = np.mean(local_bin_edges[np.argmax(local_hist_smooth)]).astype(int)
        
        # set order to be the same order as the maximum of the histogram in the local data. Should be least affected by noise
        order = (2 * locality_radius * np.mean(np.nonzero(np.sort(local_data) == most_frequent_value)) / local_data.shape[0])
        
        if np.isnan(order):
            order = np.array(0.4 * 2 * locality_radius)

        order = order.astype(int)
        lowpass_data = orderFilterCourse(data, locality_radius, order, lowpass_points)
        corrected_data = data - lowpass_data
        bin_edges = np.arange(np.min(corrected_data), np.max(corrected_data))
        
        return np.histogram(corrected_data, bin_edges)


    def indexes_peakutilsManuallyAdjusted(self, y, thres=0.3, min_dist=1):
        '''Peak detection routine.
        
        Finds the peaks in *y* by taking its first order difference. By using
        *thres* and *min_dist* parameters, it is possible to reduce the number of
        detected peaks. *y* must be signed.
        
        Parameters
        ----------
        y : ndarray (signed)
            1D amplitude data to search for peaks.
        thres : float between [0., 1.]
            Normalized threshold. Only the peaks with amplitude higher than the
            threshold will be detected.
        min_dist : int
            Minimum distance between each detected peak. The peak with the highest
            amplitude is preferred to satisfy this constraint.
            
        Returns
        -------
        ndarray
            Array containing the indexes of the peaks that were detected
            
        '''
        if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.unsignedinteger):
            raise ValueError("y must be signed")

        thres = thres * (np.max(y) - np.min(y)) + np.min(y)
        min_dist = int(min_dist)
        
        # find the peaks by using the first order difference
        dy = np.diff(y)
        peaks = np.where((np.hstack([dy, 0.]) <= 0.)  # Addition by yaroslav.gevorkov@desy.de: instead of <, here <=
                         & (np.hstack([0., dy]) >= 0.)
                         & (y > thres))[0]

        if peaks.size > 1 and min_dist > 1:
            highest = peaks[np.argsort(y[peaks])][::-1]
            rem = np.ones(y.size, dtype=bool)
            rem[peaks] = False
            
            for peak in highest:
                if not rem[peak]:
                    sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                    rem[sl] = True
                    rem[peak] = False
                    
            peaks = np.arange(y.size)[~rem]

        return peaks


    def interpolate_peakutilsManuallyAdjusted(self, x, y, ind, width, func):
        '''Tries to enhance the resolution of the peak detection by using
        Gaussian fitting, centroid computation or an arbitrary function on the
        neighborhood of each previously detected peak index.
        
        Parameters
        ----------
        x : ndarray
            Data on the x dimension.
        y : ndarray
            Data on the y dimension.
        ind : ndarray
            Indexes of the previously detected peaks. If None, indexes() will be
            called with the default parameters.
        width : int
            Number of points (before and after) each peak index to pass to *func*
            in order to encrease the resolution in *x*.
        func : function(x,y)
            Function that will be called to detect an unique peak in the x,y data.

        Returns
        -------
        ndarray :
            Array with the adjusted peak positions (in *x*)
        '''

        out = []
        for slice_ in (slice(max((0,i - width)), min((x.size, i + width))) for i in ind):
            fit = func(x[slice_], y[slice_])
            out.append(fit)
            
        return np.array(out)


    def get_photon_spacing(self, analog, lowpass=True):
        #for one pixel

        #failed = (0, 0, (0, 0), (0, 0), 0)
        failed = (0, (0, 0), 0)

        if lowpass:
            (hist, bins) = lowpass_correction(analog)
        else:
            bins = np.arange(np.min(analog), np.max(analog), 1, dtype='int16')
            (hist, _) = np.histogram(analog, bins)

        smooth_window = 11
        hist_smooth = convolve(hist, np.ones((smooth_window,)), mode='same')
            
        min_peak_dist = 25
        x = np.arange(len(hist))
        y = hist_smooth
        rough_peak_locations = indexes_peakutilsManuallyAdjusted(y, thres=0.05, min_dist=min_peak_dist)
        
        peak_width = 21
        try:
            interpolated_params = interpolate_peakutilsManuallyAdjusted(x, y, ind=rough_peak_locations, width=peak_width, func=fit_gaussian)
        except:
            return failed
        if interpolated_params.size < 2:
            return failed

        (interpolated_peak_locations, peak_stddev) = zip(*interpolated_params)

        peak_locations = np.clip(interpolated_peak_locations, 0, len(hist) - 1)
        peakStdDev = np.array(peak_stddev)


        max_peak_relocation = 15
        valid_indices = np.abs(peak_locations - rough_peak_locations) <= max_peak_relocation
        peak_locations = peak_locations[valid_indices]
        peak_stddev = peak_stddev[valid_indices]
        
        if peak_locations.size < 2:
            return failed

        peak_indices = np.round(peak_locations).astype(int)

        peak_sizes = hist_smooth[peak_indices]
        sorted_indices = np.argsort(peak_sizes)[::-1]
        sorted_peak_locations = peak_locations[sorted_indices]
        sorted_peak_indices = peak_indices[sorted_indices]
        sorted_peak_sizes = peak_sizes[sorted_indices]
        
        sorted_peak_stddev = peak_stddev[sorted_indices]
        
        photon_spacing = np.abs(sorted_peak_locations[1] - sorted_peak_locations[0])

        if photon_spacing <= 10:
            return failed

        indices_between_peaks = np.sort(sorted_peak_indices[0:2])
        quality = sorted_peak_sizes[1] - np.min(hist_smooth[indices_between_peaks[0]:indices_between_peaks[1]])
            
        peak_stddev = sorted_peak_stddev[0:2]
            
        #return (photon_spacing, spacing_error, peak_stddev, peak_errors, quality)
        return (photon_spacing, peak_stddev, quality)
