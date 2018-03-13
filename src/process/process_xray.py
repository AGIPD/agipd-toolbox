import numpy as np
from process_base import ProcessBase
from scipy.signal import convolve
from scipy.optimize import curve_fit
#import peakutils

class ProcessXray(ProcessBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initiate(self):

        n_offsets = len(self.runs)

        self.shapes = {
            "photon_spacing": (self.n_memcells,
                               self.n_rows,
                               self.n_cols)
        }

        self.result = {
            "photon_spacing": {
                "data": np.empty(self.shapes["photon_spacing"]),
                "path": "photon_spacing",
                "type": np.int16
            },
            "spacing_error": {
                "data": np.empty(self.shapes["photon_spacing"]),
                "path": "spacing_error",
                "type": np.int16
            },
            "peak_stddev": {
                "data": np.empty(self.shapes["photon_spacing"]),
                "path": "peak_stddev",
                "type": np.float
            },
            "peak_error": {
                "data": np.empty(self.shapes["photon_spacing"]),
                "path": "peak_error",
                "type": np.float
            },
            "quality": {
                "data": np.empty(self.shapes["photon_spacing"]),
                "path": "quality",
                "type": np.int16
            }
        }

    def calculate(self):
        for i, run_number in enumerate(self.runs):
            in_fname = self.in_fname.format(run_number=run_number)

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

            for mc in range(self.n_memcells):
                print("memcell {}".format(mc))
                for row in range(self.n_rows):
                    for col in range(self.n_cols):

                        try:
                            #(photon_spacing, spacing_error, peak_stddev, peak_errors, quality) = self.get_photon_spacing(analog)
                            photon_spacing = self.get_photon_spacing(analog)                           
                            idx = (mc, row, col)
                            self.result["photon_spacing"]["data"][idx] = photon_spacing
                            #self.result["spacing_error"]["data"][idx] = spacing_error
                            #self.result["peak_stddev"]["data"][idx] = peak_stddev
                            #self.result["peak_error"]["data"][idx] = peak_error
                            #self.result["quality"]["data"][idx] = quality


                        except:
                            print("memcell, row, col", mc, row, col)
                            print("analog.shape", analog.shape)
                            raise



            print("Done.")

    
    def find_peaks(self, hist, bins, npeaks, window=25):
        # stupid peak-finding function
        # finds max value of hist, the removes data 
        # from peak +/- window, continue for npeaks

        rest_data = hist
        rest_bins = bins[:-1]
        pk = np.empty(npeaks)
        pk_height = np.empty(npeaks)

        for n in range(0, npeaks):

            #find peak
            pk_height[n] = np.amax(rest_data)
            pk[n] = rest_bins[np.argmax(rest_data)]
            print(pk[n])
    
            # remove peak
            print(len(rest_bins))
            print(len(rest_data))
            print(rest_bins)
            print(np.where(rest_bins > (pk[n]+25)))
            rest_data = rest_data[np.where(rest_bins > (pk[n]+25))]
            rest_bins = rest_bins[-(len(rest_data)):]

        # sort peaks by size
        sorted_indices = np.argsort(pk_height)[::-1]
        pk_height_sorted = pk_height[sorted_indices]
        pk_sorted = pk[sorted_indices]

        return pk_sorted, pk_height_sorted


    def gauss(self, x, *p):
        a, b, c = p
        y = a*np.exp(-np.power((x - b), 2.)/(2. * c**2.))
    
        return y
    

    def fit_peaks(self, hist, bins, npeaks, initial):

        bin_width = bins[1] - bins[0]
        window_bins = int(25/bin_width)
        fit_window = np.empty(npeaks, dtype='int')
        
        popt = np.empty(npeaks)
        pcov = np.empty(npeaks)
        # loop over peaks, do fits
        for i in range(0, npeaks):

            p_initial = [initial[0][i], initial[1][0], 10]
            fit_window[i] = np.where(bins==initial[0][i])[0]

            x_fit = bins[fit_window[i]-window_bins:fit_window[i]+window_bins]

            h_fit = hist[fit_window[i]-window_bins:fit_window[i]+window_bins]
            popt[i], pcov[i] = curve_fit(self.gauss, x_fit, h_fit, p0=p_initial)
        
        return popt, pcov
        


    def get_photon_spacing(self, analog):
    #def get_photon_spacing(self, analog, lowpass=True):
        #for one pixel

        #failed = (0, 0, (0, 0), (0, 0), 0)
        failed = (0, (0, 0), 0)

        bins = np.arange(np.min(analog), np.max(analog), 1, dtype='int16')
        (hist, _) = np.histogram(analog, bins)
        smooth_window = 11
        hist_smooth = convolve(hist, np.ones((smooth_window,)), mode='same')
            
        npeaks = 2
        # find starting peak locations, heights
        peak_locations, peak_sizes = self.find_peaks(hist_smooth, bins, npeaks)
        initial = [peak_locations, peak_sizes]

        # fit peaks with gaussian
        params, pcov = self.fit_peaks(hist_smooth, bins, npeaks, initial)

        fit_peak_locations = params[:][0]
        fit_peak_sizes = params[:][1]
                
        photon_spacing = np.abs(fit_peak_locations[1] - fit_peak_locations[0])

        if photon_spacing <= 10:
            return failed

        #indices_between_peaks = np.sort(sorted_peak_indices[0:2])
        #quality = sorted_peak_sizes[1] - np.min(hist_smooth[indices_between_peaks[0]:indices_between_peaks[1]])
            
        #peak_stddev = sorted_peak_stddev[0:2]
            
        #return (photon_spacing, spacing_error, peak_stddev, peak_errors, quality)
        return (photon_spacing)




#######################################################################################
# Yaroslav's functions
#######################################################################################
#
#    def orderFilterCourse(self, data, window_radius, order, lowpass_points):
#        if data.ndim != 1:
#            raise ValueError('data must be one dimensional')
#
#        if window_radius%2 == 0:
#            window_radius -= 1
#
#        sample_positions = np.round(np.linspace(window_radius, data.size - window_radius, lowpass_points)).astype(int)
#        sample_values = np.zeros((sample_positions.size,))
#        for i in np.arange(sample_positions.size):
#            neighbourhood = data[(sample_positions[i] - window_radius):(sample_positions[i] + window_radius + 1)]
#            sample_values[i] = np.sort(neighbourhood)[order]
#
#        lowpass_data = np.interp(np.arange(data.size), sample_positions, sample_values, left = sample_values[0], right = sample_values[-1])
#        return lowpass_data
#
#
#    def lowpass_correction(self, analog, locality_radius=800, lowpass_points=1000):
#        data = analog
#
#        local_data = data[2 * locality_radius:6 * locality_radius]
#        local_bin_edges = np.arange(np.min(local_data), np.max(local_data))
#        
#        (local_hist, _) = np.histogram(local_data, local_bin_edges)
#        smooth_window_size = 11
#        local_hist_smooth = convolve(local_histogram, np.ones((smooth_window_size,)), mode='same')
#        
#        most_frequent_value = np.mean(local_bin_edges[np.argmax(local_hist_smooth)]).astype(int)
#        
#        # set order to be the same order as the maximum of the histogram in the local data. Should be least affected by noise
#        order = (2 * locality_radius * np.mean(np.nonzero(np.sort(local_data) == most_frequent_value)) / local_data.shape[0])
#        
#        if np.isnan(order):
#            order = np.array(0.4 * 2 * locality_radius)
#
#        order = order.astype(int)
#        lowpass_data = orderFilterCourse(data, locality_radius, order, lowpass_points)
#        corrected_data = data - lowpass_data
#        bin_edges = np.arange(np.min(corrected_data), np.max(corrected_data))
#        
#        return np.histogram(corrected_data, bin_edges)
#
#
#    def indexes_peakutilsManuallyAdjusted(self, y, thres=0.3, min_dist=1):
#        '''Peak detection routine.
#        
#        Finds the peaks in *y* by taking its first order difference. By using
#        *thres* and *min_dist* parameters, it is possible to reduce the number of
#        detected peaks. *y* must be signed.
#        
#        Parameters
#        ----------
#        y : ndarray (signed)
#            1D amplitude data to search for peaks.
#        thres : float between [0., 1.]
#            Normalized threshold. Only the peaks with amplitude higher than the
#            threshold will be detected.
#        min_dist : int
#            Minimum distance between each detected peak. The peak with the highest
#            amplitude is preferred to satisfy this constraint.
#            
#        Returns
#        -------
#        ndarray
#            Array containing the indexes of the peaks that were detected
#            
#        '''
#        if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.unsignedinteger):
#            raise ValueError("y must be signed")
#
#        thres = thres * (np.max(y) - np.min(y)) + np.min(y)
#        min_dist = int(min_dist)
#        
#        # find the peaks by using the first order difference
#        dy = np.diff(y)
#        peaks = np.where((np.hstack([dy, 0.]) <= 0.)  # Addition by yaroslav.gevorkov@desy.de: instead of <, here <=
#                         & (np.hstack([0., dy]) >= 0.)
#                         & (y > thres))[0]
#
#        if peaks.size > 1 and min_dist > 1:
#            highest = peaks[np.argsort(y[peaks])][::-1]
#            rem = np.ones(y.size, dtype=bool)
#            rem[peaks] = False
#            
#            for peak in highest:
#                if not rem[peak]:
#                    sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
#                    rem[sl] = True
#                    rem[peak] = False
#                    
#            peaks = np.arange(y.size)[~rem]
#
#        return peaks
#
#
#    def interpolate_peakutilsManuallyAdjusted(self, x, y, ind, width, func):
#        '''Tries to enhance the resolution of the peak detection by using
#        Gaussian fitting, centroid computation or an arbitrary function on the
#        neighborhood of each previously detected peak index.
#        
#        Parameters
#        ----------
#        x : ndarray
#            Data on the x dimension.
#        y : ndarray
#            Data on the y dimension.
#        ind : ndarray
#            Indexes of the previously detected peaks. If None, indexes() will be
#            called with the default parameters.
#        width : int
#            Number of points (before and after) each peak index to pass to *func*
#            in order to encrease the resolution in *x*.
#        func : function(x,y)
#            Function that will be called to detect an unique peak in the x,y data.
#
#        Returns
#        -------
#        ndarray :
#            Array with the adjusted peak positions (in *x*)
#        '''
#
#        out = []
#        for slice_ in (slice(max((0,i - width)), min((x.size, i + width))) for i in ind):
#            fit = func(x[slice_], y[slice_])
#            out.append(fit)
#            
#        return np.array(out)
#
#
########################################################################################



