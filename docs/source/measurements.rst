.. _measurements:

Measurement Types
=================

There are currently three, soon four different measurements which can be analyzed with the agipd-toolbox package.  

   1. Darks
   2. X-ray
   3. Pulse Capacitor
   4. Current Source (under construction)


Dark
----

The pedestals are found by taking dark images; that is, collecting data in the absence of any x-raysource. 10,000 dark images are acquired for every memory cell. For every memory cell, all 10,000 measurements are combined into a histogram. The median of the histogram is taken as the pedestal,and the standard deviation is the noise. The pedestal is used to correct the data, the noise is used to find and mask dead or noisy pixels.  This measurement is done for all three gain stages.  The pixels are forced into medium or low gain by setting specific registers.

The final output file will be saved in <module>/<temperature>/dark/dark_joined_constants_agipd.h5.  If a multimodule system is being calibrated, all modules will be joined into this file.  The output contains four datasets in addition to the metadata for each module:

- gainlevel_mean: The mean value of the gain stage value (often called "digital" data) for each gain stage.
- offset: The position of the noise peak in ADU for each gain stage.
- stddev: The standard deviation of the noise peak in ADU for each gain stage.
- threshold: The threshold between gain stages.  This is defined as the midpoint between the mean gain stage values.    
  
    





X-ray (Fluorescence)
--------------------

The x-ray fluorescence measurement gives a conversion between detector signals and the physical world.  By using a fluorescence x-ray source the photons impinging on the detector have a (known) fixed energy.  Using the finger plot, the difference between the peak positions of the 1-photon peak and noise peak give the conversion between ADU and keV.  The peak positions are found by fitting the peaks to a Gaussian.

The peak-finding to get the starting parameters as well as the peak fitting are done using SciPy functions. The fitting function returns several quantities: photon_spacing, spacing_error, peak_stddev, peak_error, and quality.  These quantities are defined here:

- photon_spacing: The distance between the 1-photon peak and the 0-photon (noise) peak in ADU
- spacing_error: The error on the photon_spacing
- peak_stddev: The standard deviation of the fitted peak
- peak_error: The error on the fitted peak position
- quality: This is a somewhat confusing quantity, which is an attempt to judge the quality of the fit.  The quality is defined as the difference between the height of the fitted 0-photon peak and the valley between the 0-photon peak and the 1-photon peak.



Dynamic Range Scan 
------------------
The ASIC includes an internal calibration source, which can be used in two different ways to scan over the entire dynamic range of the chip. These different methods are referred to as the pulse capacitor scan and the current source scan, which are discussed in more detail in the following sections.

Pulse Capacitor
~~~~~~~~~~~~~~~
The pulse capacitor scan works by changing the charge on the capacitor and injecting it into the pixel. This method is in some ways preferable for scanning the dynamic range, because the charge can be injected in the same timing scheme as real data taking. Unfortunately,the pulse capacitor can not cover the full dynamic range of the ASIC, but only reaches part way into the medium gain range.

Current Source
~~~~~~~~~~~~~~
Due to limitations in the capacity of the current injection source, not all pixels can be injected at the same time. In addition, cross talk between pixels means that it would be best not to inject all pixels simultaneously. By comparing the crosstalk between different injection schemes, it was decided to inject every 4th row of pixels. A full current source scan dataset consists of 4 complete scans which need to be combined in order to have every pixel injected.

The current source scan works by setting a constant current and increasing the integration time in order to scan the injected signal through the full dynamic range of the ASIC.

