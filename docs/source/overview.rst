Overview
========

This package is to analyse the calibration data from the AGIPD detector.
The package can handle both data from the European XFEL and the FS-DS CFEL labs.

Currently, the framework is able to analyse data from three different measurements: darks, x-ray fluorescence for absolute gain calibration, and the pulse capacitor dynamic range scan.  A fourth, the current source dynamic range scan, is currently being reimplemented.  It is possible to add more measurements in the future.  The individual measurements and how they are analyzed is described in :ref:`measurements`.



The analysis procedure is separated into several steps (called run_type), in
order to avoid redoing the whole procedure of reading and gathering the data
when something in the analysis changes.  Each step can be run on its own by
specifying the option "--run_type <run_type>", or all steps can be run in order
with "--run_type all". Some steps are facility- or measurement-specific.  The full analysis
chain involves the following steps, which are explained in greater detail in :doc:`steps`.

   1. preprocess (XFEL only)
   2. gather
   3. process
   4. merge (CFEL drscs only)
   5. join

