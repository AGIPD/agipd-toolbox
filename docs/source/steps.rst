Processing Steps
================

preprocess
----------

The preprocess step is necessary for XFEL data, and is done on a per run basis.
This step finds many parameters of the run such as the number memory cells used,
number of sequences (files) in the run, and some sanity checks such as looking
for outlier trainIDs.

The code for the preprocessing step is found under:
facility_specifics/xfel/preprocess


gather
------

The gather step combines all sequences (XFEL) or parts (CFEL) together and
breaks the data set into a separate file for each ASIC in the module.  It also
does some rearrangement of the dimensions of the datasets to deal with the
default different formats between XFEL and CFEL data.

The code for the gather step is found under: src/gather
There is a GatherBase class which handles the gathering of typical data, and
the GatherDrscs and GatherDrspc classes which inherit from GatherBase and handle
the special cases of the current source and pulse capacitor dynamic range scans.


process
-------

The process step does the analysis and data processing, such as statistics and
fitting.  This step is done on the ASIC level.

The code for the process step is found under: src/process
The ProcessBase class forms the basis, and each analysis is put into a subclass
which inherits from ProcessBase.  Currently, there are subclasses for darks,
xray fluorescence, and pulse capacitor.


merge
-----

The merge step is only needed for the current source dynamic range scan.
Because the current source needs to be operated at different currents for
different ASICs / parts of ASICs, the merge step goes through all the available
currents and takes the best one for each pixel (memcell?).

The code for the merge step is found here: src/merge_drscs.py


join
----

The join step combines the ASIC-level output files back into full modules.

The code for the join step is found here: src/join_constants.py
