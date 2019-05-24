# AGIPD detector toolbox

System dependencies:
  * Python (3.6)
  * HDF5 libraries 
  * Numpy - library for scientific computing with Python
  * YAML -  human-readable data serialization language
  
## Installation

Starting from scratch:
```
% git clone https://github.com/AGIPD/agipd-toolbox.git
``` 

To update the framework:
```
% cd /path/to/agipd-toolbox
% git pull
```

## Usage

### Calibration

#### XFEL mode

Establish environment of the XFEL offline cluster:

```
source /gpfs/cfel/cxi/common/public/cfelsoft-rh7-public/conda-setup.sh; conda activate base
```

Run:
```
cd <repo_location>
./job_scripts/sbatch_analyse.py --input_dir <input_dir> --output_dir <output_dir> --type <measurement_type> --run_list <run numbers> --run_type all
```
e.g.
```
./job_scripts/sbatch_analyse.py --input_dir /gpfs/exfel/exp/SPB/201730/p900009 --output_dir /gpfs/exfel/exp/SPB/201730/p900009/scratch/user/kuhnm/tmp --type dark --run_list 819 820 821 --run_type all
```
when running without sbatch (local, sequencially) is required: add option --no_slurm

#### CFEL mode

##### With config file

###### Adjust config file

* create a new configuration file (e.g. by copying an already existing one) into <repo-path>/conf/<my_conf_name>.ini
  e.g. /gpfs/cfel/fsds/labs/agipd/calibration/shared/shared/calibration/conf/M314.ini
* open the configuration file
* minimum parameters to adjust:
  * module
  * temperature
  * current: multple currents can be gathered at once (comma separated list)
  * input_dir in the all section

e.g.
```
M314.ini
module = M314
temperature = temperature_m15C
 
[drscs]
current = itestc80, itestc150, itestc20
 
[all]
input_dir = /gpfs/cfel/fsds/labs/agipd/calibration/raw/302-303-314-305
output_dir = /gpfs/cfel/fsds/labs/agipd/calibration/processed/
```

Run
```
cd <repo_location>
./job_scripts/sbatch_analyse.py --config <config_file> --cfel --type <measurement_type>
```

e.g.
```
cd <repo_location>
./job_scripts/sbatch_analyse.py --config M314 --cfel --type dark
```
The measurement type can also be defined statically in the config file.

###### From command line

```
cd <repo_location>
./job_scripts/sbatch_analyse.py --input_dir <input_dir> --output_dir <output_dir> --cfel --module <module> --temperature <temperature> --type <measurement_type> --run_type all
```
e.g.
```
./job_scripts/sbatch_analyse.py --input_dir /gpfs/cfel/fsds/labs/agipd/calibration/raw/315-304-309-314-316-306-307 --output_dir /gpfs/exfel/exp/SPB/201730/p900009/scratch/user/kuhnm/tmp/cfel --cfel --module M304 --temperature temperature_m25C --type dark --run_type all
```
when running without sbatch (local, sequencially) is required: add option --no_slurm

Annotation: from command line is not completely consistent because only the default (static) currents and safety factors can be used.

## Changelog
