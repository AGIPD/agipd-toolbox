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

For general information about what options are available and what they mean, a help function is included:
```
cd <repo_location>/calibration
./job_scripts/sbatch_analyse.py -h
```

Output of help:
```
usage: sbatch_analyse.py [-h] [--input_dir INPUT_DIR]
                         [--output_dir OUTPUT_DIR]
                         [--type {dark,drspc,drscs,xray}]
                         [--run_list RUN_LIST [RUN_LIST ...]]
                         [--config_file CONFIG_FILE]
                         [--run_type {preprocess,gather,process,merge,join,all}]
                         [--cfel] [--module MODULE]
                         [--temperature TEMPERATURE] [--overwrite]
                         [--no_slurm]

optional arguments:
  -h, --help            show this help message and exit
  --input_dir INPUT_DIR
                        Directory to get data from
  --output_dir OUTPUT_DIR
                        Base directory to write results to
  --type {dark,drspc,drscs,xray}
                        Which type to run: dark: generating the dark constants
                        drspc: generating the pulse capacitor constants xray:
                        generating the photon spacing constants
  --run_list RUN_LIST [RUN_LIST ...]
                        Run numbers to extract data from. Requirements: dark:
                        3 runs for the gain stages high, medium, low (in this
                        order) drspc: 8 runs
  --config_file CONFIG_FILE
                        Config file name to get config parameters from
  --run_type {preprocess,gather,process,merge,join,all}
                        Run type of the analysis
  --cfel                Activate cfel mode (default is xfel mode)
  --module MODULE       Module to be analysed (e.g M215). This is only used in
                        cfel mode
  --temperature TEMPERATURE
                        The temperature for which the data was taken (e.g.
                        temperature_m25C). This is only used in cfel mode
  --overwrite           Overwrite existing output file(s)
  --no_slurm            The job(s) are not submitted to slurm but run
                        interactively
```

Establish environment of the Maxwell cluster:

```
module load anaconda/3
```


#### XFEL mode

Run:
```
cd <repo_location>/calibration
./job_scripts/sbatch_analyse.py --input_dir <input_dir> --output_dir <output_dir> --type <measurement_type> --run_list <run numbers> --run_type all
```
e.g.
```
./job_scripts/sbatch_analyse.py --input_dir /gpfs/exfel/exp/SPB/201730/p900009 --output_dir /gpfs/exfel/exp/SPB/201730/p900009/scratch/user/kuhnm/tmp --type dark --run_list 819 820 821 --run_type all
```
when running without sbatch (local, sequencially) is required: add option --no_slurm

Note:  In XFEL mode it is required to set the parameters input_dir, output_dir, type, and run_list on the command line.  The software will NOT take these parameters from the .yaml file!!


#### CFEL mode

##### With config file

###### Adjust config file

* create a new configuration file (e.g. by copying an already existing one) into <repo-path>/calibration/conf/<my_conf_name>.yaml
  e.g. <repo_path>/calibration/conf/M304.yaml
* open the configuration file
* minimum parameters to adjust:
  * module
  * temperature
  * current: multple currents can be gathered at once (comma separated list)
  * input_dir in the all section

e.g.
```
M304.yaml

general:
    #mail_address: jennifer.poehlsen@desy.de

    # options are: all, gather, process, merge (for drscs)
    run_type: all

    module: M304
    temperature: temperature_m20C

    # options are: dark, drspc, xray, drscs (coming soon)
    measurement: dark


    #asic_set: null
    #asic_set: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    asic_set: [1]

process:
    time_limit: 02:00:00


drscs:
    current: [itestc150]
    #current: [itestc80, itestc150, itestc20]
    safety_factor: 950

drspc:
    run_name: [r1, r2, r3, r4, r5, r6, r7, r8]
    #subdir: burst

dark:
    #tint: tint50us
    tint: null
    run_name: [hg, mg, lg]
    # if underneath the measurement directory there is an subdirectory
    #subdir: burst

xray:
    element: Mo
    run_name: [175]
    #run_name: null
    run_list: [0]
    #subdir: burst

all:
    input_dir: /gpfs/cfel/fsds/labs/agipd/calibration/raw/333-325-331-304-320-312-302-311
    output_dir: /gpfs/cfel/fsds/labs/agipd/calibration/processed/


```

Run
```
cd <repo_location>/calibration
./job_scripts/sbatch_analyse.py --config <config_file> --cfel --type <measurement_type>
```

e.g.
```
cd <repo_location>/calibration
./job_scripts/sbatch_analyse.py --config M314 --cfel --type dark
```
The measurement type can also be defined statically in the config file.

###### From command line

```
cd <repo_location>/calibration
./job_scripts/sbatch_analyse.py --input_dir <input_dir> --output_dir <output_dir> --cfel --module <module> --temperature <temperature> --type <measurement_type> --run_type all
```
e.g.
```
./job_scripts/sbatch_analyse.py --input_dir /gpfs/cfel/fsds/labs/agipd/calibration/raw/315-304-309-314-316-306-307 --output_dir /gpfs/exfel/exp/SPB/201730/p900009/scratch/user/kuhnm/tmp/cfel --cfel --module M304 --temperature temperature_m25C --type dark --run_type all
```
To run without sbatch (local, sequencially) add option: --no_slurm

Annotation: from command line is not completely consistent because only the default (static) currents and safety factors can be used.

## Changelog

v1.0.0
- Added xray measurement
- Changed from .ini to .yaml config files
- Adjusted internal directory structure
