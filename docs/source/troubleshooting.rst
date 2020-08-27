Troubleshooting
===============

Some common issues along with their solutions or workarounds are listed here.  As more are found, this list will be updated.

SyntaxError on launch
---------------------

If you receive a SyntaxError when launching jobs with sbatch_analyse.py, you may have forgotten to set up the environment.  Here is an example:::

    [jsibille@max-display003 calibration]$ ./job_scripts/sbatch_analyse.py --input_dir /gpfs/exfel/exp/SPB/202031/p900146 --output_dir /gpfs/exfel/exp/SPB/202031/p900146/scratch/PC --type drspc --run_type all --run_list 401 402 403 404 405 406 407 408
      File "./job_scripts/sbatch_analyse.py", line 515
        self.overview.print()
                      ^
    SyntaxError: invalid syntax


Try the following and then launch the jobs again.::

    module load anaconda/3


Error: run_list must be specified
---------------------------------

When trying to run XFEL jobs, you receive an error that the run_list is not specified, although it is.  Here is an example, where the input_dir and output_dir parameters were specified in the xfel.yaml file, but not on the command line.::

    [jsibille@max-display003 calibration]$ ./job_scripts/sbatch_analyse.py --type drspc --run_type all --run_list 401 402 403 404 405 406 407 408
    usage: sbatch_analyse.py [-h] [--input_dir INPUT_DIR]
                         [--output_dir OUTPUT_DIR]
                         [--type {dark,drspc,drscs,xray}]
                         [--run_list RUN_LIST [RUN_LIST ...]]
                         [--config_file CONFIG_FILE]
                         [--run_type {preprocess,gather,process,merge,join,all}]
                         [--cfel] [--module MODULE]
                         [--temperature TEMPERATURE]
                         [--detector_string DETECTOR_STRING] [--overwrite]
                         [--no_slurm]
    sbatch_analyse.py: error: XFEL mode requires a run list to be specified.

This is a misleading error message and should be fixed.  At the moment, XFEL mode requires that input_dir, output_dir, type, and run_list all be specified on the command line, and will not be taken from the yaml file.  If one of these is missing, it throws the error with the message that the run_list is missing.
