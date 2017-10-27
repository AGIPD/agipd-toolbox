#!/bin/zsh

##### modify this only
git_dir=/home/kuhnm/calibration

raw_dir=/gpfs/exfel/exp/SPB/201730/p900009
output_dir=/gpfs/exfel/exp/SPB/201730/p900009/scratch/user/kuhnm
channel="0"
#channel="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"

meas_type=dark
run_list="428 429 430"

#meas_type=pcdrs
#run_list="488 489 490 491 492 493 494 495"

date_id=2017-10-27 # e.g. from here: dark_AGIPD{:02d}_agipd_2017-10-27.h5

#####


source /etc/profile.d/modules.sh; module load anaconda/3; module load texlive/2017; source /gpfs/cfel/cxi/common/cfelsoft-rh7/setup.sh; module load cfel-anaconda/py3-4.3.0

for ch in $channel
do
    if [ "$meas_type" == "pcdrs" ]
    then
        echo $ch
        python ${git_dir}/job_scripts/start_analyse.py --input_dir $raw_dir --output_dir $output_dir --channel $ch --run_type gather --type $meas_type --run_list ${run_list} --use_xfel_in_format

    elif  [ "$meas_type" == "dark" ]
    then
        for run in $run_list
        do
            echo $ch $run
            python ${git_dir}/job_scripts/start_analyse.py --input_dir $raw_dir --output_dir $output_dir --channel $ch --run_type gather --type $meas_type --run_list $run --use_xfel_in_format
        done

    fi

     echo $ch
     # generates constants in the agipd format
     python ${git_dir}/job_scripts/start_analyse.py --input_dir $output_dir --output_dir $output_dir --channel $ch --run_type process --type $meas_type --run_list ${run_list} --use_xfel_in_format

     # generates constants in the xfel format
    python ${git_dir}/job_scripts/start_analyse.py --input_dir $output_dir --output_dir $output_dir --channel $ch --run_type process --type $meas_type --run_list ${run_list} --use_xfel_in_format --use_xfel_out_format
done

# joins constants in the agipd format
python ${git_dir}/src/join_constants.py --base_dir $output_dir/$meas_type --input_file ${meas_type}_AGIPD{:02d}_agipd_${date_id}.h5 --output_file ${meas_type}_joint_constants_agipd.h5

# joins constants in the xfel format
python ${git_dir}/src/join_constants.py --base_dir $output_dir/$meas_type --input_file ${meas_type}_AGIPD{:02d}_xfel_${date_id}.h5 --output_file ${meas_type}_joint_constants_xfel.h5
