#!/bin/zsh

##### modify this only
git_dir=/home/kuhnm/calibration

#raw_dir=/gpfs/exfel/exp/SPB/201730/p900009
raw_dir=/gpfs/exfel/exp/SPB/201730/p900009/scratch/p002017_raw
output_dir=/gpfs/exfel/exp/SPB/201730/p900009/scratch/user/kuhnm/tmp
#channel="0"
#channel="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
channel="0 1 2"

meas_type=dark
# run numbers contained in run list have to have 4 digits
run_list="0004 0005 0006"
#run_list="0428 0429 0430"

#meas_type=pcdrs
#run_list="0488 0489 0490 0491 0492 0493 0494 0495"

#####

source /etc/profile.d/modules.sh; module load anaconda/3; module load texlive/2017; source /gpfs/cfel/cxi/common/cfelsoft-rh7/setup.sh; module load cfel-python3/latest

for ch in ${channel}
do
    printf "\nWorking on channel $ch\n"
    if [ "$meas_type" == "pcdrs" ]
    then
        ${git_dir}/job_scripts/start_analyse.py --input_dir $raw_dir \
                                                --output_dir $output_dir \
                                                --channel $ch \
                                                --run_type gather \
                                                --type $meas_type \
                                                --run_list ${run_list} \
                                                --use_xfel_in_format

    elif  [ "$meas_type" == "dark" ]
    then
        for run in $run_list
        do
            printf "run $run\n"
            ${git_dir}/job_scripts/start_analyse.py --input_dir $raw_dir \
                                                    --output_dir $output_dir \
                                                    --channel $ch \
                                                    --run_type gather \
                                                    --type $meas_type \
                                                    --run_list $run \
                                                    --use_xfel_in_format
        done

    fi

    # generates constants in both formats (xfel and agipd)
    ${git_dir}/job_scripts/start_analyse.py --input_dir $output_dir \
                                            --output_dir $output_dir \
                                            --channel $ch \
                                            --run_type process \
                                            --type $meas_type \
                                            --run_list ${run_list} \
                                            --use_xfel_in_format
done

# joins constants
${git_dir}/job_scripts/start_analyse.py --input_dir $output_dir \
                                        --output_dir $output_dir \
                                        --channel $channel \
                                        --run_type join \
                                        --type $meas_type \
                                        --run_list ${run_list} \
