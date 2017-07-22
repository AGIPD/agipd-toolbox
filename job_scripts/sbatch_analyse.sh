#!/usr/bin/env bash

mail_address=manuela.kuhn@desy.de

script_base_dir=/home/kuhnm/agipd-calibration
batch_job_dir=$script_base_dir/job_scripts

# options are: gather, process
#run_type=gather
run_type=process

module=M314
temperature=temperature_m15C
measurement=drscs
current=itestc150

if [ "$run_type" == "gather" ]
then
    input_dir=/gpfs/cfel/fsds/labs/agipd/calibration/raw/302-303-314-305
    time_limit="00:30:00"

else
    input_dir=/gpfs/cfel/fsds/labs/agipd/calibration/processed/
    time_limit="02:00:00"
fi

output_dir=/gpfs/cfel/fsds/labs/agipd/calibration/processed/

### Needed for gather ###
max_part=false
#column_spec="22 23 24 25"
column_spec=false

asic_set1="1"
#asic_set1="1 2 3 4 5 6 7 8"
#asic_set2="9 10 11 12 13 14 15 16"

work_dir=$output_dir/$module/$temperature/sbatch_out
if [ ! -d "$work_dir" ]; then
    printf "Creating sbatch working dir: ${work_dir}\n"
    mkdir $work_dir
fi

# getting date and time
dt=$(date '+%Y-%m-%d_%H:%M:%S')

call_sbatch()
{
    sbatch_params="--partition=all \
                   --time=${time_limit} \
                   --nodes=1 \
                   --mail-type END \
                   --mail-user ${mail_address} \
                   --workdir=${work_dir} \
                   --job-name=${run_type}_${measurement}_${module} \
                   --output=${run_type}_${measurement}_${module}_${dt}_%j.out \
                   --error=${run_type}_${measurement}_${module}_${dt}_%j.err "

    script_params="--script_base_dir ${script_base_dir} \
                   --run_type ${run_type} \
                   --measurement ${measurement} \
                   --input_dir ${input_dir} \
                   --output_dir ${output_dir} \
                   --module ${module} \
                   --temperature ${temperature} \
                   --current ${current}"

    if [ "$run_type" == "gather" ]
    then
        script_params="${script_params} \
                       --max_part ${max_part} \
                       --column_spec ${column_spec}"
    fi

    sbatch ${sbatch_params} \
           ${batch_job_dir}/analyse.sh ${script_params} $*
}

nasics=${asic_set1}
printf "Starting job for asics ${nasics}\n"
call_sbatch ${nasics}

#nasics=${asic_set2}
#printf "Starting job for asics ${nasics}\n"
#call_sbatch ${nasics}
