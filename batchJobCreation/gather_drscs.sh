#!/usr/bin/env bash

mail_address=manuela.kuhn@desy.de

base_path=/home/kuhnm/agipd-calibration
batch_job_dir=$base_path/batchJobCreation
work_dir=$base_path/sbatch_out

input_path=302-303-314-305
module=M314
temperature=temperature_m15C
current=itestc20
max_part=false
#column_spec="22 23 24 25"
column_spec=false

#asic_set1="16"
asic_set1="1 2 3 4 5 6 7 8"
asic_set2="9 10 11 12 13 14 15 16"

if [ ! -d "$work_dir" ]; then
  mkdir $work_dir
fi

call_sbatch()
{
    sbatch_params="--partition=all \
                   --time=00:30:00 \
                   --nodes=1 \
                   --mail-type END \
                   --mail-user ${mail_address} \
                   --workdir=${work_dir} \
                   --job-name=gather_drscs_${module}_asic \
                   --output=gather_drscs_${module}_asic-%j.out \
                   --error=gather_drscs_${module}_asic-%j.err "

    script_params="--input_path ${input_path} \
                   --module ${module} \
                   --temperature ${temperature} \
                   --current ${current} \
                   --max_part ${max_part} \
                   --column_spec ${column_spec}"

    sbatch ${sbatch_params} \
           ${batch_job_dir}/batchJob_gather_asics.sh ${script_params}\
                                                     $*
}

nasics=${asic_set1}
printf "Starting job for asics ${nasics}\n"
call_sbatch ${nasics}

nasics=${asic_set2}
printf "Starting job for asics ${nasics}\n"
call_sbatch ${nasics}
