#!/usr/bin/env bash

mail_address=manuela.kuhn@desy.de

base_path=/home/kuhnm/agipd-calibration/agipdCalibration/batchProcessing
batch_job_dir=/home/kuhnm/agipd-calibration/batchJobCreation
work_dir=/home/kuhnm/sbatch_out

input_path=302-303-314-305
module=M314_m3
temperature=temperature_40C
current=itestc150
max_part=false
#column_spec="22 23 24 25"
column_spec=false

nasics="1 2 3 4 5 6 7 8"
#nasics="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"

sbatch_params="--partition=all \
               --job-name=gather_drscs_${module}_asic${asic} \
               --time=1:00:00 \
               --nodes=1 \
               --mail-type END \
               --mail-user ${mail_address} \
               --workdir=${work_dir} \
               --output=gather_drscs_${module}_${asic}.out \
               --error=gather_drscs_${module}_${asic}.err \
               --job-name=gather_drscs_${module}_asic \
               --output=gather_drscs_${module}_asic.out \
               --error=gather_drscs_${module}_asic.err "

script_params="--input_path ${input_path} \
               --module ${module} \
               --temperature ${temperature} \
               --current ${current} \
               --max_part ${max_part} \
               --column_spec ${column_spec} \
               ${nasics}"

printf "Starting job for asics ${nasics}\n"

    sbatch ${sbatch_params} \
           ${batch_job_dir}/batchJob_gather_asics.sh ${script_params}
