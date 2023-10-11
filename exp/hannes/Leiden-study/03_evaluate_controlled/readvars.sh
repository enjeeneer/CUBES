#!/bin/bash
i_case=$1
for y in {2017..2022}
do
    cd "Eplus-env-evaluate_SAC_case_${i_case}_year_${y}_rep_0-v1-res1/Eplus-env-sub_run1/output"
    /usr/local/EnergyPlus-9-5-0/runreadvars eplusout.eso
    cd ../../../
done