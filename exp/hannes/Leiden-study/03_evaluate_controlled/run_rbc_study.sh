#!/bin/bash
for c in {0..14}
do
    for rbc in 0 1 2
    do
        python3.9 evaluate_rbc.py $c 2022 0 ${rbc}
    done
done