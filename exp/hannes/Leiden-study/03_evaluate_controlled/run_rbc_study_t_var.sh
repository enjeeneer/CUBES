#!/bin/bash
for tset in 15 20 25
do
    for rbc in 0 1 2
    do

        python3.9 evaluate_rbc.py 0 2022 0 ${rbc} $tset 12
        for c in 5 10
        do
            let tsb=tset-3
            python3.9 evaluate_rbc.py $c 2022 0 ${rbc} $tset $tsb
        done
    done
done
