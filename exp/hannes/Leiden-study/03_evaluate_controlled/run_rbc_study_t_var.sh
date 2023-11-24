#!/bin/bash
for tset in 15 20 25
do
    python3.9 evaluate_rbc.py 0 2022 0 2 $tset 12
    for c in 5 10
    do
        let tsb=tset-3
        python3.9 evaluate_rbc.py $c 2022 0 2 $tset $tsb
    done

    for r in {0..10}
    do
        python3.9 evaluate_rbc.py 0 2022 $r 0 $tset 12
        for c in 5 10
        do
            let tsb=tset-3
            python3.9 evaluate_rbc.py $c 2022 $r 0 $tset $tsb
        done
    done
done
