#!/bin/bash
for ((start=0; start<=18; start+=2));
do
    for ((c=start;c<=start+1;c++));
    do
        rm -r "case_${c}"
        mkdir "case_${c}"
        cd "case_${c}"
        python3.9 ../../../../../main_leiden.py --case $c --year 2022 --rep 0 &
        cd ..
    done
    wait
done
