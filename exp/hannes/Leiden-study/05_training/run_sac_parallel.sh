#!/bin/bash
for c in {0..6}
do
    rm -r "case_${c}"
    mkdir "case_${c}"
    cd "case_${c}"
    python3.9 ../../../../../main_leiden.py --case $c --year 2022 --rep 0 &
    cd ..
done
wait
for c in {7..13}
do
    rm -r "case_${c}"
    mkdir "case_${c}"
    cd "case_${c}"
    python3.9 ../../../../../main_leiden.py --case $c --year 2022 --rep 0 &
    cd ..
done
wait
for c in {14..19}
do
    rm -r "case_${c}"
    mkdir "case_${c}"
    cd "case_${c}"
    python3.9 ../../../../../main_leiden.py --case $c --year 2022 --rep 0 &
    cd ..
done
