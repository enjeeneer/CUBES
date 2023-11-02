#!/bin/bash
for c in {0..4}
do
    rm -r "case_${c}"
    mkdir "case_${c}"
    cd "case_${c}"
    python3.9 ../../../../../main_leiden.py --case $c --year 2022 --rep 0 &
    cd ..
done
wait
for c in {5..9}
do
    rm -r "case_${c}"
    mkdir "case_${c}"
    cd "case_${c}"
    python3.9 ../../../../../main_leiden.py --case $c --year 2022 --rep 0 &
    cd ..
done
wait
for c in {10..14}
do
    rm -r "case_${c}"
    mkdir "case_${c}"
    cd "case_${c}"
    python3.9 ../../../../../main_leiden.py --case $c --year 2022 --rep 0 &
    cd ..
done
wait
for c in {15..19}
do
    rm -r "case_${c}"
    mkdir "case_${c}"
    cd "case_${c}"
    python3.9 ../../../../../main_leiden.py --case $c --year 2022 --rep 0 &
    cd ..
done