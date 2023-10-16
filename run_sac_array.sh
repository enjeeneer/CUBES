#!/bin/bash
for c in {0..14}
do
    python3.9 main_leiden.py --case $c --year 2022 --rep 0
done
