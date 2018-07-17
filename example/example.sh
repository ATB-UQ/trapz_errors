#!/bin/bash

echo "Calculate total error"
python ../calculate_error.py -d ../test/eg_data.dat -c -p example.png

echo
echo "Find largest error source"
python ../reduce_error.py -d ../test/eg_data.dat -t 1.0 -c
