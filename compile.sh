#!/bin/bash
mkdir -p temp
cd temp
rm *.out
rm *.o
nvcc -arch=sm_70 -dc ../gpurecon/*.cu
nvcc -arch=sm_70 *.o -o a.out
cd ..

