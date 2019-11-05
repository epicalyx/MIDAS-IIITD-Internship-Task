#!/bin/bash

git clone https://github.com/epicalyx/MIDAS-IIITD-Internship-Task.git ../SPEECH EMOTION PROBLEM
conda create -n sep python=3.7 -y
source activate sep
pip install numpy, xlrd, pandas, matplotlib,librosa,jupyter,tensorflow,keras,scikit-learn,h5py,jsonpickle

echo"SETUP DONE"

echo "PLEASE MAKE SURE DATA FILES, WEIGHT FILES ARE PRESENT IN YOUR SYSTEM"
echo "TEST FUNCTION RUNNING..."

python test.py


echo "TESTING DONE"