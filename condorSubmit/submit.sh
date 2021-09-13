#!/bin/bash
shopt -s expand_aliases

CURRDIR=/afs/cern.ch/work/b/bdong/DUQ/FTAGUQ/
cd $CURRDIR

python3 evaluate.py -i /eos/user/b/bdong/DUQ/UmamiTrain/DL1r-PFlow_new-taggers-stats-22M/Zprime_testing_input.h5 -o output/Zprime_selected_bjets.h5 -l 2 -w 77 --nStart 0 --nEnd 1604406
