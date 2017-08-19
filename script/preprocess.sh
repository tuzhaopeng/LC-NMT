#!/bin/bash

# this sample script preprocesses a sample corpus, including tokenization,
# truecasing, and subword segmentation. 
# for application to a different language pair,
# change source and target prefix, optionally the number of BPE operations,
# and the file names (currently, data/corpus and data/newsdev2016 are being processed)

# in the tokenization step, you will want to remove Romanian-specific normalization / diacritic removal,
# and you may want to add your own.
# also, you may want to learn BPE segmentations separately for each language,
# especially if they differ in their alphabet

SCRIPT_ROOT=/home/vincent/Research/workspace/pycharm/NMT/nematus-master/data
cd ${SCRIPT_ROOT}

P=/home/vincent/Research/workspace/pycharm/NMT/nematus-master/wmt16-scripts-master/history/data/nist05



# source language (example: fr)
S=en
# target language (example: en)
T=zh

# path to nematus/data
#P1=/home/vwang/Research/workspace/dialogue/data

# path to subword NMT scripts (can be downloaded from https://github.com/rsennrich/subword-nmt)
#P2=/home/vincent/Research/Tool/nematus/subword-nmt

# tokenize
#perl $P1/tokenizer.perl -threads 5 -l $S < ${P}.${S} > ${P}.${S}.tok
#perl $P1/tokenizer.perl -threads 5 -l $T < ${P}.${T} > ${P}.${T}.tok

# learn BPE on joint vocabulary:
#cat ${P}.${S}.tok ${P}.${T}.tok | python $P2/learn_bpe.py -s 20000 > ${S}${T}.bpe

#python $P2/apply_bpe.py -c ${S}${T}.bpe < ${P}.${S}.tok > ${P}.${S}.tok.bpe
#python $P2/apply_bpe.py -c ${S}${T}.bpe < ${P}.${T}.tok > ${P}.${T}.tok.bpe

# build dictionary
python $SCRIPT_ROOT/build_dictionary.py ${P}.${S}
python $SCRIPT_ROOT/build_dictionary.py ${P}.${T}

