#!/bin/sh

# theano device
device=cpu

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=/home/vincent/Research/Tool/mosesdecoder-RELEASE-2.1

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=/home/vincent/Research/workspace/pycharm/NMT/nematus-master

dev=/home/vincent/Research/workspace/pycharm/NMT/nematus-master/wmt16-scripts-master/history/data/nist05.hist-3.zh
ref=/home/vincent/Research/workspace/pycharm/NMT/nematus-master/wmt16-scripts-master/history/data/nist05.en
#out=/home/vwang/Research/workspace/dialogue/data/iwslt10.nohb.final.zh.output

THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python $nematus/nematus/translate_src_static_gate_src_init_two.py \
     -m /home/vincent/Research/workspace/pycharm/NMT/nematus-master/wmt16-scripts-master/history/model_src_static_gate_src_init_two/model.npz.450.dev.npz \
     -i $dev \
     -o $dev.output \
     -k 12 -n -p 1

#THEANO_FLAGS=mode=FAST_RUN,floatX=float32,optimizer=None,device=$device,exception_verbosity=high python $nematus/nematus/translate_tgt_static.py \
#     -m /home/vincent/Research/workspace/pycharm/NMT/nematus-master/wmt16-scripts-master/history/model_tgt_static/model.npz.360.dev.npz \
#     -i $dev \
#     -o $dev.output \
#     -k 12 -n -p 1

## get BLEU
BLEU=`$mosesdecoder/scripts/generic/multi-bleu.perl $ref < $dev.output | cut -f 3 -d ' ' | cut -f 1 -d ','`
echo "BLEU = $BLEU"
