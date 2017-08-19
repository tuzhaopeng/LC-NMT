#!/bin/sh

uidx=${1}

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=/home/vincent/Research/workspace/pycharm/NMT/nematus-master

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=/home/vincent/Research/Tool/mosesdecoder-RELEASE-2.1

# theano device
device=cpu

#model prefix
prefix=/home/vincent/Research/workspace/pycharm/NMT/nematus-master/wmt16-scripts-master/dialogue/model/model.npz

dev=/home/vincent/Research/workspace/pycharm/NMT/nematus-master/wmt16-scripts-master/dialogue/data/iwslt10.hb.final.context.zh
ref=/home/vincent/Research/workspace/pycharm/NMT/nematus-master/wmt16-scripts-master/dialogue/data/iwslt10.hb.final.en

# decode
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python $nematus/nematus/translate_context.py \
     -m $prefix.${uidx}.dev.npz \
     -i $dev \
     -o $dev.${uidx}.output.dev \
     -k 12 -n -p 1


#./postprocess-dev.sh < $dev.output.dev > $dev.output.postprocessed.dev


## get BLEU
BEST=`cat ${prefix}_best_bleu || echo 0`
$mosesdecoder/scripts/generic/multi-bleu.perl $ref < $dev.${uidx}.output.dev >> ${prefix}_bleu_scores
BLEU=`$mosesdecoder/scripts/generic/multi-bleu.perl $ref < $dev.${uidx}.output.dev | cut -f 3 -d ' ' | cut -f 1 -d ','`
BETTER=`echo "$BLEU > $BEST" | bc`

echo "BLEU = $BLEU"

# save model with highest BLEU
if [ "$BETTER" = "1" ]; then
  echo "new best; saving"
  echo $BLEU > ${prefix}_best_bleu
  cp ${prefix}.${uidx}.dev.npz ${prefix}.npz.best_bleu
  cp ${prefix}.${uidx}.dev.npz.json ${prefix}.npz.best_bleu.json
fi

rm $dev.${uidx}.output.dev
rm ${prefix}.${uidx}.dev.npz
rm ${prefix}.${uidx}.dev.npz.json
