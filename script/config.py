import numpy
import os
import sys

VOCAB_SIZE = 976
#5921
SRC = "zh"
TGT = "en"
DATA_DIR = "/home/vincent/Research/workspace/pycharm/NMT/nematus-master/wmt16-scripts-master/history/data/"

import sys
sys.path.append('/home/vincent/Research/workspace/pycharm/NMT/nematus-master')
from nematus.nmt_context_endec import train


if __name__ == '__main__':
    validerr = train(saveto='/home/vincent/Research/workspace/pycharm/NMT/nematus-master/wmt16-scripts-master/history/model/model.npz',
                    reload_=True,
                    dim_word=30, #vector size
                    dim=80, # hidden size
                    n_words=VOCAB_SIZE,
                    n_words_src=VOCAB_SIZE,
                    decay_c=0.,
                    clip_c=1.,
                    lrate=0.0001,
                    optimizer='adadelta',
                    maxlen=30, # max len
		    patience=100000, # early-stop
                    batch_size=10, # batch size
                    valid_batch_size=10, # batch size
                    datasets=[DATA_DIR + '/iwslt10.hb.final.context.' + SRC, DATA_DIR + '/iwslt10.hb.final.' + TGT],
                    valid_datasets=[DATA_DIR + '/iwslt10.hb.final.context.' + SRC, DATA_DIR + '/iwslt10.hb.final.' + TGT],
                    dictionaries=[DATA_DIR + '/iwslt10.hb.final.' + SRC + '.json',DATA_DIR + '/iwslt10.hb.final.' + TGT + '.json'],
                    validFreq=90, # do a valid after how many updates
                    dispFreq=90, # print
                    saveFreq=90, # save after how many updates
                    sampleFreq=90, # view
                    use_dropout=True,
                    dropout_embedding=0.2, # dropout for input embeddings (0: no dropout)
                    dropout_hidden=0.2, # dropout for hidden layers (0: no dropout)
                    dropout_source=0.1, # dropout source words (0: no dropout)
                    dropout_target=0.1, # dropout target words (0: no dropout)
                    overwrite=False,
                    external_validation_script='./validate.sh')
    print validerr

# Total_Size / Batch_Size = how many updates in 1 epoch. 
# Epoch: go through all training data is 1 epoch.
# count vob size and max len
