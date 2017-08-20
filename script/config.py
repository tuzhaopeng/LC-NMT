import numpy
import os
import sys

VOCAB_SIZE = 30000
# cover 97%-96% vob
#zh 233648:48671
#en 126380:13362
SRC = "zh"
TGT = "en"
DATA_DIR = "/extradisk/vwang/context-dialogue/data/iwslt_hnmt_src_static_gate_src_init_enc/"

print "config..."

import sys
sys.path.append('/home/vwang/Research/tool/nematus-master')
from nematus.hnmt_src_static_gate_src_init_enc import train

if __name__ == '__main__':
    validerr = train(saveto='/extradisk/vwang/context-dialogue/models/iwslt_hnmt_src_static_gate_src_init_enc/model.npz',
                    reload_=True,
                    dim_word=600, #vector size
                    dim=1000, # hidden size
                    n_words=VOCAB_SIZE,
                    n_words_src=VOCAB_SIZE,
                    decay_c=0.,
                    clip_c=1.,
                    lrate=0.0001,
                    optimizer='adadelta',
                    maxlen=80, # max len
		    patience=10000000, # early-stop #20
                    batch_size=80, # batch size
                    valid_batch_size=80, # batch size
                    datasets=[DATA_DIR + 'train.hist-5.' + SRC, DATA_DIR + 'train.' + TGT],
                    valid_datasets=[DATA_DIR + 'dev.hist-5.' + SRC, DATA_DIR + 'dev.' + TGT],
                    dictionaries=[DATA_DIR + 'train.' + SRC + '.json',DATA_DIR + 'train.' + TGT + '.json'],
                    validFreq=10000, # do a valid after how many updates
                    dispFreq=10000, # print
                    saveFreq=10000, # save after how many updates
                    sampleFreq=10000, # view
                    use_dropout=True,
                    dropout_embedding=0.2, # dropout for input embeddings (0: no dropout)
                    dropout_hidden=0.2, # dropout for hidden layers (0: no dropout)
                    dropout_source=0.1, # dropout source words (0: no dropout)
                    dropout_target=0.1, # dropout target words (0: no dropout)
                    overwrite=False,
                    hist_len=5,
                    external_validation_script='./validate.sh')
    print validerr

# Total_Size / Batch_Size = how many updates in 1 epoch. 
# Epoch: go through all training data is 1 epoch.
# count vob size and max len
# validFreq * batch_size = how many examples do I see before a validation.
