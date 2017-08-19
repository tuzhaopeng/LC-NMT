#!/bin/sh

# theano device
device=gpu

THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python config.py #>zhen.record 2>&1 &
#THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python config.py

#THEANO_FLAGS=device=gpu0
