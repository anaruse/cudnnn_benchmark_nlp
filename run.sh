#!/bin/bash

N_EPOCH=3

# DATASIZE=1000
# DATASIZE=100

# VOCAB=1000
# N_UNITS=256
# N_LAYER=1
# SEQ_LENGTH=25
# BS=10

VOCAB=4096
N_UNITS=512
N_LAYER=3
SEQ_LENGTH=32
# BS=1

DATE=`date +%y%m%d-%H%M%S`

# ALGO=standard
# ALGO=static
# ALGO=dynamic

# DATASIZE=$((BS * 4))
DATASIZE=32

for ALGO in standard static dynamic
do
#for BS in 1 2 4 8 16 32
for BS in 4 8 16 32
do
    CONFIG=v${VOCAB}_u${N_UNITS}_l${N_LAYER}_s${SEQ_LENGTH}_b${BS}

    nvprof -o log.nvprof.${ALGO}.${CONFIG} \
	python run.py \
	--datasize=$DATASIZE \
	--n_vocab=$VOCAB \
	--n_units=$N_UNITS \
	--batchsize=$BS \
	--seq_length=$SEQ_LENGTH \
	--random_length=0 \
	--rnn_algo=$ALGO \
	--n_layer=$N_LAYER \
	--n_epoch=$N_EPOCH \
	--gpu=0 \
	--dropout=0.1 | tee logs/log_${ALGO}.${CONFIG}.$DATE.txt

done
done