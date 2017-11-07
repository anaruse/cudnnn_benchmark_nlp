# cudnnn_benchmark_nlp
## Install

- Cupy : https://github.com/aonotas/cupy/tree/persistent_rnn_cudnn
- Chainer : https://github.com/aonotas/chainer/tree/cudnn-persistent-rnn


```
git clone https://github.com/aonotas/chainer/tree/cudnn-persistent-rnn
cd chainer
pip install -e .

git clone https://github.com/aonotas/cupy/tree/persistent_rnn_cudnn
cd cupy
pip install -e .

```
## Run Example

## rnn_algo=standard
```
python run.py --datasize=1000 --n_vocab=1000 --n_units=256 --batchsize=10 --seq_length=25 --rnn_algo=standard --random_length=0 --n_layer=1 --gpu=0 --n_epoch=30 --dropout=0.1 > log_v7_b10_standard_h256_l1.txt
```

## rnn_algo=static

```
python run.py --datasize=1000 --n_vocab=1000 --n_units=256 --batchsize=10 --seq_length=25 --rnn_algo=static --random_length=0 --n_layer=1 --gpu=0 --n_epoch=30 --dropout=0.1 > log_v7_b10_static_h256_l1.txt

```

## rnn_algo=dynamic

```
python run.py --datasize=1000 --n_vocab=1000 --n_units=256 --batchsize=10 --seq_length=25 --rnn_algo=dynamic --random_length=0 --n_layer=1 --gpu=0 --n_epoch=30 --dropout=0.1 > log_v7_b10_dynamic_h256_l1.txt

```

results: https://gist.github.com/aonotas/b9b9dd616066d28d08e5bcc5f0a49a77#gistcomment-2229202
