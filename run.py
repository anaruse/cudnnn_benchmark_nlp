
'''
https://github.com/chainer/chainer/tree/master/examples/seq2seq


https://github.com/aonotas/test-chainer-performance

# TODO: add `algo` args in NStepLSTM.


'''
import six
import time
import numpy as np
import random
np.random.seed(1234)
random.seed(1234)

import chainer
from chainer import cuda
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer import serializers
from chainer import Variable
import timer

from seq2seq import Seq2seq


def make_random_dataset(xp=None, datasize=10000, seq_length=20, n_vocab=1000,
                        random_length=False, batchsize=32):
    if random_length:
        dataset = [np.random.randint(0, n_vocab, seq_length)
                   for _ in xrange(datasize)]
    else:
        dataset = np.random.randint(0, n_vocab, (datasize, seq_length))
        dataset = dataset.tolist()
    dataset = [xp.array(d, dtype=xp.int32) for d in dataset]
    dataset = make_minibatch(dataset, batchsize)
    return dataset


def make_minibatch(dataset, batchsize):
    n_dataset = len(dataset)
    dataset_batch = []
    for i in six.moves.range(0, n_dataset, batchsize):
        input_data = dataset[i:i + batchsize]
        dataset_batch.append(input_data)
    return dataset_batch


def test_performance(args):
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
    xp = cuda.cupy if args.gpu >= 0 else np

    # make data
    def make_seq2seq_dataset():
        dataset_s = make_random_dataset(xp, args.datasize, args.seq_length,
                                        args.n_vocab, args.random_length,
                                        args.batchsize)
        dataset_t = make_random_dataset(xp, args.datasize, args.seq_length,
                                        args.n_vocab, args.random_length,
                                        args.batchsize)

        dataset = [(s, t) for s, t in six.moves.zip(dataset_s, dataset_t)]
        return dataset

    dataset = make_seq2seq_dataset()
    dataset_test = make_seq2seq_dataset()

    model = Seq2seq(args.n_layer, args.n_vocab, args.n_vocab, args.n_units, args.rnn_algo)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    avg_time_forward = []
    avg_time_backward = []
    avg_time_forward_test = []

    # n epoch
    for i in xrange(args.n_epoch):
        sum_forward_time = 0.0
        sum_backward_time = 0.0
        for input_data in dataset:
            xs, ys = input_data

            # forward
            with timer.get_timer(xp) as t:
                loss = model(xs, ys)

            time_forward = t.total_time()

            # update
            model.cleargrads()

            # backward
            with timer.get_timer(xp) as t:
                loss.backward()
            time_backward = t.total_time()
            opt.update()

            sum_forward_time += time_forward
            sum_backward_time += time_backward

        # test data
        sum_forward_time_test = 0.0
        for input_data in dataset_test:
            xs, _ = input_data
            # forward
            with timer.get_timer(xp) as t:
                result = model.translate(xs, max_length=10, no_cpu=True)
            time_forward = t.total_time()
            sum_forward_time_test += time_forward

        avg_time_forward.append(sum_forward_time)
        avg_time_forward_test.append(sum_forward_time_test)
        avg_time_backward.append(sum_backward_time)
        print i, " time_forward       :", sum_forward_time
        print i, " time_forward (test):", sum_forward_time_test
        print i, " time_backward      :", sum_backward_time
        print '------------------------'

    def mean_time(time_list):
        return float(sum(time_list)) / len(time_list)

    print "avg_time_forward:", mean_time(avg_time_forward)
    print "avg_time_forward_test:", mean_time(avg_time_forward_test)
    print "avg_time_backward:", mean_time(avg_time_backward)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--batchsize', dest='batchsize', type=int,
                        default=32, help='learning minibatch size')
    parser.add_argument('--n_units', dest='n_units',
                        type=int, default=1024, help='n_units')
    parser.add_argument('--n_vocab', dest='n_vocab',
                        type=int, default=10000, help='n_vocab')
    parser.add_argument('--n_layer', dest='n_layer',
                        type=int, default=1, help='n_layer')
    parser.add_argument('--dropout', dest='dropout',
                        type=float, default=0.0, help='dropout')
    parser.add_argument('--seq_length', type=int,
                        dest='seq_length', default=5, help='seq_length')
    parser.add_argument('--random_length', dest='random_length',
                        type=int, default=0, help='random_length')
    parser.add_argument('--datasize', type=int,
                        dest='datasize', default=10000, help='datasize')
    parser.add_argument('--rnn_algo', default='standard',
                        type=str, help='standard, static, dynamic')

    parser.add_argument('--n_epoch', dest='n_epoch',
                        type=int, default=50, help='n_epoch')
    parser.add_argument('--save_model', dest='save_model',
                        type=int, default=0, help='n_epoch')

    args = parser.parse_args()
    print args
    test_performance(args)
