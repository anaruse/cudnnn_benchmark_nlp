

def make_script(args):

    save_name = args.save_name
    args_dict = dict(vars(args))

    for algo in ['standard', 'static', 'dynamic']:
        args_dict['rnn_algo'] = algo
        # print 'save_name:', save_name
        log_file = save_name.format(**args_dict)
        # print 'log_file:', log_file

        str_args = ' '.join(['--{}={}'.format(key, value)
                             for key, value in args_dict.items() if key not in ['save_name', 'version']])

        cmd = '''python run.py {} > {}'''.format(str_args, log_file)

        print 'cmd:', cmd


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=0, type=int,
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
                        type=float, default=0.10, help='dropout')
    parser.add_argument('--seq_length', type=int,
                        dest='seq_length', default=5, help='seq_length')
    parser.add_argument('--random_length', dest='random_length',
                        type=int, default=0, help='random_length')
    parser.add_argument('--datasize', type=int,
                        dest='datasize', default=1000, help='datasize')
    parser.add_argument('--rnn_algo', default='standard',
                        type=str, help='standard, static, dynamic')
    parser.add_argument('--n_epoch', dest='n_epoch', type=int, default=30, help='n_epoch')
    parser.add_argument('--version', dest='version', type=int, default=6, help='version')
    parser.add_argument('--save_name', dest='save_name',
                        type=str, default='log_v{version}_b{batchsize}_{rnn_algo}.txt', help='save_name')

    args = parser.parse_args()
    print '###', args

    make_script(args)
