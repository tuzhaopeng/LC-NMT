import numpy

import gzip

import shuffle
from util import load_dict

# open file
def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, target,
                 source_dicts, target_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1,
                 shuffle_each_epoch=False,
                 sort_by_length=True,
                 maxibatch_size=20):
        # source, target: file path+name
        # allow source have many dicts
        if shuffle_each_epoch:
            shuffle.main([source, target])
            self.source = fopen(source+'.shuf', 'r')
            self.target = fopen(target+'.shuf', 'r')
        else:
            self.source = fopen(source, 'r')
            self.target = fopen(target, 'r')
        self.source_dicts = []
        for source_dict in source_dicts:
            self.source_dicts.append(load_dict(source_dict))
        self.target_dict = load_dict(target_dict)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        if self.n_words_source > 0:
            for d in self.source_dicts:
                for key, idx in d.items():
                    if idx >= self.n_words_source:
                        del d[key]

        if self.n_words_target > 0:
                for key, idx in self.target_dict.items():
                    if idx >= self.n_words_target:
                        del self.target_dict[key]

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.target_buffer = []
        self.k = batch_size * maxibatch_size
        

        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        if self.shuffle:
            shuffle.main([self.source.name.replace('.shuf',''), self.target.name.replace('.shuf','')])
            self.source = fopen(self.source.name, 'r')
            self.target = fopen(self.target.name, 'r')
        else:
            self.source.seek(0)
            self.target.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        source_p = []
        target = []

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            for k_ in xrange(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                tt = self.target.readline()
                if tt == "":
                    break

                self.source_buffer.append(ss.strip().split())
                self.target_buffer.append(tt.strip().split())

            # sort by target buffer
            if self.sort_by_length:
                tlen = numpy.array([len(t) for t in self.target_buffer])
                tidx = tlen.argsort()

                _sbuf = [self.source_buffer[i] for i in tidx]
                _tbuf = [self.target_buffer[i] for i in tidx]

                self.source_buffer = _sbuf
                self.target_buffer = _tbuf

            else:
                self.source_buffer.reverse()
                self.target_buffer.reverse()

        if len(self.source_buffer) == 0 or len(self.target_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:

            # actual work here. The corpus input and convert to one-hot data.
            while True:
                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break

                tmp = []
                ss_string = ' '.join(ss).strip(' ')
                ss_string_list = ss_string.split('@@@@')
                ss = ss_string_list[1].split(' ')
                for w in ss:
                    w = [self.source_dicts[i][f] if f in self.source_dicts[i] else 1 for (i,f) in enumerate(w.split('|'))]
                    tmp.append(w)
                ss = tmp

                sp_hist = [] # they share the same vector, clear it!!
                hist_list = ss_string_list[0].split('####')
                # print ' '.join(hist_list)
                # print '-----------',len(hist_list)
                for v in hist_list:
                    # print v
                    sp = v.split(' ')
                    tmp = []
                    for w in sp:
                        w = [self.source_dicts[i][f] if f in self.source_dicts[i] else 1 for (i,f) in enumerate(w.split('|'))]
                        tmp.append(w)
                    sp = tmp
                    if len(sp) < self.maxlen: sp_hist.append(sp)

                # read from target file and map to word index
                tt = self.target_buffer.pop()
                # print ' '.join(tt)
                tt = [self.target_dict[w] if w in self.target_dict else 1
                      for w in tt]
                # print tt
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target else 1 for w in tt]
                # print tt

                if len(ss) >= self.maxlen or len(tt) >= self.maxlen or len(sp_hist) < 1:
                    continue

                source.append(ss)
                source_p.append(sp_hist)
                target.append(tt)

                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size or \
                            len(source_p) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(source) <= 0 or len(target) <= 0 or len(source_p) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        # print source
        # print source_p
        # print target
        # print '----------'

        return source, source_p, target

if __name__ == '__main__':
    DATA_DIR = "/home/vincent/Research/workspace/pycharm/NMT/nematus-master/wmt16-scripts-master/history/data/"
    SRC = "zh"
    TGT = "en"
    VOCAB_SIZE = 976
    batch_size=10
    maxlen=30

    datasets=[DATA_DIR + '/nist05.hist-5.' + SRC, DATA_DIR + '/nist05.' + TGT]
    dictionaries=[DATA_DIR + '/nist05.' + SRC + '.json',DATA_DIR + '/nist05.' + TGT + '.json']
    n_words=VOCAB_SIZE
    n_words_src=VOCAB_SIZE

    train = TextIterator(datasets[0], datasets[1],
                             dictionaries[:-1], dictionaries[-1],
                             n_words_source=n_words_src, n_words_target=n_words,
                             batch_size=batch_size,
                             maxlen=maxlen,
                             maxibatch_size=10)

    for x,xp,y in train:
        # print xp
        print '#'