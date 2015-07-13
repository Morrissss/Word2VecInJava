import math
import numpy as np

class UnigramTable(object):
    TABLE_SIZE = int(1e8)
    POWER = 0.75
    def __init__(self, hashed_vocab):
        self.table = np.empty(UnigramTable.TABLE_SIZE, np.int64)
        vocab = hashed_vocab
        vocab_size = len(hashed_vocab)
        ## normalization factor of all word's frequency's power
        train_words_pow = sum(math.pow(vw['count'], UnigramTable.POWER)
                              for vw in vocab)
        ## doing the sampling in the table
        ## the sampling probability of a unigram is proportional to its
        ## frequency to a power of POWER (=0.75)
        
        ## i marks the index of current word in vocab
        ## d1 marks the accumulative power-law probability up to the current word
        ## a / TABLE_SIZE marks the sampling proability up to the current word
        i = 0
        d1 = math.pow(vocab[i]['count'], UnigramTable.POWER) / train_words_pow
        for a in range(self.TABLE_SIZE):
            self.table[a] = i
            ## compare accumulative sampling prob with power-law accumulative prob
            ## move to the sampling of next word if they start not matching
            if a * 1. / UnigramTable.TABLE_SIZE > d1:
                i += 1
                d1 += math.pow(vocab[i]['count'], UnigramTable.POWER) / train_words_pow
            ## put the rest as sampling of the last word (round-off)
            if i >= vocab_size:
                i = vocab_size - 1
    def __getitem__(self, index):
        return self.table[index]