# -*- encoding: utf-8 -*-
import math

import numpy as np
from HashedVocab import HashedVocab
from UnigramTable import UnigramTable

MAX_EXP = 6
EXP_TABLE_SIZE = 1000
EXP_TABLE = np.arange(start=0, stop=EXP_TABLE_SIZE, step=1, dtype=np.float64)
EXP_TABLE = np.exp((EXP_TABLE / EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
EXP_TABLE = EXP_TABLE / (EXP_TABLE + 1.)


class CBOW(object):
    REAL_TYPE = np.float64

    def __init__(self, hashed_vocab, unigram_table,
                 layer1_size, learn_hs, learn_negative):
        """
        hashed_vocab: the vocab to build on
        layer1_size: the size of layer1 of the net, effectively defines the dim of feature space
        learn_hs: use hierarchical softmax learning
        learn_negative: use negative sampling learning
        """
        self.vocab = hashed_vocab
        self.table = unigram_table
        self.layer1_size = layer1_size
        self.learn_hs = learn_hs
        self.learn_negative = learn_negative

        ## initial value for learning rate, will decrease along learning
        self.starting_alpha = 0.025
        ## a sentence is a bulk of words from train file
        self.sentence_len = 1000
        ## downsampling rate to select words into a sentence
        self.sampling = 1e-4
        ## window defines the neighborhood of a word in the sentence for hs learning
        self.window = 5
        ## negative defines the neighborhood of a word for negative learning
        self.negative = 5

        ## network weights
        ## syn0 - feature representations of words (leaf nodes) in Huffman tree
        ## shape = vocab_size x layer1_size
        self.syn0 = None
        ## syn1 - feature representations of internal nodes (non-leaf) in Huffman tree
        ## shape = vocab_size x layer1_size
        self.syn1 = None  ## hidden layer for hs learning
        ## syn1neg - feature representations of negative-sampled words
        ## shape = vocab_size x layer1_size
        self.syn1neg = None  ## hidden layer for negative learning

    def init_net(self):
        """
        """
        vocab_size = len(self.vocab)
        self.syn0 = np.random.uniform(low=-.5 / self.layer1_size,
                                      high=.5 / self.layer1_size,
                                      size=(vocab_size, self.layer1_size)).astype(CBOW.REAL_TYPE)
        if self.learn_hs:
            self.syn1 = np.zeros((vocab_size, self.layer1_size), dtype=CBOW.REAL_TYPE)
        if self.learn_negative:
            self.syn1neg = np.zeros((vocab_size, self.layer1_size), dtype=CBOW.REAL_TYPE)

    def fit(self, train_words):
        """
        train_words: list of training words
        """
        ## initialize net structure
        self.init_net()
        ntotal = len(train_words)
        ## initialize learning parameters
        alpha = self.starting_alpha
        next_random = 0

        ## read and process words sentence by sentence
        ## from the train_words
        nprocessed = 0
        while nprocessed < ntotal:
            ## adjust learning rate based on how many words have
            ## been trained on
            alpha = max(self.starting_alpha * (1 - nprocessed / (ntotal + 1.)),
                        self.starting_alpha * 0.0001)
            ## refill the sentence
            sentence = []
            while nprocessed < ntotal and len(sentence) < self.sentence_len:
                ## sampling down the infrequent words
                word = train_words[nprocessed]
                word_index = self.vocab.index_of(word)
                nprocessed += 1
                if word_index == -1: continue
                word_count = self.vocab[word_index]['count']
                if self.sampling > 0:
                    ran = ( (math.sqrt(word_count / (self.sampling * ntotal)) + 1)
                            * (self.sampling * ntotal) / word_count )
                    next_random = next_random * 25214903917 + 11
                    ## down sampling based on word frequency
                    if ran < (next_random & 0xFFFF) / 65536: continue
                sentence.append(word_index)
            ## for each word in the preloaded sentence
            ## pivot is the vocab index of current word to be trained on
            ## ipivot is the index in the current setence
            for ipivot, pivot in enumerate(sentence):
                next_random = next_random * 25214903917 + 11
                ## window-b defines the length of the window
                ## which is the neighborhood size for the current
                ## word in the sentence
                b = next_random % self.window
                ## initialize temp variable
                ## neu1: the sum of neigbhor vectors in the setence
                ## neu1e: the gradient vector wrt syn0
                ## see the explaination above the code
                neu1 = np.zeros(self.layer1_size, dtype=self.REAL_TYPE)
                neu1e = np.zeros(self.layer1_size, dtype=self.REAL_TYPE)
                ## accumulate sum of neighbor words into neu1
                ## neighbors are defined as [ipivot-(window-b), ipivot+(window-b)]
                left = max(0, ipivot - (self.window - b))
                right = min(len(sentence) - 1, ipivot + self.window - b)
                ## all neighborhood index should >= 0 (in vocab) as otherwise
                ## it won't go into the sentence in the first place
                neighborhood = [sentence[n] for n in range(left, right + 1) if sentence[n] >= 0]
                neu1 = np.sum(self.syn0[neighborhood, :], axis=0)
                ## hierarchical softmax learning
                if self.learn_hs:
                    ## for each output node in layer1
                    ## which are parent nodes of pivot in Huffman tree
                    ## notice the last element of 'path' is the word itself,
                    ## so exclude it here
                    for parent_index, parent_code in zip(self.vocab[pivot]['path'][:-1],
                                                         self.vocab[pivot]['code']):
                        ## F is the logistic transformation of dot product
                        ## between neu1 and each parent repr in syn1
                        f = np.dot(neu1, self.syn1[parent_index, :])
                        ## output out of range
                        if f <= - MAX_EXP or f >= MAX_EXP:
                            continue
                        ## logistic function transformation
                        else:
                            f = EXP_TABLE[int((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                        ## pseduo target is 1 - parent_code
                        g = (1 - parent_code - f) * alpha
                        ## accumulate neu1 to update syn0 later
                        neu1e += g * self.syn1[parent_index]
                        ## update syn1 of current parent
                        self.syn1[parent_index] += g * neu1
                ## negative sampling learning
                ## select one positive and several negative samples
                if self.learn_negative:
                    for d in range(self.negative + 1):
                        ## make sure to select the current word
                        ## as positive sample
                        if d == 0:
                            target = pivot
                            label = 1
                        ## select some 'negative' samples randomly
                        else:
                            next_random = next_random * 25214903917 + 11
                            target = self.table[(next_random >> 16) % self.table.TABLE_SIZE]
                            if (target == pivot): continue  ## ignore if it is still positive
                            label = 0
                        ## this time f is dot product of neu1 with
                        ## syn1neg
                        f = np.dot(neu1, self.syn1neg[target, :])
                        ## pseudo target is label
                        ## NOTE the differece between hs and negative-sampling
                        ## when dealing with f out of [-MAX_EXP, MAX_EXP]
                        if f > MAX_EXP:
                            g = (label - 1) * alpha
                        elif f < - MAX_EXP:
                            g = (label - 0) * alpha
                        else:
                            g = alpha * (label - EXP_TABLE[int((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))])
                        ## accumulate changes to syn0 to neu1e again
                        neu1e += g * self.syn1neg[target]
                ## update syn0 after hs and/or negative-sampling learning
                self.syn0[neighborhood, :] += neu1e
            if np.linalg.norm(neu1e, 2) < 1e-6:
                print(' '.join([str(i) for i in neu1e]))
        with open('/home/morris/github/Word2vecInJava/text_test_vec_python', 'w') as f:
            f.write(str(len(self.vocab)) + " " + str(self.layer1_size) + "\n")
            for word in self.vocab:
                f.write(word['word'].decode() + ' ' + ' '.join([str(i) for i in self.syn0[hashed_vocab.index_of(word['word']), :]]) + '\n')


if __name__ == '__main__':
    hashed_vocab = HashedVocab().fit(HashedVocab.file2ws('/home/morris/github/Word2vecInJava/text_test'))
    # hashed_vocab.inspect_vocab_tree()
    unigram_table = UnigramTable(hashed_vocab)
    train_words = list(HashedVocab.file2ws('/home/morris/github/Word2vecInJava/text_test'))
    with open('/home/morris/github/Word2vecInJava/text_test_vocab_python', 'w') as f:
        for word in hashed_vocab:
            f.write(word['word'].decode() + '\t' + str(word['count']) + '\n')

    cbow_model = CBOW(hashed_vocab, unigram_table,
                      layer1_size=100,
                      learn_hs=True, learn_negative=False)
    cbow_model.fit(train_words)