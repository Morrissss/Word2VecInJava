from collections import Counter
import mmap
import networkx as nx
import numpy as np
import re


class HashedVocab(object):
    HASH_SIZE = 30000000 ## max hash size
    ## count threshold for a word to be in the vocab - ignore infrequent word
    MIN_COUNT = 5 
    @staticmethod
    def file2ws(fpath):
        """
        file to wordstream: lazily read words from a file as an iterator
        """
        with open(fpath) as fin:
            word_pattern = re.compile(b'(.*?)\s')
            mf = mmap.mmap(fin.fileno(), 0, access = mmap.ACCESS_READ)
            for match in word_pattern.finditer(mf):
                word = match.group(1)
                if word: yield word
    def __init__(self):
        ## vocab stores vocab_word dict
        self.vocab = []
        ## vocab_hash stores the index of vocab_word in vocab
        self.vocab_hash = np.empty(HashedVocab.HASH_SIZE, dtype = np.int64)
        self.vocab_hash.fill(-1)
        ## house-keeping - total number of word occurences in training set
        ## it will be used as an estimate of training size later, which 
        ## affect the adjustment of learning rate of the deep structure
        self.n_train_words = 0
    def fit(self, word_stream):
        """
        build hashed_vocab and Huffman tree from word stream
        the word_stream is usually from reading a word file, e.g., using file2ws
        """
        ## word counter
        wc = Counter(word_stream)
        ## total occurances of training words
        self.n_train_words = sum(wc.values())
        ## Sort the words by their counts, filter out infrequent words,
        ## construct vocab_word (a dict) and put them in self.vocab
        self.vocab = list(map(lambda x: dict(zip(['word', 'count'], x)),
                         filter(lambda x: x[1] > HashedVocab.MIN_COUNT, 
                                wc.most_common(len(wc)))))
        ## if vocab is already too big for hash, either (1) ignoring more infrequent
        ## words (as implemented in C by ReduceVocab), (2) making hash size bigger
        ## here we simply raise an exception for simplicity
        if len(self.vocab) > HashedVocab.HASH_SIZE * 0.7:
            raise RuntimeError('vocab size too large for hash, increase MIN_COUNT or HASH_SIZE')
        self.build_hash()
        self.build_huffman_tree()
        return self
    def index_of(self, word):
        """
        Get the index of word in vocab by using hash,
        return -1 if it is NOT there
        """
        word_hash = self.get_word_hash(word)
        while True:
            word_index = self.vocab_hash[word_hash]
            if word_index == -1:
                return -1
            elif word == self.vocab[word_index]['word']:
                return word_index
            else:
                word_hash = (word_hash + 1) % HashedVocab.HASH_SIZE
    def __getitem__(self, word_index):
        """
        get vocab_word in vocab by its index
        """
        return self.vocab[word_index]
    def __len__(self):
        return len(self.vocab)
    def __iter__(self):
        return iter(self.vocab)
    
    
    def build_hash(self):
        self.vocab_hash = np.empty(HashedVocab.HASH_SIZE, dtype = np.int64)
        self.vocab_hash.fill(-1)
        for word_index, vocab_word in enumerate(self.vocab):
            word = vocab_word['word']
            word_hash = self.get_word_hash(word)
            self.add_to_hash(word_hash, word_index)
        return self
    def get_word_hash(self, word):
        word_hash = sum([c*(257**i) for i, c in zip(range(len(word))[::-1], word)])
        word_hash %= HashedVocab.HASH_SIZE
        return word_hash
    def add_to_hash(self, word_hash, word_index):
        while self.vocab_hash[word_hash] != -1:
            word_hash = (word_hash + 1) % HashedVocab.HASH_SIZE
        self.vocab_hash[word_hash] = word_index
        return self
    def build_huffman_tree(self):
        """Build the Huffman tree representation for word based on their freq.
        The vocab_word structure in self.vocab is a dict {word, count, path, code}
        where vocab_word['code'] is the Huffman coding of word, and
        vocab_word['path'] will be the path from root to leaf
        """
        ## for a full binary tree with n leaves, n-1 internal nodes will be needed
        ## for the 2*n-1 long data array (e.g. count and binary), the first n will be
        ## for the leaf nodes, and the last n-1 will be for the internal nodes
        vocab_size = len(self)
        ## workhorse structure for tree construction
        ## count - the count of words (leaves) and internal nodes (sum of leave counts)
        count = np.empty(vocab_size * 2 - 1, dtype = np.int64)
        count.fill(1e15)
        count[:vocab_size] = [vw['count'] for vw in self.vocab]
        ## binary - boolean repr for leaves and internal nodes
        binary = np.zeros(vocab_size*2-1, dtype=np.int64)
        ## parent_node - storing the path for each node
        parent_node = np.empty(vocab_size*2-1, dtype = np.int64)
        ## construct the tree 
        ## DESCRIPTION: iteratively group the two ungrouped nodes (leaf or internal) that 
        ## have the smallest counts
        ## Since the vocab is sorted in decreasing counts order (first half ) and 
        ## the newly created internal nodes (second half) will be the order of 
        ## increasing counts (the algorithm invariant), so we only need to check
        ## the two nodes in the middle of array to look for candidates, that is the role
        ## of min1i and min2i
        ## start searching for min1i and min2i from the middle of array
        pos1, pos2 = vocab_size - 1, vocab_size
        ## construct the vocab_size -1 internal nodes
        for a in range(vocab_size-1):
            ## min1i
            if pos1 >= 0:
                # min1i = pos1
                if count[pos1] < count[pos2]:
                    min1i, pos1 = pos1, pos1-1
                # min1i = pos2
                else: 
                    min1i, pos2 = pos2, pos2+1
            else: ## running out of leaf nodes
                min1i, pos2 = pos2, pos2+1
            ## min2i
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min2i, pos1 = pos1, pos1-1
                else:
                    min2i, pos2 = pos2, pos2+1
            else:
                min2i, pos2 = pos2, pos2+1
            ## count(parent_node) = count(child1) + count(child2)
            count[vocab_size + a] = count[min1i] + count[min2i]
            ## link parent node index
            parent_node[min1i] = vocab_size + a
            parent_node[min2i] = vocab_size + a
            ## binary encoding for min1i is 0 (left), for min2i is 1 (right)
            binary[min2i] = 1
        ## put the built Huffman tree structure in the vocab_word in vocab
        ## for each leaf node
        for a in range(vocab_size):
            ## b starting from leaf, along parent_nodes, to the root
            b = a
            code, path = [], []
            ## traverse along the path to root
            while True:
                code.append(binary[b])
                path.append(b)
                b = parent_node[b]
                ## stop when reaching root
                if b == vocab_size * 2 - 2: break
            ## path (or point) is from root to leaf, with an index offset
            ## -vocab_size
            self.vocab[a]['path'] = [vocab_size - 2] + [p - vocab_size
                                                        for p in path[::-1]]
            self.vocab[a]['code'] = code[::-1]
    def inspect_vocab_tree(self):
        """Draw the built Huffman binary tree for the vocab
        """
        g = nx.DiGraph()
        vocab_size = len(self)
        edges = set()
        for vw in self.vocab:
            tree_path = [i + vocab_size for i in vw['path']]
            tree_path = [i if i >= vocab_size
                         else "%s(%d)" % (self.vocab[i]['word'], self.vocab[i]['count'])
                         for i in tree_path]
            edges.update(zip(tree_path[:-1], tree_path[1:]))
        g.add_edges_from(edges)
        pos = nx.graphviz_layout(g, prog = 'dot')
        nx.draw(g, pos, with_labels = True, arrows = True)
        return g