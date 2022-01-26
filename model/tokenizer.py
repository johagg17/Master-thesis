import numpy as np


class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)
                
                
                
                                
class EHRTokenizer(object):
    def __init__(self, data_dir, special_tokens = ("[PAD]", "[CLS]", "[MASK]", "[SEP]")):
        
        self.vocab = Voc()
        
        self.vocab.add_sentence(special_tokens)
        
        self.add_vocab(r'C:\Users\Johan\Documents\Skola\MasterThesis\Master-thesis\pre-processing\code_voc.npy'.replace('\\', '/'))
        self.add_vocab(r'C:\Users\Johan\Documents\Skola\MasterThesis\Master-thesis\pre-processing\age_voc.npy'.replace('\\', '/'))
        
        
    def add_vocab(self, vocab_file):
        voc = self.vocab
        array = list(map(str,np.load(vocab_file)))
        
        voc.add_sentence(array)
      
    
    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab.word2idx[token])
        return ids

    
    def convert_ids_to_tokens(self, ids):
        
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.vocab.idx2word[i])
        return tokens


