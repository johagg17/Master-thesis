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
                
    def get(self):
        return self.word2idx
    
                
                                                
class EHRTokenizer(object):
    def __init__(self, special_tokens = ("[PAD]", "[CLS]", "[MASK]", "[SEP]")):
        
        self.vocab = Voc()
        
        self.vocab.add_sentence(special_tokens)
        
        self.code_voc = self.add_vocab(r'processing\code_voc.npy'.replace('\\', '/'))
        
        self.code_voc.add_sentence(special_tokens)
        self.age_voc = self.add_vocab(r'processing\age_voc.npy'.replace('\\', '/'))
        self.age_voc.add_sentence(special_tokens)
        
        
    def add_vocab(self, vocab_file):
        voc = self.vocab
        array = list(map(str,np.load(vocab_file)))
        
        voc.add_sentence(array)
        
        specific_voc = Voc()
        specific_voc.add_sentence(array)
        
        return specific_voc
      
    
    def convert_tokens_to_ids(self, tokens, voc):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        vocab = self.age_voc if voc=='age' else self.code_voc
        for token in tokens:
            if str(token) == '-1':
                ids.append(-1)
            else:    
                ids.append(vocab.word2idx[token])
        return ids

    
    def convert_ids_to_tokens(self, ids, voc):
        
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        vocab = self.age_voc if voc=='age' else self.code_voc
        for i in ids:
            if str(i) == '-1':
                tokens.append('-1')
            else:
                tokens.append(vocab.idx2word[i])
        return tokens
    
    def getVoc(self, voc_str):
        
        if voc_str == 'age':
            return self.age_voc.get()
        elif voc_str == 'code':
            return self.code_voc.get()
        else:
            return self.vocab.get()
            

