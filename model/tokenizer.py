import numpy as np

'''
This file contains implementation for the custom EHRtokenizer. 

In convert_tokens_to_ids and convert_ids_to_tokens, the -1 is an additional functionality we added when doing the masked language modeling. We will make this more consistent as soon as possible. 

'''
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
    """ 
        Tokenizer used for managing the vocabularies and 
    """
    def __init__(self, task, filenames, special_tokens = ("[PAD]", "[CLS]", "[MASK]", "[SEP]")):
        
        
        self.vocab = Voc()
        self.vocab.add_sentence(special_tokens)
        
        ## Add code vocabulary, also used as tokens to input to the model.
        ## Combination of special and code tokens. 
        self.code_voc = Voc()
        file_codevoc = filenames['code']
        self.add_vocab(file_codevoc, self.code_voc)
        self.code_voc.add_sentence(special_tokens)
        
        ## Gender vocabulary
        self.gender_voc = Voc()
        self.gender_voc.add_sentence(['M', 'F', '[PAD]'])
        
        ## Age vocabulary
        self.age_voc = Voc()
        file_agevoc = filenames['age']
        self.add_vocab(file_agevoc, self.age_voc)
        self.age_voc.add_sentence(['[PAD]'])
        
        # Used for nextvisit
        if task == 'nextvisit':
            self.label_voc = Voc()
            path = '../processing/labelsvoc_ccsr.npy'
            if task == 'ndc':
                path = '../processing/labelsvoc_ndc.npy'
            self.add_vocab(path, self.label_voc)
        
    def add_vocab(self, vocab_file, vocab):
        voc = vocab
        vocarray = list(map(str,np.load(vocab_file)))
        voc.add_sentence(vocarray)
  
    def convert_tokens_to_ids(self, tokens, voc):
        """Converts a sequence of tokens into ids using the vocab."""
        
        '''
            This if-statements needs to be fixed, could be replaced with a dict instead
        '''
        ids = []
        if voc=='code':
            vocab=self.code_voc
        elif voc=='gender':
            vocab=self.gender_voc
        elif voc=='label':
            vocab=self.label_voc
        else:
            vocab=self.age_voc
            
        for token in tokens:
            if str(token) == '-1':
                ids.append(-1)
            else:    
                ids.append(vocab.word2idx[str(token)]) # needs to be fixed 
        return ids

    
    def convert_ids_to_tokens(self, ids, voc):
        
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        '''
            Thisprior_guide if-statements needs to be fixed, could be replaced with a dict instead
        '''
        if voc=='code':
            vocab=self.code_voc
        elif voc=='gender':
            vocab=self.gender_voc
        elif voc=='label':
            vocab=self.label_voc
        else:
            vocab=self.age_voc
            
        for i in ids:
            if str(i) == '-1':
                tokens.append('-1')
            else:
                tokens.append(vocab.idx2word[i])
        return tokens
    
    def getVoc(self, voc_str):
        
        # Replace if statements with a dictionary
        if voc_str == 'gender':
            return self.gender_voc.get()
        elif voc_str == 'code':
            return self.code_voc.get()
        elif voc_str == 'label':
            return self.label_voc.get()
        else:
            return self.age_voc.get()
            
            

