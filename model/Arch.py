import torch as nn

#import pytorch_pretrained_bert as Bert

class Embeddings(nn.Module):
    
    def __init__(self, config):
        super(Embeddings, self).__init__()
    
        self.word_embed = nn.Embedding(config.wordvocab_size, config.hidden_size)
        self.segment = nn.Embedding(config.segvocab_size, config.hidden_size)
        self.age = nn.Embedding(config.agevocab_size, config.hidden_size)
        self.pos = nn.Embedding(config.max_position_embeddings, config.hidden_size).from_pretrained(embeddings=self._init_posi_embedding(config.max_position_embeddings, config.hidden_size))
         

    def forward(self, word_x, age_x, pos_x):
        pass






