
import torch as nn
import numpy as np
import math as m


def gelu(x):
   return x * 0.5 * (1.0 + nn.erf(x / m.sqrt(2.0)))

def get_attn_pad_mask(seq_q, seq_k):
   batch_size, len_q = seq_q.size()
   batch_size, len_k = seq_k.size()
   # eq(zero) is PAD token
   pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
   return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k


class PointWiseFeedForwardNet(nn.Module):
    def __init__(self, config):
        super(PointWiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)

    def forward(self, x):
        # (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
        return self.fc2(gelu(self.fc1(x)))


class SelfAttention(nn.Module):
   def __init__(self):
       super(SelfAttention, self).__init__()

   def forward(self, Q, K, V, attn_mask):
       scores = nn.matmul(Q, K.transpose(-1, -2)) / np.sqrt(Q.size(-1)) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
       scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
       attn = nn.Softmax(dim=-1)(scores)
       context = nn.matmul(attn, V)
       return scores, context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        embed_dim = config.embed_dim
        num_heads = config.num_heads

        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        
        self.d_s = config.dimension_s
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.q = nn.Linear(embed_dim, embed_dim * num_heads)
        self.k = nn.Linear(embed_dim, embed_dim * num_heads)
        self.v = nn.Linear(embed_dim, embed_dim * num_heads)

    def forward(self, q, k, v, mask=None):

        res, batch_size = q, q.size(0)

        q_s = self.W_Q(q).view(batch_size, -1, self.num_heads, self.embed_dim).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(k).view(batch_size, -1, self.num_heads, self.embed_dim).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(v).view(batch_size, -1, self.num_heads, self.embed_dim).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]


        attn_mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]
        _, values, attention = SelfAttention(Q=q, K=k, V=v, attn_mask=mask)
        values = values.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.embed_dim) # context: [batch_size x len_q x n_heads * d_v]
        output = nn.Linear(self.num_heads * self.embed_dim, self.d_s)(values)

        return nn.LayerNorm(self.d_s)(output + res), attention


class BertEmbeddings(nn.Module):
    
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
    
        self.word_embed = nn.Embedding(config.wordvocab_size, config.hidden_size)
        self.segment = nn.Embedding(config.segvocab_size, config.hidden_size)
        self.age = nn.Embedding(config.agevocab_size, config.hidden_size)
        self.pos = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.norm = nn.LayerNorm(config.d_s)

    def forward(self, word_x, age_x, seg_x, pos_x):

        w = self.word_embed(word_x)
        a = self.age(age_x)
        s = self.segment(seg_x)
        p = self.pos(pos_x)

        embeddings = self.norm(w + a + s + p)

        return embeddings


class BertEncoder(nn.Module):
    
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.enc_self_attn = MultiHeadAttention(config=config)
        self.pos_ffn = PointWiseFeedForwardNet(config=config)

    def forward(self, enc_inputs, enc_self_attn_mask):
       enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
       enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
       return enc_outputs, attn



        


