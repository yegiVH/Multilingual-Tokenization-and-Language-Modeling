import json

import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F


from data import LANGUAGES, read_lines
from character_tokenizer import CharTokenizer
from bpe_tokenizer import BPETokenizer


## --------------------> general settings

EOS_TOKEN_ID = 3 # cuz all tokenizers have to use the same ID for the end of setnece token

TOKENIZERS = { # just loading all three tokenizers in a dict to use it later
    "char" : lambda: CharTokenizer.load("char_vocab.json"),
    "bpe4000": lambda: BPETokenizer("bpe4000.model"),
    "bpe10000": lambda: BPETokenizer("bpe10000.model"),
}

# the transformer architecture and ofc this is the same for all three tokenizatio methods
MODEL_SETTINGS = {
    "n_layer" : 2, # num of transformer layers
    "d_model": 256, # hidden dimension (so every vector length is 256)
    "n_head": 4, # num of heads (attension is split into 4 heads and each head learns different patterns)
    "d_ff": 1024, # feedforward size
    "dropout": 0.1, 
    "ctx": 256, # context length max
}



## --------------------> Transformer

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.n_heads = n_heads
        
        # we first do the layer normalization before the attention
        self.layer_norm_1 = nn.LayerNorm(d_model) # it normalizes activations so values become more stable which helps the training
        
        # we create the query, key and value at the same time for attention
        # so the output size would be 3 * d_model, cuz we create Q, K and V
        self.qkv = nn.Linear(d_model, 3*d_model) # so input is 256 num and output 768
        
        # then project the attention output back to d_model
        self.attention_output = nn.Linear(d_model, d_model) # 256 to 256, just to mix info
        
        # another layer normalization before the feed-forward network
        self.layer_norm_2 = nn.LayerNorm(d_model)
        
        # feed-forward neural network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), # 256 to 1024
            nn.GELU(), # 1024
            nn.Linear(d_ff, d_model), # 1024 to 256
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, sequence_length, d_model = x.shape
        
        ## multi head self attention
        
        # we first have to normalize the input
        normalized_x = self.layer_norm_1(x)
        
        # then we create query, key and value
        q, k, v = self.qkv(normalized_x).split(d_model, dim=2)
        
        head_size = d_model // self.n_heads
        
        # transpose so that it goes into the shape pytorch expects
        q = q.view(batch_size, sequence_length, self.n_heads, head_size).transpose(1, 2)
        
        k = k.view(batch_size, sequence_length, self.n_heads, head_size).transpose(1, 2)
        
        v = v.view(batch_size, sequence_length, self.n_heads, head_size).transpose(1, 2)
        
        
        # casual self attention also which means that a token cannot look at future tokens
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0) # each token recieves info from previous tokens
        
        # put the attention heads back together
        attention = attention.transpose(1,2)
        
        attention = attention.contiguous().view(batch_size, sequence_length, d_model) # to make it into the dimension that pytorch expects
        
        # project attention output
        attention = self.attention_output(attention) # mix all the things heads found into one representation
        
        # residual connection
        x = x + self.dropout(attention) # so old info wont be disappeared
        
        ### feed forward network
        normalized_x = self.layer_norm_2(x) # we normalized again
        
        self.feed_forward_output = self.feed_forward(normalized_x) # 256 -> 1024 -> 256
        
        # second residual connection
        x = x + self.dropout(self.feed_forward_output) # so old representation + new information
        
        return x
    
    
## --------------------> Language Model
class LanguageModel(nn.Module): # recieves token IDs and predicts the next token
    def __init__(self, vocab_size, n_layer, d_model, n_head, d_ff, dropout, ctx):
        super().__init__()
        self.context_length = ctx
        
        ##  building token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        ## positional embeddings
        self.position_embedding = nn.Embedding(ctx, d_model) # giving the model info about the position of each token in the sequence
        
        self.dropout = nn.Dropout(dropout)
        
        ## transformer block
        self.blocks = nn.ModuleList() # empty list
        
        for _ in range(n_layer): # so for each layer which are 2 we make the transformer block
            block = TransformerBlock(
                d_model= d_model,
                n_heads= n_head,
                d_ff= d_ff,
                dropout= dropout
            )
            self.blocks.append(block)
            
        # final layer normalization
        self.final_layer_norm = nn.LayerNorm(d_model)
        
        
        ## Output layer
        # to covert the transformer representation into scores for every token in the vocab
        self.output_layer =  nn.Linear(d_model, vocab_size, bias=False)
        
        # then we should initialize model weights
        self.apply(self.initialize_weights) # so the function will be invokec for every linear, embedding, layernorm, .. in the model
    

    @staticmethod
    def initialize_weights(module): # initializing the model's parameters before training starts
        # to initialize linear and embedding layers
        if isinstance(module, (nn.Linear, nn.Embedding)): # if the layers is a linear or embedding layer
            nn.init.normal_( # so small random values for weights
                module.weight,
                mean=0.0,
                std=0.02
            )
        
        # and for initializing biases
        if isinstance(module, nn.Linear) and module.bias is not None: # if it is a linear layer and it has a bias vector
            nn.init.zeros_(module.bias) # starts bias values as zero
            
    def forward(self, token_ids):
        sequence_length = token_ids.size(1)
        
        # first we create position ids
        position_ids = torch.arange(sequence_length, device=token_ids.device)
        
        token_embeddings = self.token_embedding(token_ids)
        position_embeddings = self.position_embedding(position_ids)
        
        # after it we add the token and positional info cuz we wanna have both
        x = token_embeddings + position_embeddings
        
        x = self.dropout(x)
        
        # then passing through every transformer block
        for block in self.blocks:
            x = block(x)
            
        # final normalization
        x = self.final_layer_norm(x)
        
        # predicting the next token
        logits = self.output_layer(x)
        
        return logits
    
    # a function to count the parameters in different parts of the model
    def get_parameter(self):
        total_parameters = sum(parameter.numel() for parameter in self.parameters())
        input_embedding_parameters = (self.token_embedding.weight.numel())
        output_layer_parameters = (self.output_layer.weight.numel())
        positional_paramters = (self.position_embedding.weight.numel())
        
        transformer_parameters = (
            total_parameters
            - input_embedding_parameters
            - output_layer_parameters
            - positional_paramters
        )
        
        return { # and returning all of them
            "total" : total_parameters,
            "input_embeddings": input_embedding_parameters,
            "output_layer" : output_layer_parameters,
            "positional" : positional_paramters,
            "transformer_body" : transformer_parameters,
        }
        
        
        
        