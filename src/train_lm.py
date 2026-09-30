import random
import math
import os
import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


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
        x = (token_embeddings + position_embeddings)
        
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

        return { # and returning all of them
            "total" : total_parameters,
            "input_embeddings": input_embedding_parameters,
            "output_layer" : output_layer_parameters
        }
        
        
        
## --------------------> Data Preparation
def build_token_stream(tokenizer, sentences): # gets sentences and gives a large list of token ids
    token_ids = []
    
    for sentence in sentences:
        # convertig the sentence into token ids
        sentence_ids = tokenizer.encode(sentence) # get the ids for the sentence
        token_ids.extend(sentence_ids) # to add it to the sentence token
        
        token_ids.append(EOS_TOKEN_ID)# put this at the end of the sentence
        
    return np.array(token_ids, dtype=np.int32)

def laod_training_sentences(seed=0):
    # we load the training sentences from all languages
    sentences = []
    
    for lang in LANGUAGES:
        language_sentences = read_lines("train", lang)
        sentences.extend(language_sentences)
        
    # and then shuffle it to avoid overfitting
    random.Random(seed).shuffle(sentences)
    
    return sentences


## --------------------> Validation

@torch.no_grad()
def evaluate_nll(model, token_stream, sequence_length, device, batch_size=64): # we want to compute negative log likelihood
    # total negative log likelihood on a token stream, and cuz the stream is huge it is divided into non-overlapping windows
    model.eval() # so that we disable dropout and training behaviour
    
    number_of_predictions = len(token_stream) - 1 # the actual number of tokens that actually need prediction
    
    number_of_windows = math.ceil(number_of_predictions / sequence_length) # num of windows needed
    
    # and also num of padding positions needed
    padding = (number_of_windows*sequence_length - number_of_predictions)
    
    ## creating the input tokens
    input_tokens = np.concatenate(
        [token_stream[:-1], np.zeros(padding, dtype=np.int32)] # dropping the last token id then adding padding
    )
    input_tokens = input_tokens.astype(np.int64)
    
    input_tokens = torch.from_numpy(input_tokens).view(number_of_windows, sequence_length) # turn it to tensor and then applying the window on it
    
    ## creating the target tokens
    target_tokens = np.concatenate(
        [token_stream[1:], np.full(padding, -100, dtype=np.int32)] # drops the first token id and also padding with value -100 cuz pytorch will know and ignore it later
    )
    target_tokens = target_tokens.astype(np.int64)
    
    target_tokens = torch.from_numpy(target_tokens).view(number_of_windows, sequence_length)
    
    ## calculating loss
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, number_of_windows, batch_size): # goes through each window
        input_batch = input_tokens[start : start + batch_size].to(device)
        target_batch = target_tokens[start : start + batch_size].to(device)
        
        logits = model(input_batch)
            
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), target_batch.view(-1), ignore_index=-100, reduction="sum")
        
        total_loss += loss.item()
        total_tokens += (target_batch != -100).sum().item()
        
    model.train()
    
    return (total_loss / total_tokens)
            
        
def evaluate_validation_set(model, validation_streams, sequence_length, device):
    total_loss = 0.0
    
    for lang in LANGUAGES: # for each language
        language_loss = evaluate_nll(model=model, token_stream= validation_streams[lang], sequence_length=sequence_length, device=device)
        total_loss += language_loss
    
    # then we can compute the average validation loss across languages
    validation_loss = (total_loss / len(LANGUAGES))

    return validation_loss
        
   
## --------------------> Train one model
def train_model(tokenizer_name, device, epochs=4, batch_size = 64, learning_rate = 0.001):
    print()
    print("~" * 60)
    print("Training:", tokenizer_name)
    print("~" * 60)
    
    ## first we load the tokenizer
    tokenizer = TOKENIZERS[tokenizer_name]()
    vocab_size = tokenizer.vocab_size()
    print("Vocab size: ", vocab_size)
    
    ## then we train the data
    training_sentences = (laod_training_sentences()) # first getting all the sentences shuffled
    training_stream = build_token_stream(tokenizer, training_sentences) # then we get token ids as a big stream  
    training_characters = sum(len(sentence) for sentence in training_sentences)# total number of chars
    # num of chars are important and also I colculate other values as well in terms of chars cuz by basing the total amount of training on chars makes the training comparison more fair, rather than only and simply focusing on token count
    
    ## validation data
    validation_streams = {}
    for lang in LANGUAGES:
        sentences = read_lines("valid", lang)
        validation_streams[lang] = (build_token_stream(tokenizer, sentences))
    
    # I want each tokenizer to process the same amount of original text
    context_length = MODEL_SETTINGS["ctx"]
    
    tokens_per_step = (batch_size*context_length)
    characters_per_token = (training_characters / len(training_stream))
    character_per_step = (tokens_per_step*characters_per_token)
    total_characters = (epochs*training_characters)
    number_of_steps = round(total_characters/character_per_step)
    
    print("Training characters:", training_characters)
    print("Training steps:", number_of_steps)
    
    ## then we create the model
    model = LanguageModel(vocab_size=vocab_size, **MODEL_SETTINGS).to(device)
    parameters = model.get_parameter()
    
    print("Total parameters:", parameters["total"])
    print("Input embedding parameters:", parameters["input_embeddings"])
    print("Output layer parameters:", parameters["output_layer"])
    
    ## optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    ## training data tensor
    training_data = torch.tensor(training_stream, dtype=torch.long, device=device)
    max_start = (len(training_data) - context_length - 1)
    
    # storing losses for the learning curve
    training_losses = []
    validation_losses = []
    
    # evaluate 20 times during training
    evaluation_interval = max(1, number_of_steps // 20)
    
    ## training loop
    for step in range(1, number_of_steps + 1):
        model.train() # we go to train mode first
        
        # random starting positions
        starts = torch.randint(0, max_start, (batch_size,), device=device)
        
        # positions within each sequence
        positions = torch.arange(context_length, device=device)
        
        indices = (starts[:, None] + positions) # actual indices
        
        x = training_data[indices] # input sequence
        
        y = training_data[indices + 1] # target sequence, one token ahead of the input
        
        # forward pass
        logits = model(x)
        
        # training loss
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1)) # the reshape thing is to turn both of input and target to the shape it is expected by pytorch
        
        ## updating the model
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        
        ## validation
        if(step % evaluation_interval == 0 or step == number_of_steps): # every 20 steps we evaluate
            training_loss = loss.item()
            
            validation_loss = (evaluate_validation_set(model=model, validation_streams=validation_streams, sequence_length=context_length, device=device))
            
            training_losses.append(training_loss)
            validation_losses.append(validation_loss)
            
            print(
                f"Step {step}/{number_of_steps} | train loss: "
                f"{training_loss: .4f} | validation loss: {validation_loss:.4f}"
            )
            
        
    ### saving the trained the model
    os.makedirs("checkpoints", exist_ok=True)
    model_path = f"checkpoints/{tokenizer_name}.pt"
        
    torch.save(
        {
            "model_state_dict" : model.state_dict(),
            "vocab_size" : vocab_size,
            "tokenizer" : tokenizer_name, 
            "parameters" : parameters
        },
        model_path
    )
    print("Saved:", model_path)
        
    return {
        "training_losses" : training_losses,
        "validation_losses" : validation_losses,
        "parameters": parameters
    }


## --------------------> Main
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print("using device: ", device)
    
    results = {}
    
    # training one model for each tokenizer
    for tokenizer_name in TOKENIZERS:
        results[tokenizer_name] = train_model(tokenizer_name=tokenizer_name, device=device)
        
    ## learning curves
    plt.figure(figsize=(8, 5))
    
    for tokenizer_name in TOKENIZERS:
        training_losses = results[tokenizer_name]["training_losses"]
        validation_losses = results[tokenizer_name]["validation_losses"]
        
        plt.plot(training_losses, label=f"{tokenizer_name} train")
        plt.plot(validation_losses, label=f"{tokenizer_name} validation")
    
    plt.xlabel("Evaluation")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    
    plt.legend()
    plt.tight_layout()
    
    os.makedirs("./plots", exist_ok=True)
    plt.savefig("./plots/learning_curves.png")
    print("Saved: learning_curves.png")
    
    
    ## parameter counts
    print()
    print("Parameter counts")
    print("----------------------")
    
    for tokenizer_name in TOKENIZERS:
        parameters = results[tokenizer_name]["parameters"]
        print(tokenizer_name)
        print(" Total: ", parameters["total"])
        print(" Input embeddings: ", parameters["input_embeddings"])
        print(" Output layer: ", parameters["output_layer"])
        