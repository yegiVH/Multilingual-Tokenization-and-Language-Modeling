import json # save the vocabulary to a file
from data import LANGUAGES, read_lines 

# special tokens
PAD = 0 # for making sentences the same length
UNK = 1 # for unknown character
BOS = 2 # beginning of sentence
EOS = 3 # end of sentence

class CharTokenizer:
    
    def __init__(self, lines):
        chars = set() # a set for storing unique chars
        for line in lines:
            chars.update(line) # adding all the chars from all the lines
        
        # index to string   
        self.itos = ["<pad>", "<unk>", "<bos>", "<eos>"] + sorted(chars)
        
        self.stoi = {} # for saving string to index
        for i, char in enumerate(self.itos):
            self.stoi[char] = i 
            
    def encode(self, text): # for converting text to ids
        ids = []
        for char in text: # we loop through the characters
            ids.append(self.stoi.get(char, UNK)) # we look up each character, if the char is there we return the index, if not we return UNK index (1)
        return ids
        
    def pieces(self, text): # for human-readable output
        tokens = []
        for char in text:
            if char in self.stoi:
                tokens.append(char) # each char that have index, meaning they are known, go to the token list
            else:
                tokens.append("<unk>")
        return tokens
    
    def decode(self, ids): # converting token ids back to text
        tokens = []
        
        for token_id in ids:
            tokens.append(self.itos[token_id])
        
        return "".join(tokens)
            
    def vocab_size(self): # returning the vocab size
        return len(self.itos)
        
    def save(self, path): # writing the vocabulary in a file
        with open(path, "w", encoding="utf-8") as f: # open file for writing
            json.dump(self.itos, f, ensure_ascii=False, indent=2) # ensure_ascii=False to make it more readable 
        
    @staticmethod
    def load(path): # for loading a vocab
        with open(path, encoding="utf-8") as f:
            vocab = json.load(f)
        tok = CharTokenizer([]) # create an empty tokenizer
            
        tok.itos = vocab
        tok.stoi = {} # and the reverse dict
        for i, char in enumerate(tok.itos):
            tok.stoi[char] = i
        return tok
        
if __name__ == "__main__":
    
    train = {}
    # reading the training data
    for lang in LANGUAGES: # for each lang
        train[lang] = read_lines("train", lang)
        
    all_sentences = [] # to collect all sentences from all languages
    for lang in LANGUAGES:
        all_sentences.extend(train[lang]) # all langs in one large list
        
    # train tokenizer
    tok = CharTokenizer(all_sentences)
    
    print("Character tokenizer")
    print("--------------------")
    print("vocab size:", tok.vocab_size())
    
    # to save the vocabulary
    tok.save("char_vocab.json")
    
    # count distinct characters per language
    for lang in LANGUAGES:
        all_text = "".join(train[lang]) # all sentences into one huge string
        print(lang, "distinct train chars:", len(set(all_text)))
        
    # and also unknown char rate
    for split in ["valid", "test"]: # for validation and test set
        for lang in LANGUAGES:
            ids = []
            for sentence in read_lines(split, lang):
                ids.extend(tok.encode(sentence))
            
            unk_count = ids.count(UNK)
            
            if len(ids) > 0:
                unk_rate = unk_count / len(ids)
            else:
                unk_rate = 0.0
            
            print(split, lang, f"unk rate in validation and test data = {unk_rate:.5f}")
            
    # checking that known validation sentences are represented without unk
    print("\nExample character tokenizations:")
    
    for lang in LANGUAGES:
        sentence = read_lines("valid", lang)[0]
        
        print(f"\n{lang}:")
        print("Sentence:", sentence)
        print("Tokens:", tok.pieces(sentence))