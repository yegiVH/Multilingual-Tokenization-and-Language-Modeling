import json
import sentencepiece as spm # google's tokenization library
# so sentencepiece is a tokenizer that learns subword units and convert text to token ids and also token ids to text
from data import LANGUAGES, read_lines

PAD = 0
UNK = 1
BOS = 2
EOS = 3

# for SentencePiece training for the parameter character_coverage
COVERAGES = [0.9995, 0.999, 0.998, 0.995, 0.99, 0.985, 0.98] # cuz sentencepiece needs character_coverage more than 0.98 and won't get lower
# so if the vocab is too small to hold all the chinese characters then I try lower coverage until training is successful
# higher coverage keeps more rare characters

BALANCED_CORPUS = "/srv/data/lt2326-h26/a1/tokenizer/balanced.txt"


class BPETokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor( model_file = str(model_path)) # so this loads a trained sentencepiece model
        
    def encode(self, text): # text into token ids
        return self.sp.encode(text) # returns a list of token ids
    
    def pieces(self, text):  # gets text and returns token strings
        return self.sp.encode(text, out_type=str)
    
    def decode(self, ids): # token ids back to text
        return self.sp.decode(ids) # so ids back to text
    
    def vocab_size(self):
        return self.sp.get_piece_size()
    
    def is_byte(self, token_id): # whether a token is a byte fallback token
        return self.sp.is_byte(token_id) # returns true if this id is a byte-fallback piece
    
    def piece(self, token_id): # a token id into its token string
        return self.sp.id_to_piece(token_id) # string form of a single id
    

def train_bpe(corpus_path, vocab_size, model_prefix):
    used_coverage = None
    for coverage in COVERAGES: # for all the coverages
        try:
            spm.SentencePieceTrainer.train( # we train the sentencepiece
                input = corpus_path,
                model_prefix = model_prefix, # output file names
                vocab_size = vocab_size,
                model_type = "bpe",
                character_coverage = coverage,
                byte_fallback = True, # it ensures every unicode character can be represented
                normalization_rule_name = "identity", # so do not modify the text
                add_dummy_prefix = False,
                remove_extra_whitespaces = False,
                max_sentence_length = 16384, # which allows very long lines
                pad_id = PAD,
                unk_id = UNK,
                bos_id = BOS,
                eos_id = EOS,
                pad_piece = "<pad>",
                unk_piece = "<unk>",
                bos_piece = "<bos>",
                eos_piece= "<eos>",
            )
            used_coverage = coverage
            break
        
        except RuntimeError as e: # if we faced runtime error
            print(f"vocab_size {vocab_size} and coverage={coverage} failed, error: {e}")
            
    if used_coverage is None: # and if all the coverages failed
        raise SystemExit(
            f"Could not train BPE with {vocab_size} vocab size even at coverage 0.98. Vocabulary is too small to fit all required characters plus the 256, byte-fallback tokens. Try larger vocab_size."
        )
        
    # saving what coverage was actually used, for the report
    info = {"vocab_size": vocab_size, "character_coverage": used_coverage} # so saving the metadata
    json.dump(info, open(f"{model_prefix}_info.json", "w")) # write it into a json file
    print(f"{model_prefix}: trained with coverage= {used_coverage}")
    
    

if __name__ == "__main__":
    for V in [2000, 10000]:
        train_bpe(BALANCED_CORPUS, vocab_size=V, model_prefix = f"bpe{V}")

    for V in [2000, 10000]:
        tok = BPETokenizer(f"bpe{V}.model")
        print(f"\nbpe{V} vocab size:", tok.vocab_size())
        for lang in LANGUAGES:
            sentence = read_lines("valid", lang)[0]
            ids = tok.encode(sentence)
            decoded = tok.decode(ids)
            assert decoded == sentence, f"NOT LOSSLESS for {lang}: {decoded!r} != {sentence!r}" # cuz the reconstructed sentence has to be exactly identical
            print(f"  {lang}: {tok.pieces(sentence)}")