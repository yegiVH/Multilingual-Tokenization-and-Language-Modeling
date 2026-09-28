import json
from data import LANGUAGES, read_lines
from character_tokenizer import CharTokenizer
from bpe_tokenizer import BPETokenizer

# analyzes one tokenizer
def analyze_tokenizer(tokenizer, tokenizer_name):
    vocab_size = tokenizer.vocab_size()
    
    results = {}
    
    for lang in LANGUAGES: # for each lng
        sentences = read_lines("valid", lang) # storing all the lines in validation file
        
        total_tokens = 0
        total_characters = 0
    
        for sent in sentences: # we go through each sentence
            total_characters += len(sent)
            token_ids = tokenizer.encode(sent)
            total_tokens += len(token_ids)
        
        average_tokens_per_sentence = (total_tokens / len(sentences))
        characters_per_token = (total_characters / total_tokens)
    
        # for each lang we save the results
        results[lang] = {
            "vocab_size" : vocab_size,
            "total_tokens" : total_tokens,
            "average_tokens_per_sentence": average_tokens_per_sentence,
            "characters_per_token" : characters_per_token,
        }
    
    return results

# for printing everything in a ncie format
def print_summary(all_results):
    print("\n")
    print("=" * 100)
    print("Summary Part 1") 
    print("=" * 100)
    # I want to print it in a table so
    
    # first headers
    print(
        f"{'Tokenizer':<15}"
        f"{'Language':<12}"
        f"{'Vocab':>10}"
        f"{'Total tokens':>18}"
        f"{'Avg tokens/sent':>20}"
        f"{'Chars/token':>18}"
    )
    
    print("-"*100)
    
    for tokenizer_name, language_results in all_results.items():
        for lang, result in language_results.items():
            # and data rows
            print(
                f"{tokenizer_name:<15}"
                f"{lang:<12}"
                f"{result['vocab_size']:>10}"
                f"{result['total_tokens']:>18}"
                f"{result['average_tokens_per_sentence']:>20.3f}"
                f"{result['characters_per_token']:>18.3f}"
            )
            

def print_examples(tokenizers):
    print("\n\n")
    print("="*80)
    print("Tokenization Examples")
    print("="*80)
    
    for lang in LANGUAGES:
        sentence = read_lines("valid", lang)[0] # cuz it is an example just first validation sentence is enough
        print("\n")
        print("-"*80)
        print(f"Language: {lang}")
        print("-"*80)
        
        print("\nOriginal:")
        print(sentence)
        
        for tokenizer_name, tokenizer in tokenizers.items():
            print(f"\n{tokenizer_name}:")
            print(tokenizer.pieces(sentence))
            
            
if __name__ == "__main__":
    # we first load the tokenizers
    tokenizers = {
        "Character" : CharTokenizer.load("char_vocab.json"),
        "BPE-4000" : BPETokenizer("bpe4000.model"),
        "BPE-10000" : BPETokenizer("bpe10000.model"),
    }
    
    # analyzing
    all_results = {}
    
    for tokenizer_name, tokenizer in tokenizers.items():
        all_results[tokenizer_name] = analyze_tokenizer(tokenizer, tokenizer_name)

    print_summary(all_results)
    print_examples(tokenizers)
    
    # saving the results
    with open("part1_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print("\nResults saved to part1_results.json")