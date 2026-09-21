## Questions
### 1. **Which language will require the most tokens?**  
As for character tokenizer, I think it produces long token sequences for all three languages, but for BPE, I think Turkish requires the most BPE tokens per sentence, then followed by English and then Chinese. Because Turkish has productive morphology, so a single word can contain a stem followed by several suffixes. A BPE tokenizer may split these words into several subword units. English though has many recurring words and subwords that BPE can potentially learn as larger units. Chinese characters are really different from those two, they can themselves represent meaningful units, and BPE may learn frequent multi-character sequences.

### 2. **What will happen when vocabulary size increases from 2,000 to 10,000?**  
I think 10K token BPE tokenizer would produce fewer tokens per sentence than the 2K tokenizer for all 3 languages. Because with larger vocabulary, BPE has more capacity to store frequent character sequences as individual tokens. For example, a sequence that is represented as several tokens with a 2k token vocabulary may become one or two tokens with 10k token vocabulary.

### 3. **What kinds of units do you expect BPE to learn in English, Turkish, and Chinese?** 
for English:  
many whole common words and recurring word fragments, like "ing" or "the" or "and".  

for Turkish:  
word stems and recurring morphological suffixes. I think the 2k tokenizer splits Turkish words into smaller pieces more often and 10k would be able to represent more frequent complete words or longer morphological sequences as single tokens.  

for Chinese:  
individual characters as well as frequent multi-character sequences. I know Chinese does not normally use spaces to separate words, so I don't expect the tokenizer to behave like it does for english.

### 5. **How will the number of characters per token change?**  
character < BPE 2k < BPE 10k 


### 6. **Which tokenizer do you expect to work best for language modeling?**   
Probably 10k BPE tokenizer then after it BPE 2k and then character tokenizer I think would perform less efficiently.  
As I said before a larger BPE vocabulary should produce shorter sequences and allow the Transformer to operate on meaningful recurring units rather than individual characters.
But I don't know if the 10k vocabulary would also increase the number of model paramers in the embedding and output layers.
