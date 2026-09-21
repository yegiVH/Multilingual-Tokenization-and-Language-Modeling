## I do data loading here

from pathlib import Path

DATA =  Path("/srv/data/lt2326-h26/a1")
LANGUAGES = ["en", "tr", "zh"]

def read_lines(split, lang):
    with open(DATA / split / f"{lang}.txt", encoding="utf-8") as f: # so for each lng builds the path and opens it for reading the lines
        lines = [l.rstrip("\r\n") for l in f] # cleans each line and put it in list
    return [l for l in lines if l.strip()] # so that there is no empty line

if __name__ == "__main__":
    for split in ["train", "valid", "test"]:
        for lang in LANGUAGES:
            L = read_lines(split, lang) # so run the function on each lang
            chars = sum(map(len, L)) # number of chars total in the file
            avg_len = chars / len(L) # avg char per line
            print(split, lang, len(L), "sentences", chars, "chars", avg_len) # and I also print number of sentences and chars for each lang and split and the avg thing as well
        