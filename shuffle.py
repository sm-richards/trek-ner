import jsonlines
import random

def shuffle_corpus(file):
    with open(file, 'r') as infile:
        reader = jsonlines.Reader(infile)
        lines = reader.iter()
        lines = list(lines)
        random.shuffle(lines)

    with open(file, 'w') as outfile:
        writer = jsonlines.Writer(outfile)
        for line in lines:
            writer.write(line)


shuffle_corpus("annotated_data.jsonl")