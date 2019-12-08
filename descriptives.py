import jsonlines
from collections import Counter
from collections import defaultdict


def read_in_jsonl(file):
    with open(file, 'r') as infile:
        reader = jsonlines.Reader(infile)
        lines = reader.iter()
        lines = list(lines)
    return lines


def count_entity_types(json_docs):
    all_types = []
    for annotation in json_docs:
        labels = annotation['labels']
        all_types.append([type for start, end, type in labels])

    return Counter(all_types)


def spans_by_type(json_docs):
    spans_by_type = defaultdict(list)
    for annotation in json_docs:
        labels = annotation['labels']
        text = annotation['text']
        for start, end, type in labels:
            span = text[start:end].lower().replace("\n", " ")
            spans_by_type[type].append(span)

    for type, spans in spans_by_type.items():
        spans_by_type[type] = Counter(spans)

    return spans_by_type


if __name__=="__main__":

    luc = read_in_jsonl("annotated_batches/batch_1_annotated_lucino.jsonl")
    em = read_in_jsonl("annotated_batches/batch_3_annotated_emily.jsonl")
    sam = read_in_jsonl("annotated_batches/batch_4_annotated_samantha.jsonl")
    mark = read_in_jsonl("annotated_batches/batch_2_annotated_mark.jsonl")
    all = read_in_jsonl("annotated_data.jsonl")

    #print("LUCINO ", count_entity_types(luc))
    #print("EMILY ", count_entity_types(em))
    #print("SAM ", count_entity_types(sam))
    #print("MARK ", count_entity_types(mark))
    #print("ALL ", count_entity_types(all))

    for type, spans in spans_by_type(all).items():
        print(len(spans))
        print(type, spans)



