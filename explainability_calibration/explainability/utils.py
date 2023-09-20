import re


def align_annotations_with_model(tokenizer, sentence, annotations):

    def space_tokenize(s):
        splitted, offset_mapping = zip(*[(m.group(0), (m.start(), m.end())) for m in re.finditer(r'\S+', s)])
        return list(splitted), list(offset_mapping)

    _, offset_mapping = space_tokenize(sentence)
    encoded_input = tokenizer(sentence, return_offsets_mapping=True)
    new_annotations = []
    for start, end in encoded_input.offset_mapping:
        if start == end == 0:
            continue
        counter = 0
        for mm, MM in offset_mapping:
            if start >= mm and end <= MM:
                break
            counter += 1
        if counter >= len(annotations):
            new_annotations.append(0)
        else:
            new_annotations.append(annotations[counter])
            
    return new_annotations


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base", local_files_only=True)
    # sentence = "This is theee most  beautifully of the sentences"
    sentence = 'The stylist stations are spread  far apart so no one else is eavesdropping or breathing down your neck'
    print(tokenizer.tokenize(sentence))
    annotations = [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1]
    new_annotations = align_annotations_with_model(tokenizer, sentence, annotations)
    print(new_annotations)