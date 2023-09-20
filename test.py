from transformers import AutoTokenizer
import numpy as np
import re

def align_with_model(tokenizer, sentence, annotations):

    def space_tokenize(s):
        s = re.sub(r"\s+"," ",s)
        s = re.sub(r"^\s","",s)
        s = re.sub(r"\s$","",s)
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
            if start >= mm and start < MM:
                break
            counter += 1
        new_annotations.append(annotations[counter])
            
        # counter = 0
        # start_idx = None
        # for i, w in enumerate(tokenized_sentence):
        #     for c in w:
        #         if counter == start:
        #             start_idx = i
        #             break
        #         counter += 1
        #     if start_idx is not None:
        #         break
        #     counter += 1
        # print(start_idx)
        # new_annotations.append(annotations[start_idx])
        
    return new_annotations


def main():
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base", local_files_only=True)
    # sentence = "This is theee most  beautifully of the sentences"
    sentence = 'The stylist stations are spread  far apart so no one else is eavesdropping or breathing down your neck'
    print(tokenizer.tokenize(sentence))
    annotations = [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1]
    new_annotations = align_with_model(tokenizer, sentence, annotations)
    print(new_annotations)



    # print(tokenizer(sentence, return_offsets_mapping=True))
    # print(tokenizer.tokenize(sentence,add_special_tokens=True))
    # print(np.argsort(-np.array([1,4,3,5,20,6,8,9,10]))[:4])
    # print(np.argsort(np.array([1,4,3,5,20,6,8,9,10]))[:-5:-1])
    


if __name__  == "__main__":
    main()