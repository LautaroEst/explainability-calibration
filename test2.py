
import pandas as pd
import ast
import re
import numpy as np


def space_tokenize(s):
    splitted, offset_mapping = zip(*[(m.group(0), (m.start(), m.end())) for m in re.finditer(r'\S+', s)])
    return list(splitted), list(offset_mapping)

def main():
    df = pd.read_csv("data/fair-data/data/SST/BO_processed.csv")
    df["rationale_binary"] = df["rationale_binary"].apply(ast.literal_eval)
    print(sum(df["rationale_binary"].str.len() != df["sentence"].apply(lambda x: len(space_tokenize(x)[0]))))
    # df["sentence"] = df["sentence"].apply(lambda x: space_tokenize(x)[0])
    # df["rationale_index"] = df["rationale_index"].apply(ast.literal_eval)
    # df["rationale"] = df["rationale"].str.split(",")
    # mask = df.apply(lambda ds: [ds.loc["sentence"][i] for i in ds.loc["rationale_index"]] != [s for s in ds.loc["rationale"] if s not in [""," "]], axis=1)
    # print(mask.sum())

    # for i, row in df.loc[mask,["sentence","rationale", "rationale_binary", "rationale_index"]].iterrows():
    #     print(row.sentence)
    #     print(row.rationale)
    #     print(row.rationale_binary)
    #     print(row.rationale_index)
    #     print()

    
    # mask = sentences.str.len() != sequences.str.len()
    # print(mask.sum() / len(sequences))




    # rs = np.random.RandomState(10)
    # idx = rs.permutation(len(sentences))
    # for s1, s2 in zip(sentences.iloc[idx].head(),sequences.iloc[idx].head()):
    #     print(s1)
    #     print(s2)
    #     print()


if __name__ == "__main__":
    main()