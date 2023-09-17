import ast
from collections import defaultdict
from enum import Enum
import json
import re
from typing import List, Tuple
import numpy as np
from torch.utils.data import Dataset
from datasets import load_from_disk
import os
import pandas as pd

class DataDict(dict):

    def __init__(self, data):
        """
            :param root_directory: Root directory of the project.
            :param mode: Mode on which the data will be loaded:
                - "original": samples of the original train, dev and test splits. 
                - "annotated_rationales": samples of the annotated fair data that belongs to the train, dev an test splits. 
                - "non_annotated_rationales": original splits excluding the annotated samples for the fair dataset.
                - "train_on_non_annotated": original but the train split excludes the annotated samples.
        """
        super().__init__(**data)
        self._data = data

    def __getitem__(self, key):
        if key in ["train", "validation", "test"]:
            return super().__getitem__(key)
        else:
            raise ValueError("Split availables are 'train', 'validation' or 'test'.")

    def __setitem__(self, key, value):
            raise ValueError("DataDict object is not writable")

    def __repr__(self):
        return f"{self.__class__.__name__}(train samples={len(self._data['train'])}, validation samples={len(self._data['validation'])}, test samples={len(self._data['test'])})"

    def __str__(self):
        return f"{self.__class__.__name__}(train samples={len(self._data['train'])}, validation samples={len(self._data['validation'])}, test samples={len(self._data['test'])})"
    

class BaseDataset(Dataset):
    
    groups = ["BO", "BY", "LO", "LY", "WO", "WY"]

    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return {
            "sentence": self.data.loc[idx,"sentence"],
            "label": self.data.loc[idx,"label"],
            "sentence_id": self.data.loc[idx,"sentence_id"],
            **{group: self.data.loc[idx,group] for group in self.groups}
        }
    
    def __len__(self):
        return len(self.data)


class SST2DataDict(DataDict):

    def __init__(self, root_directory, mode="original"):
        """
            :param root_directory: Root directory of the project.
            :param mode: Mode on which the data will be loaded:
                - "original": samples of the original train, dev and test splits. 
                - "trainon_nonannot_valon_nonannot_teston_annot": Train on the original train set excluding the annotated samples of this set, validate on the original validation set excluding the annotated samples of this set and test on the annotated samples.
        """
        self.root_directory = root_directory
        self.mode = mode
        _data = load_from_disk(os.path.join(root_directory,"data/sst2"))
        dfs = {}
        if mode == "original":
            for _split in ["train", "validation", "test"]:
                _data_split = _data[_split].to_pandas().reset_index(drop=True)
                _data_split.rename(columns={"idx": "sentence_id"},inplace=True)
                for group in BaseDataset.groups:
                    _data_split[group] = None
                dfs[_split] = BaseDataset(_data_split)
        elif mode == "trainon_nonannot_valon_nonannot_teston_annot":
            for _split in ["train", "validation"]:
                annotations_path = os.path.join(root_directory, f"data/fair-data/data/SST/sst2_{_split}set_indexes.csv")
                idxs_from_annotations = pd.read_csv(annotations_path)["sst2_idxs"]
                _data_split = _data[_split].to_pandas()
                _data_split = _data_split[~_data_split.idx.isin(idxs_from_annotations)].reset_index(drop=True)
                _data_split.rename(columns={"idx": "sentence_id"},inplace=True)
                for group in BaseDataset.groups:
                    _data_split[group] = None
                dfs[_split] = BaseDataset(_data_split)
            test_dfs = []
            for _split in ["train", "validation", "test"]:
                annotations_path = os.path.join(root_directory, f"data/fair-data/data/SST/sst2_{_split}set_indexes.csv")
                idxs_from_annotations = pd.read_csv(annotations_path)["sst2_idxs"]
                _data_split = _data[_split].to_pandas()
                _data_split = _data_split[_data_split.idx.isin(idxs_from_annotations)]#.reset_index(drop=True)
                _data_split = self._add_rationale_annotations(_data_split, root_directory)
                test_dfs.append(_data_split)
            test_dfs = pd.concat(test_dfs,axis=0).reset_index(drop=True)
            test_dfs.rename(columns={"idx": "sentence_id"},inplace=True)
            dfs[_split] = BaseDataset(test_dfs)
        else:
            raise ValueError(f"Dataset sst2 not supported on mode {mode}.")
        super().__init__(dfs)

    @property
    def num_labels(self):
        return 2
    
    @staticmethod
    def _add_rationale_annotations(_data_split, root_directory):
        _data_split = _data_split.set_index("idx")
        for group in BaseDataset.groups:
            df = pd.read_csv(os.path.join(root_directory,f"data/fair-data/data/SST/{group}_processed.csv"))
            df = df.loc[:,["sst2_id","rationale_binary"]]
            df = df.set_index("sst2_id")
            df["rationale_binary"] = df["rationale_binary"].apply(ast.literal_eval)
            _data_split[group] = None
            valid_idx = df.index[df.index.isin(_data_split.index)].values
            _data_split.loc[valid_idx,group] = df.loc[valid_idx,"rationale_binary"]
        _data_split = _data_split.reset_index(drop=False)
        return _data_split



class DYNASENTDataDict(DataDict):

    label2integer = {"positive": 0, "negative": 1, "neutral": 2}

    def __init__(self, root_directory, mode="original"):
        """
            :param root_directory: Root directory of the project.
            :param mode: Mode on which the data will be loaded:
                - "original": samples of the original train, dev and test splits. 
                - "trainon_nonannot_valon_nonannot_teston_annot": Train on the original train set excluding the annotated samples of this set, validate on the original validation set excluding the annotated samples of this set and test on the annotated samples.
        """
        self.root_directory = root_directory
        self.mode = mode

        _data = {}
        if mode == "original":
            for _split in ["train", "validation", "test"]:
                _data_split = self._read_split(_split)
                for group in BaseDataset.groups:
                    _data_split[group] = None
                _data[_split] = BaseDataset(_data_split)
        elif mode == "trainon_nonannot_valon_nonannot_teston_annot":
            for _split in ["train", "validation"]:
                _data_split = self._read_split(_split)
                for group in BaseDataset.groups:
                    _data_split[group] = None
                _data[_split] = BaseDataset(_data_split)
            df = pd.read_csv(os.path.join(root_directory,"data/fair-data/data/dynasent/BO_processed.csv"))
            sentences_id_str = list(set(df["originaldata_id"]))
            _data_split = self._read_split("test",include_only=sentences_id_str)
            _data_split = self._add_rationale_annotations(_data_split, root_directory)
            _data["test"] = BaseDataset(_data_split)
        else:
                raise ValueError(f"Dataset dynasent not supported on mode {mode}.")
        super().__init__(_data)

    @staticmethod
    def _add_rationale_annotations(_data_split, root_directory):
        _data_split = _data_split.set_index("sentence_id")
        for group in BaseDataset.groups:
            df = pd.read_csv(os.path.join(root_directory,f"data/fair-data/data/dynasent/{group}_processed.csv"))
            df = df.loc[:,["originaldata_id","rationale_binary"]]
            df["originaldata_id"] = df["originaldata_id"].apply(lambda i: int(i[len("r2-") :]))
            df = df.set_index("originaldata_id")
            df["rationale_binary"] = df["rationale_binary"].apply(ast.literal_eval)
            _data_split[group] = None
            _data_split.loc[df.index,group] = df.loc[:,"rationale_binary"]
        _data_split = _data_split.reset_index(drop=False)
        return _data_split


    @property
    def num_labels(self):
        return len(self.label2integer)

    def _read_split(self, split, include_only=None):
        if split == "validation":
            split = "dev"
        sentences = []
        labels = []
        sentences_ids = []
        with open(os.path.join(self.root_directory,f"data/dynasent-v1.1/dynasent-v1.1-round02-dynabench-{split}.jsonl")) as f:
            for line in f:
                d = json.loads(line)
                if (include_only is None and d["gold_label"] in self.label2integer.keys()) \
                or (include_only is not None and d["text_id"] in include_only and d["gold_label"] in self.label2integer.keys()):
                    sentences.append(d["sentence"])
                    labels.append(self.label2integer[d["gold_label"]])
                    sentences_ids.append(int(d["text_id"][len("r2-") :]))
        return pd.DataFrame({"sentence": sentences, "label": labels, "sentence_id": sentences_ids})


class COSEDataDict(DataDict):

    label2integer = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

    def __init__(self, root_directory, mode="original"):
        """
            :param root_directory: Root directory of the project.
            :param mode: Mode on which the data will be loaded:
                - "original": samples of the original train, dev and test splits. 
                - "trainon_nonannot_valon_nonannot_teston_annot": Train on the original train set excluding the annotated samples of this set, validate on the original validation set excluding the annotated samples of this set and test on the annotated samples.
        """
        self.root_directory = root_directory
        self.mode = mode

        _data = {}
        if mode == "original":
            for _split in ["train", "validation", "test"]:
                _data_split = self._read_split(_split)
                for group in BaseDataset.groups:
                    _data_split[group] = None
                _data[_split] = BaseDataset(_data_split)
        elif mode == "trainon_nonannot_valon_nonannot_teston_annot":
            for _split in ["train", "validation"]:
                _data_split = self._read_split(_split)
                for group in BaseDataset.groups:
                    _data_split[group] = None
                _data[_split] = BaseDataset(_data_split)
            _data_split = self._read_split("test")
            df_annot = pd.read_csv(os.path.join(root_directory,f"data/fair-data/data/cose/BO_processed.csv"))
            _data_split = _data_split[_data_split["sentence_id"].apply(lambda x: x.split("_")[0]).isin(df_annot["originaldata_id"])]
            _data_split = self._add_rationale_annotations(_data_split, root_directory)
            _data["test"] = BaseDataset(_data_split)
        else:
            raise ValueError(f"Dataset CoS-E not supported on mode {mode}.")
        super().__init__(_data)

    def _add_rationale_annotations(self,_data_split, root_directory):
        _data_split = _data_split.set_index("sentence_id")
        for group in BaseDataset.groups:
            df = pd.read_csv(os.path.join(root_directory,f"data/fair-data/data/cose/{group}_processed.csv"))
            df = df.loc[:,["originaldata_id","rationale_binary"]]
            df = df.set_index("originaldata_id")
            df["rationale_binary"] = df["rationale_binary"].apply(ast.literal_eval)
            _data_split[group] = None
            valid_idx = df.index[df.index.isin(_data_split.index)].values
            _data_split.loc[valid_idx,group] = df.loc[valid_idx,"rationale_binary"]
        _data_split = _data_split.reset_index(drop=False)
        return _data_split

    @property
    def num_labels(self):
        return len(self.label2integer)

    def _label2int(self, y: List[str]) -> List[int]:
        return [self.label2integer[i] for i in y]

    def _read_split(self, split):
        if split == "validation":
            split = "val"
        ids_queries = []
        data = defaultdict(defaultdict)
        with open(os.path.join(self.root_directory,f"data/cose/{split}.jsonl")) as f:
            for line in f:
                d = json.loads(line)
                id = d["annotation_id"]
                data[id]["query"] = d["query"].split("[sep]")
                data[id]["label"] = d["classification"]
                data[id]["evidence"] = d["evidences"][0][0]["text"]
                ids_queries.append(id)
        with open(os.path.join(self.root_directory,f"data/cose/docs.jsonl")) as f:
            for line in f:
                d = json.loads(line)
                id_ = d["docid"]
                if id_ in ids_queries:
                    # removing the whitepaces left after spacy tokenization
                    sent, _, _ = self.merge_cose_whitespaces_sentence(d["document"])
                    data[id_]["question"] = sent
    
        x, y_int = zip(
            *[
                (
                    [data[i]["question"], data[i]["evidence"]] + data[i]["query"],
                    self._label2int([data[i]["label"]])[0],
                )
                for i in ids_queries
            ]
        )
        ids_queries_final = ids_queries

        return pd.DataFrame.from_dict({"sentence": x, "label": y_int, "sentence_id": ids_queries_final},orient="index").transpose()

    def merge_cose_whitespaces_sentence(self, sentence: str) -> Tuple[str, List[int], int]:
        dash = False
        rationale_id_merged = []
        word_to_list_of_word_id = {
            word: [
                self._get_token(sentence, w.start())
                for w in re.finditer(r"\b" + re.escape(word) + r"\b", sentence)
            ]
            for word in sentence.split()
        }
        for k, v in word_to_list_of_word_id.items():
            if len(v) == 0:
                word_to_list_of_word_id[k] = [
                    self._get_token(sentence, w.start())
                    for w in re.finditer(re.escape(k), sentence)
                ]

        matches = reversed(list(re.finditer(r'(\w+)\s([?,.!"](?:|$))', sentence)))
        for m in matches:
            span = m.span()
            words = m.group().split()
            try:
                rationale_id_merged.append(
                    self._select_token_from_list(
                        word_to_list_of_word_id[words[1]], sentence[: span[1]]
                    )
                )
            except KeyError:
                print(sentence, words)
            sentence = (
                sentence[: span[0]]
                + sentence[span[0] : span[1]].replace(" ", "")
                + sentence[span[1] :]
            )

        matches_contractions = reversed(
            list(
                re.finditer(r"(\w+)\s(\'s|\'re|n\'t|\'ll|\'ve|\'t|\'am|\'m|\'d)", sentence)
            )
        )
        for m in matches_contractions:
            span = m.span()
            words = m.group().split()
            rationale_id_merged.append(
                self._select_token_from_list(
                    word_to_list_of_word_id[words[1]], sentence[: span[1]]
                )
            )
            sentence = (
                sentence[: span[0]]
                + sentence[span[0] : span[1]].replace(" ", "")
                + sentence[span[1] :]
            )

        # Correct specific cases:
        if " - " in sentence:
            idx_dash = sentence.index(" - ")
            dash = sentence.split().index("-")
            sentence = sentence[:idx_dash] + "-" + sentence[idx_dash + 3 :]
            rationale_id_merged.append(
                self._select_token_from_list(
                    word_to_list_of_word_id["-"], sentence[: idx_dash + 3]
                )
            )
            # Also the word to the left and right side of the dash
            rationale_id_merged.append(rationale_id_merged[-1] + 1)
        if sentence.endswith("? ."):
            sentence = sentence[:-2] + "."
            rationale_id_merged.append(max(word_to_list_of_word_id["."]))

        return sentence, rationale_id_merged, dash
    
    def _select_token_from_list(self, token_ids: List[int], sentence_half: str) -> int:
        if len(token_ids) == 1:
            return token_ids[0]
        else:
            # get number closest to a given value
            return min(token_ids, key=lambda x: abs(x - len(sentence_half.split())))
    
    def _get_token(self, st: List[str], i: int) -> int:
        word = st[:i].split(" ")[-1] + st[i:].split(" ")[0]  # else return the word
        word_list = st[: i + len(word)].split()
        return len(word_list)


class SupportedDatasets(str, Enum):
    SST2 = "sst2"
    DYNASENT = "dynasent"
    COSE = "cose"

    def __str__(self):
        return self.value


def load_dataset(dataset_name, root_directory, mode="original"):
    if dataset_name == SupportedDatasets.SST2:
        return SST2DataDict(root_directory, mode=mode)
    elif dataset_name == SupportedDatasets.DYNASENT:
        return DYNASENTDataDict(root_directory, mode=mode)
    elif dataset_name == SupportedDatasets.COSE:
        return COSEDataDict(root_directory, mode=mode)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")