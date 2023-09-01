from enum import Enum
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
import os

class SupportedDatasets(str, Enum):
    SST2 = "sst2"
    DYNASENT = "dynasent"
    COSE = "cose"
    COSE_SIMPLIFIED = "cose_simplified"

    def __str__(self):
        return self.value


def load_dataset(dataset_name, root_directory):
    if dataset_name == SupportedDatasets.SST2:
        return SST2Dataset(root_directory)
    elif dataset_name == SupportedDatasets.DYNASENT:
        return DYNASENTDataset()
    elif dataset_name == SupportedDatasets.COSE:
        return COSEDataset(simplified=False)
    elif dataset_name == SupportedDatasets.COSE_SIMPLIFIED:
        return COSEDataset(simplified=True)
    

class SST2Dataset(Dataset):
    def __init__(self, root_directory):
        self._data = load_from_disk(os.path.join(root_directory,"data/sst2"))

class DYNASENTDataset(Dataset):
    pass

class COSEDataset(Dataset):
    pass


class DynamicPaddingCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, sentences_batch):
        return self.tokenizer(
            sentences_batch,
            padding=True,
            return_tensors="pt",
            truncate=True,
            max_seq_len=self.tokenizer.max_seq_len
        )


class LoaderWithDynamicPadding(DataLoader):
    def __init__(self,tokenizer,*args,**kwargs):
        super().__init__(
            *args,
            collate_fn=DynamicPaddingCollator(
                tokenizer=tokenizer
            ),
            **kwargs
        )