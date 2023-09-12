import torch
from torch.utils.data import DataLoader, RandomSampler

    
class DynamicPaddingCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        batch = {k: [sample[k] for sample in batch] for k in batch[0]}
        tokenizer_output = self.tokenizer(
            batch["sentence"],
            padding=True,
            return_tensors="pt",
            truncation="longest_first",
            return_token_type_ids=True,
            return_attention_mask=True
        )
        batch.update(tokenizer_output)
        batch["label"] = torch.tensor(batch["label"],dtype=torch.long)
        return batch


class LoaderWithDynamicPadding(DataLoader):
    def __init__(self, dataset, tokenizer, batch_size, shuffle=True, random_state=None):
        if shuffle:
            generator = torch.Generator()
            if random_state is not None:
                generator.manual_seed(random_state)
            sampler = RandomSampler(
                dataset, replacement=False, generator=generator
            )
        else:
            sampler = None
        super().__init__(
            dataset=dataset,
            collate_fn=DynamicPaddingCollator(tokenizer=tokenizer),
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
        )