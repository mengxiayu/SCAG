
from torch.utils.data import Dataset
from pathlib import Path
import linecache
import torch
from typing import Callable, Dict, Iterable, List, Tuple, Union, Optional

class SeqClassificationDatset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        type_path="train",
        n_obs=None,
        prefix="",
        **dataset_kwargs
        ):
        super().__init__()
        self.src_file = Path(data_dir).joinpath(type_path + ".source")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".labels")
        self.src_lens = self.get_char_lens(self.src_file)
        self.max_source_length = max_source_length
        self.tokenizer = tokenizer
        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.dataset_kwargs = dataset_kwargs
    def __len__(self):
        return len(self.src_lens)
    def get_char_lens(self,data_file):
        return [len(x) for x in Path(data_file).open().readlines()]
    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        index = index + 1
        src_line = linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert src_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"

        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        source_inputs = self.encode_line(self.tokenizer, src_line, self.max_source_length)
        source_ids = source_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()
        target_ids = torch.Tensor([int(tgt_line)]).long()
        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "labels": target_ids,
        }
    def encode_line(self, tokenizer, line, max_length, pad_to_max_length=True, return_tensors="pt"):
        """Only used by LegacyDataset"""
        return tokenizer(
            [line],
            max_length=max_length,
            padding="max_length" if pad_to_max_length else None,
            truncation=True,
            return_tensors=return_tensors,
            **self.dataset_kwargs,
        )
