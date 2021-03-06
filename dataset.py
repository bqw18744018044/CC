# -*- coding:utf-8 -*-
"""
Author: Qiangwei Bai
Date: 2022-04-10 11:20:15
LastEditTime: 2022-04-11 09:56:43
LastEditors: Qiangwei Bai
FilePath: /CC/dataset.py
Description: 
"""
import os
import torch
import pandas as pd

from tqdm import tqdm
from typing import List
from logger import logger
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import BertTokenizer
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence


class SimpleDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: BertTokenizer, labels: List[int] = None, max_length: int = 512, verbose = True):
        self.total = len(texts)
        if labels:
            assert len(texts)==len(labels)
        if verbose:
            texts = tqdm(texts)
        self.inputs = [tokenizer.encode_plus(text, return_tensors="pt", max_length=max_length, truncation=True) for text in texts]
        self.labels = labels

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        item = {
            "input_ids": self.inputs[idx]["input_ids"],
            "token_type_ids": self.inputs[idx]["token_type_ids"],
            "attention_mask": self.inputs[idx]["attention_mask"]
        }
        if self.labels:
            item["label"] = self.labels[idx]
        return item


@dataclass
class DataCollator:
    def __call__(self, features):
        batch_input_ids = [f["input_ids"].squeeze() for f in features]
        batch_token_type_ids = [f["token_type_ids"].squeeze() for f in features]
        batch_attention_mask = [f["attention_mask"].squeeze() for f in features]
        batch_label = torch.LongTensor([f["label"] for f in features])
        batch_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=0)
        batch_attention_mask = pad_sequence(batch_attention_mask, batch_first=True, padding_value=0)
        batch_token_type_ids = pad_sequence(batch_token_type_ids, batch_first=True, padding_value=0)
        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "token_type_ids": batch_token_type_ids,
            "labels": batch_label
        }


def get_dataset(name: str):
    filename = os.path.join("./data", name)
    data = pd.read_csv(filename)
    labels = data["label"].tolist()
    texts = data["text"].tolist()
    num_classes = len(set(labels))
    return texts, labels, num_classes

if __name__ == "__main__":
    pass