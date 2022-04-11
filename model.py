# -*- coding:utf-8 -*-
"""
Author: Qiangwei Bai
Date: 2022-04-10 11:21:56
LastEditTime: 2022-04-10 11:22:23
LastEditors: Qiangwei Bai
FilePath: /CC/model.py
Description: 
"""
import torch
import numpy as np
import torch.nn as nn

from numpy import ndarray
from dataclasses import dataclass
from typing import Union, Optional
from torch.nn import functional as F
from loss import InstanceLoss, ClusterLoss, DecLoss
from torch.utils.data import DataLoader
from dataset import get_dataset, SimpleDataset, DataCollator
from transformers.modeling_outputs import ModelOutput
from transformers import AutoTokenizer, AutoModel


@dataclass
class CCOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    embeddings: Optional[torch.FloatTensor] = None
    cluster_prob: Optional[torch.FloatTensor] = None
    labels: Optional[torch.LongTensor] = None


class CC(nn.Module):
    def __init__(self, backbone, num_classes: int, batch_size: int):
        super(CC, self).__init__()
        self.backbone = backbone
        bb_hidden_size = self.backbone.config.hidden_size
        self.instance_head = nn.Sequential(
            nn.Linear(bb_hidden_size, bb_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(bb_hidden_size, 128))
        self.cluster_head = nn.Sequential(
            nn.Linear(bb_hidden_size, bb_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(bb_hidden_size, num_classes),
            nn.Softmax(dim=1)
        )
        self.criterion_instance = InstanceLoss(batch_size, 0.5, torch.device("cuda")).to("cuda")
        self.criterion_cluster = ClusterLoss(num_classes, 1.0, torch.device("cuda")).to("cuda")
        self.criterion_dec = DecLoss()

    def  get_embeddings(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        embeddings = torch.sum(outputs[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return embeddings

    def get_cluster_prob(self, embeddings):
        norm_squared = torch.sum((embeddings.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            pos=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None):
        embed0 = self.get_embeddings(input_ids, attention_mask)
        embed1 = self.get_embeddings(input_ids, attention_mask)
        ins_feat1 = F.normalize(self.instance_head(embed0), dim=1)
        ins_feat2 = F.normalize(self.instance_head(embed1), dim=1)
        clu_feat1 = self.cluster_head(embed0)
        clu_feat2 = self.cluster_head(embed1)
        loss_instance = self.criterion_instance(ins_feat1, ins_feat2)
        loss_cluster = self.criterion_cluster(clu_feat1, clu_feat2)
        loss_dec = self.criterion_dec(clu_feat1)
        loss_dec + self.criterion_dec(clu_feat2)
        loss = loss_instance + loss_cluster
        return CCOutput(
            loss=loss,
            logits=None,
            embeddings=embed0,
            cluster_prob=clu_feat1,
            labels=labels
        )


if __name__ == "__main__":
    """
    model_path = "E:/pretrained/bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    backbone = AutoModel.from_pretrained(model_path)
    texts, labels, num_classes = get_dataset("SearchSnippets")
    dataset = SimpleDataset(texts=texts, labels=labels, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=28, collate_fn=DataCollator())
    batch = next(iter(dataloader))
    model = CC(backbone, 20, )
    out = model(**batch)
    print("aaa")
    """
    pass
