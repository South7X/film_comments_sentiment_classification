#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
import pandas as pd
import os
from transformers import BertConfig, BertTokenizer
from config import base_config


class bertDatasets(Dataset):
    def __init__(self, tokenizer, args, mode=0):
        super(bertDatasets, self).__init__()
        self.mode = mode
        self.args = args
        self.path = self._get_data_path()
        self.df = self._get_data_file()
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        series = self.df.iloc[index, :]
        if "label" in set(series.index):
            sen_id, text, label = series
        else:
            sen_id, text = series
            label = None
        return self.transform(sen_id, text, label)

    def __len__(self):
        return self.df.shape[0]

    def _get_data_path(self):
        if self.mode == 0:  # train the model
            return os.path.join(self.args.data_dir, self.args.train_name)
        if self.mode == 1:
            return os.path.join(self.args.data_dir, self.args.dev_name)
        if self.mode == 2:  # predict the test data
            return os.path.join(self.args.data_dir, self.args.test_name)

    def _get_data_file(self):
        return pd.read_csv(self.path, sep=',')

    def transform(self, sen_id, text, label):
        if label is None:
            label = 0
        else:
            dic = {'positive': 1, 'negative': 0}
            label = dic[label]
        sentence_tokens = self.tokenizer.tokenize(text)
        if len(sentence_tokens) > self.args.max_seq_len - 2:
            sentence_tokens = sentence_tokens[:self.args.max_seq_len - 2]
        tokens = ['[CLS]'] + sentence_tokens + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        mask_ids = [1] * len(input_ids)
        padding_len = self.args.max_seq_len - len(tokens)
        input_ids = input_ids + [0] * padding_len
        mask_ids = mask_ids + [0] * padding_len
        input_ids = torch.tensor(input_ids)
        mask_ids = torch.tensor(mask_ids)
        label = torch.tensor(label, dtype=torch.long)
        assert input_ids.shape[0] == mask_ids.shape[0], "assert {} {}". format(input_ids.shape[0], mask_ids
                                                                               .shape[0])
        return input_ids, mask_ids, label


if __name__ == '__main__':
    args = base_config
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_path)
    dataset = bertDatasets(tokenizer, args)
    sampler = RandomSampler(dataset)
    print("total data: %d" % len(dataset))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, sampler=sampler)
    for batch in dataloader:
        # sentence_ids = batch[0]
        # print("sentence_ids: ", len(sentence_ids[0]))
        print(batch)
        break
    print(tokenizer.convert_ids_to_tokens([101,101,101]))


