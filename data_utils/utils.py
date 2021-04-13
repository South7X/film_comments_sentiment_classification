#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
df = pd.read_csv("../data/train_data.csv", sep=',')
pos = df[df.label == 'positive']
neg = df[df.label == 'negative']
pos_train = pos.sample(frac=0.8, random_state=42)
pos_dev = pos.append(pos_train).drop_duplicates(keep=False)
neg_train = neg.sample(frac=0.8, random_state=42)
neg_dev = neg.append(neg_train).drop_duplicates(keep=False)
train = pos_train.append(neg_train)
dev = pos_dev.append(neg_dev)
train.to_csv("../data/train.csv", header=True, index=False, sep=",")
dev.to_csv("../data/dev.csv", header=True, index=False, sep=",")
# train = pos[:len].merge(neg[:len])
# dev = pos[len:].merge(neg[len:])
# print(train.shape[0], " ", dev.shape[0])
