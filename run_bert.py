#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import os
from itertools import cycle
import logging
import random
from tqdm import tqdm
from torch import nn
import numpy as np
from config import base_config
from torch.nn.parallel import DataParallel
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from transformers import AdamW
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from data_utils.bertDatasets import bertDatasets
from sklearn.metrics import accuracy_score

class Instructor(object):

    def __init__(self, args):
        self.args = args
        self.set_seed()
        self.load_model()
        self.train_set = None
        self.dev_set = None
        self.test_set = None


    def train(self):
        if not self.train_set:
            self.train_set = bertDatasets(tokenizer=self.tokenizer, args=self.args, mode=0)
        sampler = RandomSampler(self.train_set)
        train_loader = DataLoader(self.train_set, batch_size=self.args.batch_size, sampler=sampler)
        print("Train loader length: {}".format(len(train_loader)))
        bar = tqdm(range(len(train_loader) * self.args.epochs), total=len(train_loader) * self.args.epochs)
        train_loader = cycle(train_loader)

        param_optimizer = list(self.model.named_parameters())
        param_optimizer = [n for n in param_optimizer]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        best_acc = 0

        for step in bar:
            batch = next(train_loader)
            input_ids = batch[0]
            mask_ids = batch[1]
            labels = batch[2]
            input_dict = {"input_ids": input_ids, "attention_mask": mask_ids, "labels": labels}
            if args.n_gpus > 0:
                for k, v in input_dict.items():
                    input_dict[k] = v.cuda()
            loss, logits = self.model(**input_dict)[:2]
            if self.args.n_gpus > 0:
                loss = loss.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            bar.set_description("Train loss: {}".format(loss.item()))
            if (step + 1) % self.args.eval_steps == 0:
                eval_acc = self.eval()
                print("Eval acc: {}".format(eval_acc))
                if eval_acc > best_acc:
                    best_acc = eval_acc
                    print("Saving model of best acc: {}".format(best_acc))
                    self.save_model()
        print("Training finished.")


    def eval(self):
        self.model.eval()
        if not self.dev_set:
            self.dev_set = bertDatasets(tokenizer=self.tokenizer, args=self.args, mode=1)
        sampler = RandomSampler(self.dev_set)
        dev_loader = DataLoader(dataset=self.dev_set, batch_size=self.args.eval_batch_size, sampler=sampler)
        bar = tqdm(dev_loader, total=len(dev_loader))

        all_labels = []
        all_logits = []
        eval_loss = 0
        cnt = 0
        for batch in bar:
            input_ids = batch[0]
            mask_ids = batch[1]
            labels = batch[2]
            input_dict = {"input_ids": input_ids, "attention_mask": mask_ids, "labels": labels}
            if args.n_gpus > 0:
                for k, v in input_dict.items():
                    input_dict[k] = v.cuda()
            with torch.no_grad():
                loss, logits = self.model(**input_dict)[:2]
            if self.args.n_gpus>1:
                loss = loss.mean().item()
            else:
                loss = loss.item()
            eval_loss += loss
            bar.set_description("batch {} eval loss: {}".format(cnt, loss))
            labels = labels.detach().cpu().numpy()
            logits = logits.detach().cpu().numpy()
            all_labels.append(labels)
            all_logits.append(logits)
            cnt = cnt + 1
        all_labels = np.concatenate(all_labels, axis=0)
        all_logits = np.concatenate(all_logits, axis=0)
        preds = all_logits.argmax(axis=1)
        acc = accuracy_score(y_true=all_labels, y_pred=preds)
        print("Overall eval loss: {}".format(eval_loss / len(dev_loader)))
        self.model.train()
        return acc


    def test(self):
        self.model.eval()
        if not self.test_set:
            self.test_set = bertDatasets(tokenizer=self.tokenizer, args=self.args, mode=2)
        sampler = SequentialSampler(self.test_set)
        test_loader = DataLoader(self.test_set, batch_size=self.args.eval_batch_size, sampler=sampler)
        bar = tqdm(test_loader, total=len(test_loader))

        all_logits = []
        for batch in bar:
            input_ids = batch[0]
            mask_ids = batch[1]
            labels = batch[2]
            input_dict = {"input_ids": input_ids, "attention_mask": mask_ids, "labels": labels}
            if self.args.n_gpus > 0:
                for k, v in input_dict.items():
                    input_dict[k] = v.cuda()
            with torch.no_grad():
                _, logits = self.model(**input_dict)[:2]
            logits = logits.detach().cpu().numpy()
            all_logits.append(logits)
        all_logits = np.concatenate(all_logits, axis=0)
        preds = all_logits.argmax(axis=1)
        self.write_result(preds, os.path.join(self.args.out_dir, "answer.txt"))
        print("Write answer to file.")
        self.model.train()

    def save_model(self):
        model_path = os.path.join(self.args.out_dir, "model/pytorch_model.bin")
        torch.save(self.model.state_dict(), model_path)

    def write_result(self, result, file):
        result = result.reshape(-1,1).tolist()
        with open(file, 'w') as f:
            for pred in result:
                f.write(str(pred[0])+'\n')

    def load_model(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.args.pretrained_path)
        self.config = BertConfig.from_pretrained(self.args.pretrained_path, num_labels=self.args.num_labels)
        self.model = BertForSequenceClassification.from_pretrained(self.args.pretrained_path, config=self.config)
        if self.args.cuda:
            self.model = self.model.cuda()
            if self.args.n_gpus > 1:
                self.model = DataParallel(self.model)

    def set_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = base_config
    args.cuda = torch.cuda.is_available()
    args.n_gpus = torch.cuda.device_count()
    if not os.path.exists(os.path.join(args.out_dir, "model")):
        os.makedirs(os.path.join(args.out_dir, "model"))
    trainer = Instructor(args)
    if args.do_train:
        trainer.train()
    if args.do_test:
        trainer.test()

