import numpy as np
import pandas as pd
import torch
import random
from datasets import load_dataset
from torchtext.data import get_tokenizer
from transformers import BertTokenizer


class Dataset(torch.utils.data.Dataset):
    name = "noname"
    multilabel = False

    def __init__(self, path, labels_ids=[], setting="standard"):
        super().__init__()
        self.path = path
        self.labels = None
        self.texts = None
        if setting == "BERT":
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = get_tokenizer("basic_english")
        self.setting = setting

        self.label_to_id = {}
        self.id_to_label = {}
        for tuple in labels_ids:
            self.label_to_id[tuple[0]] = tuple[1]
            self.id_to_label[tuple[1]] = tuple[0]
        self.sep = "[SEP]"
        self.pad = "[PAD]"

    def custom_load(self, used_set, vec_max_len, tokenize=True):
        pass

    def get_class_dist(self):
        dis = None
        if self.multilabel:
            dis = np.array([0]*len(self.labels[0]))

            for l in self.labels:
                dis += l

            dis = dis.tolist()

        else:
            dis = [0]*len(self.label_to_id)
            for l in self.labels:
                dis[l] += 1
        return dis

    def load_set(self, used_set="train", vec_max_len=256, tokenize=True):
        self.texts, self.labels = self.custom_load(
            used_set, vec_max_len, tokenize)
        self.correct_labels = self.labels.copy()

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        if self.setting == "BERT":
            return np.array(self.labels[idx])
        return self.labels[idx]

    def get_batch_texts(self, idx):
        if self.setting == "BERT":
            return self.texts["input_ids"][idx]
        return self.texts[idx]

    def get_batch_masks(self, idx):
        return self.texts["attention_mask"][idx]

    def get_batch_token_type(self, idx):
        return self.texts["token_type_ids"][idx]

    def __getitem__(self, idx):
        if self.setting == "BERT":
            return ({"input_ids": torch.tensor(self.get_batch_texts(idx)).squeeze(),
                    "attention_mask": torch.tensor(self.get_batch_masks(idx)).squeeze(),
                     "token_type_ids": torch.tensor(self.get_batch_token_type(idx)).squeeze()},
                    self.get_batch_labels(idx),
                    idx)
        return self.get_batch_texts(idx), self.get_batch_labels(idx), idx


class DatasetNLI2(Dataset):
    name = "SNLI"
    multilabel = False
    partitions = ["train", "test", "validation"]

    def custom_load(self, used_set, vec_max_len=256, tokenize=True):

        dataset = load_dataset("snli")

        premise = dataset[used_set]["premise"]
        hypothesis = dataset[used_set]["hypothesis"]
        labels = dataset[used_set]["label"]

        input = []
        labels_c = []

        tot_max_len = [0]*123

        indexes = list(range(len(labels)))

        if self.setting == "BERT":

            if tokenize:
                tmp = [premise[ind] + " " + self.sep + " " + hypothesis[ind]
                       for ind in indexes if labels[ind] != -1]
                input = self.tokenizer(tmp, padding='max_length', max_length=vec_max_len, truncation=True,
                                       return_tensors="pt")
            else:
                input = [x for x in premise[ind] + " " + self.sep + " " +
                         hypothesis[ind] for ind in indexes if labels[ind] != -1]

            labels_c = [labels[ind] for ind in indexes if labels[ind] != -1]
        else:
            for ind in indexes:
                if labels[ind] != -1:
                    text = self.tokenizer(
                        premise[ind]) + [self.sep] + self.tokenizer(hypothesis[ind])
                    tot_max_len[len(text)-1] += 1

                    if len(text) <= vec_max_len:

                        if len(text) < vec_max_len:
                            text = text+[self.pad]*(vec_max_len - len(text))

                        input.append(text)
                        labels_c.append(labels[ind])
            self.tot_max_len = tot_max_len

        return input, labels_c

    def get_stats(self):
        dataset = load_dataset("snli")

        partitions = ["train", "test", "validation"]

        stats = []

        lens = [0]
        vals = []

        for partition in partitions:

            premise = dataset[partition]["premise"]
            hypothesis = dataset[partition]["hypothesis"]
            labels = dataset[partition]["label"]

            counter = 0

            indexes = list(range(len(labels)))

            for ind in indexes:
                if labels[ind] != -1:
                    text = self.tokenizer(
                        premise[ind]) + [self.sep] + self.tokenizer(hypothesis[ind])

                    vals.append(len(text))

                    if len(text) >= len(lens):
                        lens += [0]*(1+len(text)-len(lens))

                    lens[len(text)] += 1
                    counter += 1
                else:
                    pass

            stats.append(counter)

        final_stats = []
        final_stats.append(tuple(stats))
        max_s_len = len(lens)

        lens = np.cumsum(lens)

        lens = lens/lens[-1]

        val = 0

        min_l = -1
        found = False

        for i, v in enumerate(lens):
            if min_l == -1 and v > 0:
                min_l = i

            if v > 0.85 and not found:
                val = i
                found = True

        final_stats.append((np.mean(vals), np.median(vals), np.std(vals)))
        final_stats.append((min_l, max_s_len))
        final_stats.append(len(self.label_to_id))
        final_stats.append(self.multilabel)
        final_stats.append(val)

        return final_stats


class DatasetIMDB2(Dataset):
    name = "IMDB"
    multilabel = False
    partitions = ["train", "test", "validation"]

    dataframe = None

    def custom_load(self, used_set, vec_max_len=256, tokenize=True):
        df = pd.read_csv("./data/imdb_pd.csv")
        df = df.loc[df.partition == used_set]
        texts, labels = df["text"].tolist(), df["label"].tolist()

        indexes = list(range(len(texts)))

        input = []
        labels_final = []

        if self.setting == "BERT":

            if tokenize:
                tmp = [texts[ind] for ind in indexes]
                input = self.tokenizer(tmp, padding='max_length', max_length=vec_max_len, truncation=True,
                                       return_tensors="pt")
            else:
                input = [x for x in texts[indexes]]

            labels_final = [self.label_to_id[labels[ind]] for ind in indexes]

        else:

            for ind in indexes:
                text = self.tokenizer(texts[ind])
                if len(text) < vec_max_len:
                    text = text+[self.pad]*(vec_max_len - len(text))
                if len(text) > vec_max_len:
                    text = text[0:vec_max_len+1]

                input.append(text)
                labels_final.append(self.label_to_id[labels[ind]])

        return input, labels_final

    def get_stats(self):
        df = pd.read_csv("./data/imdb_pd.csv")

        partitions = ["train", "test", "validation"]

        stats = []

        lens = [0]
        vals = []

        for partition in partitions:

            df2 = df.loc[df.partition == partition]
            texts, labels = df2["text"].tolist(), df2["label"].tolist()

            counter = 0

            indexes = list(range(len(labels)))

            for ind in indexes:

                if labels[ind] != -1:
                    text = self.tokenizer(texts[ind])

                    vals.append(len(text))

                    if len(text) >= len(lens):
                        lens += [0]*(1+len(text)-len(lens))

                    lens[len(text)] += 1
                    counter += 1

            stats.append(counter)

        final_stats = []
        final_stats.append(tuple(stats))
        max_s_len = len(lens)

        lens = np.cumsum(lens)

        lens = lens/lens[-1]

        val = 0

        min_l = -1
        found = False

        for i, v in enumerate(lens):
            if min_l == -1 and v > 0:
                min_l = i

            if v > 0.85 and not found:
                val = i
                found = True

        final_stats.append((np.mean(vals), np.median(vals), np.std(vals)))
        final_stats.append((min_l, max_s_len))
        final_stats.append(len(self.label_to_id))
        final_stats.append(self.multilabel)
        final_stats.append(val)

        return final_stats


class Reuters(Dataset):
    name = "Reuters"
    multilabel = True
    partitions = ["train", "test", "val"]

    dataframe = None

    def custom_load(self, used_set, vec_max_len=256, tokenize=True):
        path = self.path

        df = pd.read_csv("insert_path")

        df = df.loc[df.partition == used_set]

        texts, labels = df["Text"].tolist(), df["target_formatted"].tolist()
        indexes = list(range(len(texts)))

        input = []
        labels_final = []

        if self.setting == "BERT":

            if tokenize:
                tmp = [texts[ind] for ind in indexes]
                input = self.tokenizer(tmp, padding='max_length', max_length=vec_max_len, truncation=True,
                                       return_tensors="pt")
            else:
                input = [x for x in texts[indexes]]

            labels_final = [eval(labels[ind])
                            for ind in indexes]

        else:

            for ind in indexes:
                text = self.tokenizer(texts[ind])
                if len(text) < vec_max_len:
                    text = text+[""]*(vec_max_len - len(text))
                if len(text) > vec_max_len:
                    text = text[0:vec_max_len+1]

                input.append(text)
                labels_final.append(eval(labels[ind]))

        return input, labels_final

    def get_stats(self):
        path = self.path
        if self.path != "":
            path = "_"+self.path

        df = pd.read_csv("insert_path")

        partitions = ["train", "test", "val"]

        stats = []

        lens = [0]
        vals = []

        for partition in partitions:

            df2 = df.loc[df.partition == partition]
            texts, labels = df2["Text"].tolist(
            ), df2["target_formatted"].tolist()

            counter = 0

            indexes = list(range(len(labels)))

            for ind in indexes:

                if labels[ind] != -1:
                    text = self.tokenizer(texts[ind])

                    vals.append(len(text))

                    if len(text) >= len(lens):
                        lens += [0]*(1+len(text)-len(lens))

                    lens[len(text)] += 1
                    counter += 1

            stats.append(counter)

        final_stats = []
        final_stats.append(tuple(stats))
        max_s_len = len(lens)

        lens = np.cumsum(lens)

        lens = lens/lens[-1]

        val = 0

        min_l = -1
        found = False

        for i, v in enumerate(lens):
            if min_l == -1 and v > 0:
                min_l = i

            if v > 0.85 and not found:
                val = i
                found = True

        final_stats.append((np.mean(vals), np.median(vals), np.std(vals)))
        final_stats.append((min_l, max_s_len))
        final_stats.append(len(self.label_to_id))
        final_stats.append(self.multilabel)
        final_stats.append(val)

        return final_stats


class AAPD(Dataset):
    name = "AAPD"
    multilabel = True
    partitions = ["train", "test", "val"]

    dataframe = None

    def custom_load(self, used_set, vec_max_len=256, tokenize=True):
        path = self.path
        if self.path != "":
            path = "_"+self.path

        df = pd.read_csv("insert path")
        AAPD.dataframe = df

        texts, labels = df["Text"].tolist(), df["Labels"].tolist()
        indexes = list(range(len(texts)))

        input = []
        labels_final = []

        if self.setting == "BERT":

            if tokenize:
                tmp = [texts[ind] for ind in indexes]
                input = self.tokenizer(tmp, padding='max_length', max_length=vec_max_len, truncation=True,
                                       return_tensors="pt")
            else:
                input = [x for x in texts[indexes]]

            labels_final = [eval(labels[ind])
                            for ind in indexes]

        else:

            for ind in indexes:
                text = self.tokenizer(texts[ind])
                if len(text) < vec_max_len:
                    text = text+[""]*(vec_max_len - len(text))
                if len(text) > vec_max_len:
                    text = text[0:vec_max_len+1]

                input.append(text)
                labels_final.append(eval(labels[ind]))

        return input, labels_final

    def get_stats(self):
        partitions = ["train", "test", "val"]

        stats = []

        lens = [0]
        vals = []

        for partition in partitions:

            df2 = pd.read_csv("insert path to partition")
            texts, labels = df2["Text"].tolist(), df2["Labels"].tolist()

            counter = 0

            indexes = list(range(len(labels)))

            for ind in indexes:

                if labels[ind] != -1:
                    text = self.tokenizer(texts[ind])

                    vals.append(len(text))

                    if len(text) >= len(lens):
                        lens += [0]*(1+len(text)-len(lens))

                    lens[len(text)] += 1
                    counter += 1

            stats.append(counter)

        final_stats = []
        final_stats.append(tuple(stats))
        max_s_len = len(lens)

        lens = np.cumsum(lens)

        lens = lens/lens[-1]

        val = 0

        min_l = -1
        found = False

        for i, v in enumerate(lens):
            if min_l == -1 and v > 0:
                min_l = i

            if v > 0.85 and not found:
                val = i
                found = True

        final_stats.append((np.mean(vals), np.median(vals), np.std(vals)))
        final_stats.append((min_l, max_s_len))
        final_stats.append(len(self.label_to_id))
        final_stats.append(self.multilabel)
        final_stats.append(val)

        return final_stats
