from torchtext.vocab import GloVe
import torch
import os.path
import numpy as np


class GloveExtended(GloVe):
    def pad_init(dim): return torch.zeros(dim)
    update = False
    extended_w = {}

    def __init__(self, **kwargs) -> None:
        super(GloveExtended, self).__init__(**kwargs)

        pad = None
        self.avg, self.std = None, None

        if os.path.isfile("pad_vec.pt") and os.path.isfile("unk_vec.pt"):
            pad = torch.load("pad_vec.pt")
            unk = torch.load("unk_vec.pt")
        else:

            vecs = np.array(self.vectors)
            self.avg = np.mean(vecs, axis=0)
            self.std = np.std(vecs, axis=0)

            els = self.generate_new(1)

            pad = torch.Tensor(els[0])
            unk = torch.Tensor(self.avg)

            torch.save(pad, "pad_vec.pt")
            torch.save(unk, "unk_vec.pt")

        pad = torch.zeros(300)

        self.pad_init = lambda _: pad
        self.unk_init = lambda _: unk

    def set_update(self, value):
        if value:
            if self.avg is None or self.std is None:
                vecs = np.array(self.vectors)
                self.avg = np.mean(vecs, axis=0)
                self.std = np.std(vecs, axis=0)
            self.update = True
        else:
            self.update = False

    def generate_new(self, how_many=1):
        els = []
        for d in range(len(self.avg)):
            els.append(list(np.random.normal(
                self.avg[d], self.std[d], how_many)))
        els = np.array(els)
        els = els.transpose()
        return els

    def __getitem__(self, token):
        if token in self.stoi:
            return self.vectors[self.stoi[token]]
        else:
            if token == "[PAD]":
                return self.pad_init(torch.Tensor(self.dim))
            else:
                if self.update:
                    if token not in self.extended_w:
                        self.extended_w[token] = torch.Tensor(
                            self.generate_new()[0])
                    return self.extended_w[token]
                else:
                    return self.unk_init(torch.Tensor(self.dim))
