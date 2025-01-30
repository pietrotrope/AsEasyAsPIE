import torch.nn.utils.prune as prune
import numpy as np
import torch
from pruning.tools import retrieve_tuples


def choose_device(tensor):
    device = tensor.get_device()
    if device == -1:
        device = "cpu"
    return device


class Magnitude:
    def __init__(self, percentage, model):
        self.percentage = percentage
        self.model = model
        self.tuples = retrieve_tuples(model)

    def compute_new_masks(self):
        MagnitudePruningStructured.percentage = self.percentage
        mp = MagnitudePruningStructured()
        for layer, name in self.tuples:
            mp.apply(layer, name)


class MagnitudePruningStructured(prune.BasePruningMethod):
    percentage = 20
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        device = choose_device(default_mask)
        tmp = t.cpu().detach().numpy()
        mask = default_mask.cpu().detach().numpy()
        weights = tmp[np.nonzero(mask)]
        weights = np.absolute(weights.flatten())
        threshold = np.percentile(
            weights, MagnitudePruningStructured.percentage)
        new_mask = mask.copy()
        new_mask[abs(tmp) < threshold] = 0
        return torch.tensor(new_mask, dtype=torch.float32).to(device)


class MagnitudeUnstructered(Magnitude):
    def _compute_threshold(self, weights):
        return np.percentile(np.absolute(weights), self.percentage)

    def compute_new_masks(self):
        mp = MagnitudePruningUnstructured()
        concatenated = None

        for layer, name in self.tuples:
            weights = None
            mask = None
            if hasattr(layer, name+"_mask"):
                weights = layer.__getattr__(
                    name+"_orig").cpu().detach().numpy()
                mask = layer.__getattr__(name+"_mask").cpu().detach().numpy()
            else:
                weights = layer.__getattr__(name).cpu().detach().numpy()
                mask = np.ones(weights.shape)
            masked_weights = weights[np.nonzero(mask)]

            if concatenated is None:
                concatenated = masked_weights
            else:
                concatenated = np.concatenate(
                    (concatenated, masked_weights), axis=None)

        threshold = self._compute_threshold(concatenated)
        MagnitudePruningUnstructured.threshold = threshold

        for layer, name in self.tuples:
            mp.apply(layer, name)


class MagnitudePruningUnstructured(prune.BasePruningMethod):
    threshold = 0
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        tmp = t.cpu().detach().numpy()
        mask = default_mask.cpu().detach().numpy()

        new_mask = mask.copy()
        new_mask[abs(tmp) < self.threshold] = 0
        return torch.tensor(new_mask, dtype=torch.float32)


class MyStructured:
    def __init__(self, threshold, model):
        self.percentage = threshold

        self.model = model
        self.tuples = retrieve_tuples(model)
        self.count = len(self.tuples)-1

    def compute_new_masks(self):
        mp = RandomPruner()
        RandomPruner.percentage = self.percentage
        layer, name = self.tuples[self.count]
        mp.apply(layer, name)
        self.count -= 1


class MyPruner(prune.BasePruningMethod):

    PRUNING_TYPE = 'unstructured'
    percentage = 20

    def compute_mask(self, t, default_mask):
        device = choose_device(default_mask)
        tmp = t.cpu().detach().numpy()
        mask = default_mask.cpu().detach().numpy()
        weights = tmp[np.nonzero(mask)]
        weights = np.absolute(weights.flatten())
        threshold = np.percentile(
            weights, MagnitudePruningStructured.percentage)
        new_mask = mask.copy()
        new_mask[abs(tmp) < threshold] = 0
        return torch.tensor(new_mask, dtype=torch.float32).to(device)


class RandomStructured:
    def __init__(self, threshold, model):
        self.percentage = threshold
        self.model = model
        self.tuples = retrieve_tuples(model, False)

    def compute_new_masks(self):
        RandomPruner.percentage = self.percentage
        mp = RandomPruner()
        for layer, name in self.tuples:
            mp.apply(layer, name)


class RandomPruner(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'
    percentage = 20

    def compute_mask(self, t, default_mask):
        tmp = self.percentage/100
        mask = default_mask.cpu().detach().numpy()
        new_mask = mask.copy()
        device = choose_device(default_mask)

        tmp_list = np.argwhere(mask == 1)

        how_many = round(len(tmp_list)*tmp)

        indexes = np.random.choice(
            range(len(tmp_list)), size=how_many,  replace=False)

        for index in indexes:
            new_mask[tuple(tmp_list[index].T)] = 0

        return torch.tensor(new_mask, dtype=torch.float32).to(device)


class ImpactStructured:
    def __init__(self, threshold, model):
        self.percentage = threshold

        self.model = model
        self.tuples = retrieve_tuples(model, False)

    def compute_new_masks(self):
        mp = ImpactPruner()
        ImpactPruner.percentage = self.percentage
        for layer, name in self.tuples:

            tmpname = name
            if hasattr(layer, name+"_orig"):
                tmpname = name+"_orig"
            t = getattr(layer, tmpname)
            tmp = t.grad.cpu().detach().numpy()
            weights = t.cpu().detach().numpy()
            importance_score = torch.tensor(weights*tmp).to(choose_device(t))

            mp.apply(layer, name, importance_scores=importance_score)


class ImpactPruner(prune.BasePruningMethod):

    PRUNING_TYPE = 'unstructured'
    grads = None
    percentage = 20

    def compute_mask(self, t, default_mask):
        device = choose_device(default_mask)
        tmp = t.cpu()
        mask = default_mask.cpu().detach().numpy()
        weights = tmp[np.nonzero(mask)]
        weights = np.absolute(weights.flatten())
        threshold = np.percentile(
            weights, ImpactPruner.percentage)
        new_mask = mask.copy()
        new_mask[abs(tmp) < threshold] = 0
        return torch.tensor(new_mask, dtype=torch.float32).to(device)


class RandomPrunerOld(prune.BasePruningMethod):

    PRUNING_TYPE = 'unstructured'
    percentage = 20

    def compute_mask(self, t, default_mask):
        device = choose_device(default_mask)
        tmp = self.percentage/100
        mask = default_mask.cpu().detach().numpy()

        new_mask = mask.copy()
        tmp_mask = np.random.choice(
            [0, 1], size=new_mask.shape, p=[tmp, 1-tmp])
        new_mask = new_mask*tmp_mask

        return torch.tensor(new_mask, dtype=torch.float32).to(device)
