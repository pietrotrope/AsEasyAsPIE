import torch.nn as nn
from torch import optim
from pipeline import *
from datasets_folder.dataset import *
from architectures.languageModel import *
from pruning.IMP import *
from pruning.IBP import *
from pruning.nopruning import *
from pruning.random import *
import json
from math import isclose


def find_threshold(n_prunings, final_percentage):
    if n_prunings == 1:
        return final_percentage*100
    x = 1
    goal = 1 - final_percentage
    res = 0

    minf = 1
    maxf = 0

    while not isclose(res, goal, abs_tol=10**-5):
        if res < goal:
            x = max(x*(0.9), minf*0.9999)
        else:
            if x*1.5 <= 1:
                x = x*1.5
            else:
                x = x+(1-x)/2
            x = min(x, maxf*1.0001)

        maxf = max(x, maxf)
        minf = min(x, minf)

        tmp = 1
        for _ in range(n_prunings):
            tmp = tmp-tmp*x
        res = tmp
    return round(x, 4)*100


def get_config(specs):
    f = open('configurations.json')
    conf = json.load(f)
    f.close()
    tmp = conf[specs["dataset"]][specs["architecture"]]
    print(tmp)
    return tmp["batch_size"], tmp["epochs"], tmp["lr"]


def json_to_experiment_data(specs, device="cuda:0"):
    print("Start formatting")
    batch_size, epochs, lr = get_config(specs)
    num_classes = 2
    model_info, training_params = {}, {}
    pruner_info = {"frequency": [epochs, 2*epochs, 3*epochs]}
    multilabel = False

    seeds = list(range(20))
    if "seeds" in specs:
        seeds = list(range(int(specs["seeds"])))

    ### DATASET ###

    dataset_info = {
        "training_set": True,
        "test_set": True,
        "validation_set": True,
        "model_architecture": specs["architecture"],
        "batch_size": batch_size,
        "noise": int(specs["noise"])
    }

    if specs["dataset"] == "SNLI":
        dataset_info.update({
            "dataset": DatasetNLI2,
            "path": "",
            "labels": [("entailment", 0), ("neutral", 1), ("contradiction", 2)],
            "vec_max_len": 128,
        })
        num_classes = 3
    elif specs["dataset"] == "IMDB":
        dataset_info.update({
            "dataset": DatasetIMDB2,
            "path": "",
            "labels": [("pos", 1), ("neg", 0)],
            "vec_max_len": 512,
        })
        num_classes = 2

    elif specs["dataset"] == "Reuters":
        lines = []
        v = "0"
        with open('./data/Reuters/reuters_labels_'+v+'.txt') as f:
            lines = f.read().splitlines()
        labels = [(lines[i], i) for i in range(len(lines))]

        dataset_info.update({
            "dataset": Reuters,
            "path": v,
            "labels": labels,
            "vec_max_len": 256,
        })
        num_classes = len(dataset_info["labels"])
        multilabel = True

    elif specs["dataset"] == "AAPD":
        dataset_info.update({
            "dataset": AAPD,
            "path": "",
            "labels": [(0, 0)]*54,
            "vec_max_len": 256,
        })
        num_classes = len(dataset_info["labels"])
        multilabel = True

    ### MODEL ARCHITECTURE ###

    if specs["architecture"] == "BiLSTM":

        model_info = {
            "model": BiLSTMClassifier,
            "embed_dim": 300,
            "num_class": num_classes,
            "batch_size": batch_size,
            "hidden_dim": 100,
            "dropout_prob": 0.0,
            "device": device,
            "num_layers": 2
        }
    elif specs["architecture"] == "BERT":
        model_info = {
            "model": BertClassifier,
            "n_classes": num_classes,
            "device": device,
            "max_input_length": dataset_info["vec_max_len"]
        }

    ### PRUNING ALGORITHM ###

    if "ini" in specs["pruner"]:
        pruner_info["frequency"] = [0]
    else:
        if "NoPruner" != specs["pruner"]:
            epochs = (len(pruner_info["frequency"])+1)*epochs

    # Magnitude based pruners
    if specs["pruner"] == "IMP-WR":
        pruner_info.update({"pruner": NewIMPWRstructured})
    elif specs["pruner"] == "IMP_ini":
        pruner_info.update({"pruner": NewIMPstructured})
    elif specs["pruner"] == "IMP":
        pruner_info.update({"pruner": NewIMPstructured})

    # Random based pruners
    elif specs["pruner"] == "Random-WR":
        pruner_info.update({"pruner": RandomPruningWRStructured})
    elif specs["pruner"] == "Random_ini":
        pruner_info.update({"pruner": RandomPruningStructured})
    elif specs["pruner"] == "Random":
        pruner_info.update({"pruner": RandomPruningStructured})

    # Gradient based pruners
    elif specs["pruner"] == "IBP-WR":
        pruner_info.update({"pruner": ImpactPruningWRStructured})
    elif specs["pruner"] == "IBP_ini":
        pruner_info.update({"pruner": ImpactPruningStructured})
    elif specs["pruner"] == "IBP":
        pruner_info.update({"pruner": ImpactPruningStructured})

    else:
        pruner_info = None

    if pruner_info:
        pruner_info["removal_percentage"] = int(specs["pruning_threshold"])
        pruner_info["percentage"] = find_threshold(
            len(pruner_info["frequency"]), pruner_info["removal_percentage"]/100)

    ### TRAINING PARAMETERS ###

    training_params = {
        "epochs": epochs,
        "learning_rate": lr,
        "loss_fn": nn.CrossEntropyLoss().to(device),
        "optimizer": optim.Adam,
        "device": device,
        "track_stability": True,
        "architecture": specs["architecture"]

    }

    if multilabel:
        training_params["loss_fn"] = nn.BCEWithLogitsLoss().to(device)

    print("Finished formatting")
    return (dataset_info, model_info, pruner_info, training_params, seeds)
