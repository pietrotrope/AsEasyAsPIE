from training.training import train_model
from architectures.tools import GloveExtended as GloVe
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm
from pruning.nopruning import NoPruning
import wandb
import torch
import random
import numpy as np


def retrieve_dataset(args):
    training_set, test_set, tr, te = None, None, None, None

    if args["training_set"]:
        training_set = args["dataset"](
            args["path"], args["labels"], setting=args["model_architecture"])

        training_set.load_set(
            "train", vec_max_len=args["vec_max_len"])
        print("Training set")

    if args["test_set"]:
        test_set = args["dataset"](
            args["path"], args["labels"], setting=args["model_architecture"])

        test_set.load_set("test", vec_max_len=args["vec_max_len"])
        print("Test set")

    elif args["validation_set"]:
        test_set = args["dataset"](
            args["path"], args["labels"], setting=args["model_architecture"])

        test_set.load_set("validation", vec_max_len=args["vec_max_len"])
        print("Validation set")

    if args["model_architecture"] == "BiLSTM":
        global_vectors = GloVe(name='840B', dim=300,
                               unk_init=lambda x: torch.ones(x.shape))

        max_len = args["vec_max_len"]

        def vectorize_batch(batch):
            X, Y, idxs = list(zip(*batch))
            X = [tokens+[""] * (max_len-len(tokens)) if len(tokens)
                 < max_len else tokens[:max_len] for tokens in X]
            X_tensor = torch.zeros(len(batch), max_len, 300)
            for i, tokens in enumerate(X):
                X_tensor[i] = global_vectors.get_vecs_by_tokens(tokens)
            return X_tensor.squeeze(1), torch.tensor(Y).float(), torch.tensor(idxs)

        tr = DataLoader(
            dataset=training_set, batch_size=args["batch_size"], pin_memory=True, num_workers=4, shuffle=True, collate_fn=vectorize_batch)

        tr_100 = DataLoader(
            dataset=training_set, batch_size=10, pin_memory=True, num_workers=4, shuffle=True, collate_fn=vectorize_batch)

        te = DataLoader(
            dataset=test_set, batch_size=args["batch_size"], pin_memory=True, num_workers=4, collate_fn=vectorize_batch)
    else:

        tr = DataLoader(
            dataset=training_set, batch_size=args["batch_size"], pin_memory=True, num_workers=4, shuffle=True)

        tr_100 = DataLoader(
            dataset=training_set, batch_size=10, pin_memory=True, num_workers=4, shuffle=True)

        te = DataLoader(
            dataset=test_set, batch_size=args["batch_size"], pin_memory=True, num_workers=4)

    return training_set, test_set, tr, te, tr_100


def load_model(args):
    return args["model"](args)


def load_pruner(args):
    tmp = args.copy()
    del tmp["pruner"]
    del tmp["removal_percentage"]
    return args["pruner"](**tmp)


def perform_training(dataset, model, pruner, training_params, experimental=False, name="test", verbose=False):
    _ = train_model(model, training_params["epochs"], dataset[2], pruner,
                    training_params["loss_fn"], training_params["optimizer"],
                    verbose, training_params["device"], dataset[3],
                    training_params["track_stability"], training_params["architecture"],
                    experimental=experimental, name=name, ds=dataset, hyp_opt=False, log=True, lr=training_params["learning_rate"])

    del model
    torch.cuda.empty_cache()


def set_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def execute_pipeline(dataset_info, model_info, pruner_info, training_parameters, seeds=None, dataset=None, project="test", tags=[]):
    set_seeds(42)
    if seeds is None:
        seeds = list(range(20))

    if pruner_info is None:
        pruner_info = {"pruner": NoPruning,
                       "removal_percentage": 0}

    if dataset is None:
        dataset = retrieve_dataset(dataset_info)

    skip_to = -1

    for seed in tqdm(seeds):
        if seed > skip_to:

            set_seeds(seed)

            model = load_model(model_info)

            model.to(training_parameters["device"])

            pruner_info["model"] = model
            pruner = load_pruner(pruner_info)

            name = experiment_name("results/", dataset_info, model_info,
                                   pruner, str(pruner_info["removal_percentage"]), True)

            for i in range(seeds[-1]+1):
                if not os.path.exists(name+"/"+str(i)+"_measures.json"):
                    break
                else:
                    skip_to = i

            if seed > skip_to:
                # wandb init line removed

                perform_training(dataset, model, pruner,
                                 training_parameters, True, name+"/"+str(seed), True)
                wandb.finish(0)

            del pruner


def experiment_name(header, data_info, model_info, pruner, removal_percentage="", create_dir=False):
    out = header + data_info["dataset"].name + "_" + str(
        data_info["noise"]) + "_" + model_info["model"].name + "_" + pruner.name + "_" + removal_percentage
    if create_dir and not os.path.exists(out):
        os.makedirs(out)
    return out
