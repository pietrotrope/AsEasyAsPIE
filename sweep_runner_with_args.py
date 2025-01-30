import wandb
from pipeline import *
from experiment_formatter import *
import json
import sys

from pruning.nopruning import NoPruning
from torch import optim


wandb.login(key="token")
random.seed(42)


def perform_training_r(dataset, model, pruner, training_params, experimental=False, name="test", verbose=True):
    tmp = training_params["optimizer"](
        model.parameters(), lr=training_params["learning_rate"])
    accuracies, losses, f1, _ = train_model(model, training_params["epochs"], dataset[2], pruner,
                                            training_params["loss_fn"], tmp,
                                            verbose, training_params["device"], dataset[3],
                                            training_params["track_stability"], training_params["architecture"],
                                            experimental=experimental, name=name, multilabel=dataset[0].multilabel, hyp_opt=True, log=True)
    return losses[1][-1], accuracies[1][-1], f1["macro"][1][-1]


f = open("recipes/"+sys.argv[1]+'.json')
recipes = json.load(f)
f.close()

k = None

ds = None
for key in recipes:
    k = key

dataset_info, model_info, pruner_info, training_params, seeds = json_to_experiment_data(
    recipes[k], "cuda:0")

dataset = retrieve_dataset(dataset_info)


def get_name(_):
    return "name"+str(random.random())


def objective(config):

    if "epochs" in config:
        config.training_params["epochs"] = config.epochs
    if "lr" in config:
        config.training_params["learning_rate"] = config.lr

    module = __import__("architectures.languageModel")
    mf = config.model_info.copy()
    tmp = config.model_info["model"].split(".")[-1]
    mf["model"] = getattr(getattr(module, "languageModel"), tmp)

    model = load_model(mf)
    model.to(config.training_params["device"])

    pruner_info = None
    if config.pruner_info is None:
        pruner_info = {"pruner": NoPruning,
                       "removal_percentage": 0}
    else:
        pruner_info = config.pruner_info.copy()
        tmp = pruner_info["pruner"].split(".")
        module = __import__(tmp[:-1].join("."))
        t = getattr(module, tmp[1])
        for w in tmp[2:]:
            t = getattr(t, w)
        pruner_info["pruner"] = t

    pruner_info["model"] = model

    pruner = load_pruner(pruner_info)

    training_params = config.training_params
    training_params["optimizer"] = optim.Adam

    if not dataset[0].multilabel:
        training_params["loss_fn"] = nn.CrossEntropyLoss().to(
            training_params["device"])
    else:
        training_params["loss_fn"] = nn.BCEWithLogitsLoss().to(
            training_params["device"])

    name = get_name(config)

    l, a, f1 = perform_training_r(
        dataset, model, pruner, training_params, False, name, True)

    return l, a, f1


config_defaults = {"dataset_info": dataset_info, "model_info": model_info,
                   "pruner_info": pruner_info, "training_params": training_params}


def main():
    wandb.init(project='project_name', name=get_name(
        wandb.config), config=config_defaults)
    loss, accuracy, f1 = objective(wandb.config)
    wandb.log({'loss': loss, 'accuracy': accuracy, 'f1': f1})


f = open("search_spaces.json")
search_spaces = json.load(f)
f.close()
sweep_configuration = search_spaces[dataset[0].name][model_info["model"].name]

sweep_id = wandb.sweep(
    sweep=sweep_configuration,
    project="project_name"
)

wandb.agent(sweep_id, function=main, count=100)
