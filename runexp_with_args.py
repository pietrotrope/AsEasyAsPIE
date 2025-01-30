import wandb
from pipeline import *
from experiment_formatter import *
import json
import wandb
import sys

f = open("recipes/"+sys.argv[1]+'.json')

recipes = json.load(f)
f.close()

wandb.login(key="key")
print("Start")
ds = None
for key in recipes:
    print("Formatting: ")
    dataset_info, model_info, pruner_info, training_params, seeds = json_to_experiment_data(
        recipes[key])
    dataset_info["validation_set"] = False
    print("seeds count: "+str(seeds))
    print("Run: ")
    execute_pipeline(dataset_info, model_info, pruner_info,
                     training_params, seeds, project="project_name", tags=["tags"])
