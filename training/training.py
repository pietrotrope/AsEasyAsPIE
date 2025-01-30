import torch
import torch.nn as nn
from pruning.tools import compute_compression
from measures.utils import pred_to_file
import json
import numpy as np
import wandb
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
import warnings
warnings.filterwarnings("ignore")


def compute_batch(model, x_batch, architecture="BiLSTM", device="cpu"):
    if architecture == "BiLSTM":
        model.hidden = model.init_hidden(len(x_batch))
        input = x_batch.to(device)

        predictions = model(input)

    elif architecture == "BERT":
        input_id = x_batch['input_ids'].to(device)
        mask = x_batch['attention_mask'].to(device)
        ttid = x_batch['token_type_ids'].to(device)

        predictions = model(input_id, mask, token_type_ids=ttid)

    return predictions


def sample_100(data_loader):
    sampled = []
    for batch in data_loader:
        if len(sampled) < 10:
            sampled.append(batch)
        else:
            break
    return sampled


def perform_epoch(model, data_loader, loss_fn, optimizer=None, device="cuda:0", architecture="BiLSTM", save=False, name="", multilabel=False, label_type=torch.LongTensor, hyp_opt=False, optimizer_step=True, best_threshold=None):

    f1 = {"micro": [], "macro": []}
    precision = {"micro": [], "macro": []}
    recall = {"micro": [], "macro": []}
    accuracy = 0
    cumulated_loss = 0
    guessed = 0
    tot_tested = 0
    final_preds = None
    final_correct = None
    indexes = []

    if optimizer is not None:
        model.train()
    else:
        model.eval()

    for _, batch in enumerate(data_loader):
        (x_batch, y_batch, idxs) = batch
        indexes += [t.item() for t in idxs]

        labels = y_batch.type(label_type).to(device)
        predictions = compute_batch(model, x_batch, architecture, device)

        if multilabel or (save and not hyp_opt):
            if final_preds is None:
                final_preds = predictions.detach()
                final_correct = labels.detach()
            else:
                final_preds = torch.cat((final_preds, predictions.detach()), 0)
                final_correct = torch.cat((final_correct, labels.detach()), 0)

        loss = loss_fn(predictions, labels)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()

            if optimizer_step:
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), 1)

        with torch.no_grad():

            if not multilabel:
                _, predicted = predictions.max(dim=1)
                guessed += (predicted == labels).sum().item()

            tot_tested += labels.size(0)
            cumulated_loss += loss.item()

    if multilabel:

        labels = final_correct.tolist()
        tmp_preds = final_preds
        m = nn.Sigmoid()
        tmp_preds = m(tmp_preds)

        if best_threshold is None:
            how_many_steps = 100

            best = (0, -1)
            for i in range(how_many_steps):
                preds = (tmp_preds > i/how_many_steps).long()
                preds = preds.tolist()
                tmp = f1_score(y_true=labels, y_pred=preds, average="macro")
                if tmp > best[1]:
                    best = (i/how_many_steps, tmp)

            best_threshold = best[0]

        predictions = (tmp_preds > best_threshold).long()
        predictions = predictions.tolist()

        accuracy = accuracy_score(y_true=labels, y_pred=predictions)

        pmic, rmic, fmic, _ = precision_recall_fscore_support(
            y_true=labels, y_pred=predictions, average="micro")
        pmac, rmac, fmac, _ = precision_recall_fscore_support(
            y_true=labels, y_pred=predictions, average="macro")
        precision["micro"].append(pmic)
        precision["macro"].append(pmac)
        f1["micro"].append(fmic)
        f1["macro"].append(fmac)
        recall["micro"].append(rmic)
        recall["macro"].append(rmac)

        print("Best threshold: "+str(best_threshold))

    if save and not hyp_opt:
        pred_to_file(final_preds, name)

    return tot_tested, cumulated_loss, guessed, (f1, precision, recall, accuracy, multilabel, best_threshold), indexes


def print_verbose(epoch, results, num=0, multilabel=False):
    s = "Training" if num == 0 else "Test"
    if multilabel:
        print(s+" set: \nEpoch "+str(epoch+1) +
              ", Average loss on epoch: " +
              str(round(results["loss"][num][-1], 3)) +
              " Accuracy: "+str(round(results["accuracy"][num][-1], 3)) +

              "\nMacro: F1: "+str(round(results["f1"]["macro"][num][-1], 3)) +
              " Precision: "+str(round(results["precision"]["macro"][num][-1], 3)) +
              " Recall: "+str(round(results["recall"]["macro"][num][-1], 3)) +

              "\nMicro: F1: "+str(round(results["f1"]["micro"][num][-1], 3)) +
              " Precision: "+str(round(results["precision"]["micro"][num][-1], 3)) +
              " Recall: "+str(round(results["recall"]["micro"][num][-1], 3)) +
              "\n\n")
    else:
        print(s+" set: \nEpoch "+str(epoch+1) +
              ", average loss on epoch: " +
              str(round(results["loss"][num][-1], 3)) +
              "\nAccuracy: "+str(round(results["accuracy"][num][-1], 3)) +
              "\n\n")


def update_results(results, tracked, cumulated_loss, guessed, tot_tested, data_loader, num=0, log=False):
    f1, precision, recall, accuracy, multilabel, threshold = tracked
    results["loss"][num].append(cumulated_loss/len(data_loader))

    if multilabel:
        results["f1"]["micro"][num].append(f1["micro"][0])
        results["f1"]["macro"][num].append(f1["macro"][0])

        results["precision"]["micro"][num].append(precision["micro"][0])
        results["precision"]["macro"][num].append(precision["macro"][0])

        results["recall"]["micro"][num].append(recall["micro"][0])
        results["recall"]["macro"][num].append(recall["macro"][0])

        results["accuracy"][num].append(accuracy)
    else:
        results["accuracy"][num].append(guessed/tot_tested)
        results["f1"]["micro"][num].append(guessed/tot_tested)
        results["f1"]["macro"][num].append(guessed/tot_tested)

    if log and num != 0:
        if multilabel:
            results["threshold"].append(threshold)

            wandb.log({'test loss': cumulated_loss/len(data_loader),
                       'test accuracy': accuracy,
                       'conf threshold': threshold,

                       'test precision micro': precision["micro"][0],
                       'test precision macro': precision["macro"][0],

                       'test recall micro': recall["micro"][0],
                       'test recall macro': recall["macro"][0],

                       'test f1 micro': f1["micro"][0],
                       'test f1 macro': f1["macro"][0],


                       'train precision micro': results["precision"]["micro"][0][-1],
                       'train precision macro': results["precision"]["macro"][0][-1],

                       'train recall micro': results["recall"]["micro"][0][-1],
                       'train recall macro': results["recall"]["macro"][0][-1],

                       'train f1 micro': results["f1"]["micro"][0][-1],
                       'train f1 macro': results["f1"]["macro"][0][-1],


                       'train loss': results["loss"][0][-1],
                       'train accuracy': results["accuracy"][0][-1],
                       "epoch": len(results["accuracy"][0])})
        else:
            wandb.log({'test loss': cumulated_loss/len(data_loader),
                       'test accuracy': guessed/tot_tested,
                       'train loss': results["loss"][0][-1],
                       'train accuracy': results["accuracy"][0][-1],
                       "epoch": len(results["accuracy"][0])})


def train_model(model, epochs, train_data_loader, pruner=None, loss_fn=None, optimizer=None, verbose=False, device="cpu", test_data_loader=False, track_stability=False, architecture="BiLSTM", experimental=False, name="test", ds=None, hyp_opt=False, log=False, lr=None):
    multilabel = ds[0].multilabel

    sample_loader = ds[4]

    optimizerT = optimizer

    if optimizerT is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optimizerT(model.parameters(), lr=lr)

    print("Start training")

    label_type = torch.LongTensor
    if loss_fn.__class__ is nn.BCEWithLogitsLoss:
        label_type = torch.float

    results = {"accuracy": [[], []],
               "f1": {"micro": [[], []], "macro": [[], []]},
               "precision": {"micro": [[], []], "macro": [[], []]},
               "recall": {"micro": [[], []], "macro": [[], []]},
               "loss": [[], []],
               "stability": [],
               "efficiency": [],
               "zeroed": [],
               "threshold": [],
               "train_indexes": {}}

    if pruner:
        sampled_examples = None
        if pruner.sampling and pruner.frequency.next_check == 0:

            sampled_examples = sample_100(sample_loader)

            _, _, _, _, _ = perform_epoch(
                model, sampled_examples, loss_fn, optimizer, device, architecture,
                False, "", multilabel, label_type, False, optimizer_step=False)

        with torch.no_grad():
            if pruner.check_and_prune(0):
                zeroed, cr = compute_compression(pruner.scoring.tuples)
                results["efficiency"].append(cr)
                results["zeroed"].append(zeroed)

                if "WR" in pruner.name:
                    optimizer = optimizerT(model.parameters(), lr=lr)

    for epoch in range(epochs):
        best_threshold = None
        savet = False
        namet = ""
        tot_tested, cumulated_loss, guessed, tracked, _ = perform_epoch(model, train_data_loader, loss_fn, optimizer,
                                                                        device, architecture, savet, namet, multilabel,
                                                                        label_type, hyp_opt, best_threshold=best_threshold)

        if multilabel:
            best_threshold = tracked[5]

        update_results(results, tracked, cumulated_loss, guessed,
                       tot_tested, train_data_loader, 0, log=log)

        t1 = experimental and (epoch == 0 or epoch+1 == epochs or (pruner.name == "NoPruning" and (epoch ==
                               round(epochs*0.8) or epoch == round(epochs*0.6) or epoch == round(epochs*0.3))))
        if t1:
            with torch.no_grad():
                optimizer.zero_grad()
                _, _, _, _, idxs = perform_epoch(
                    model, train_data_loader, loss_fn, None, device, architecture,
                    True, name+"_epoch_"+str(epoch)+"_train", multilabel, label_type, hyp_opt, best_threshold=best_threshold)
                results["train_indexes"][epoch] = idxs

        if verbose:
            print_verbose(epoch, results, 0, multilabel)

        if test_data_loader:

            if hyp_opt is False:
                save = t1 or (hasattr(pruner, "frequency")
                              and pruner.frequency.next_check == epoch and experimental)

                # Compute test predictions
                with torch.no_grad():
                    tot_tested, cumulated_loss, guessed, tracked, _ = perform_epoch(
                        model, test_data_loader, loss_fn, None, device, architecture,
                        save, name+"_epoch_"+str(epoch), multilabel, label_type, hyp_opt, best_threshold=best_threshold)

                    update_results(results, tracked, cumulated_loss,
                                   guessed, tot_tested, test_data_loader, 1, log=log)

                if verbose:
                    print_verbose(epoch, results, 1, multilabel)

            else:
                if epoch+1 == epochs:

                    with torch.no_grad():
                        tot_tested, cumulated_loss, guessed, tracked, _ = perform_epoch(
                            model, test_data_loader, loss_fn, None, device, architecture,
                            True, name+"_epoch_"+str(epoch), multilabel, label_type, hyp_opt, best_threshold=best_threshold)

                    update_results(results, tracked, cumulated_loss,
                                   guessed, tot_tested, test_data_loader, 1, log=log)

        if pruner and epoch != 0:

            sampled_examples = None
            if pruner.sampling and pruner.frequency.next_check == epoch:

                sampled_examples = sample_100(sample_loader)

                _, _, _, _, _ = perform_epoch(
                    model, sampled_examples, loss_fn, optimizer, device, architecture,
                    False, "", multilabel, label_type, False, optimizer_step=False, best_threshold=best_threshold)

            with torch.no_grad():
                if pruner.check_and_prune(epoch):
                    if "WR" in pruner.name:
                        optimizer = optimizerT(model.parameters(), lr=lr)
                    optimizer.zero_grad()

                    # And save new compression level
                    zeroed, cr = compute_compression(pruner.scoring.tuples)
                    results["efficiency"].append(cr)
                    results["zeroed"].append(zeroed)

                    # If we need to track stability
                    if test_data_loader and track_stability and hyp_opt is False:

                        # Compute predictions on test set right after pruning
                        with torch.no_grad():
                            tot_tested, cumulated_loss, guessed, _, _ = perform_epoch(
                                model, test_data_loader, loss_fn, None, device, architecture,
                                True, name+"_stab_epoch_"+str(epoch), multilabel, label_type, hyp_opt, best_threshold=best_threshold)

                            # Compute stability
                            if multilabel:
                                stability = 0
                            else:
                                stability = (results["accuracy"][1][-1] -
                                             (guessed/tot_tested))/(results["accuracy"][1][-1]+1e-9)

                            # Update stability records
                            results["stability"].append(stability)

    # If experimental we want to save all of the info we tracked
    if experimental:
        with open(name+"_measures.json", "w") as outfile:
            json.dump(results, outfile)

    metrics = (results["stability"], "metric")
    if not multilabel:
        results["f1"]["micro"][1] = (results["accuracy"])
    return results["accuracy"], results["loss"], results["f1"], metrics
