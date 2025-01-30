from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_recall_fscore_support
import numpy as np
import seaborn as sns


def stability(accuarcies, epochs):
    out = []
    for i in range(len(epochs)):
        out.append((accuarcies[i-1]-accuarcies[i])/accuarcies[i-1])

    return out


def mean_stability(accuracies, epochs):
    tmp = np.array(stability(accuracies, epochs))
    return np.mean(tmp)


def generalization_gap(losses):
    out = []
    for i in range(len(losses[0])):
        out.append(losses[0][i]-losses[1][i])
    return out


def mean_generalization_gap(losses):
    tmp = np.array(generalization_gap(losses))
    return np.mean(tmp)


def plot_ie_data(preds_old, preds_dense, preds, labels, n_groups=20, softmax=False, sub_after=False, name='', acc=False):
    score, std = influential_examples(preds_old, preds_dense, preds, labels,
                                      number_of_groups=n_groups, acc=acc, softmax=softmax, sub_after=sub_after)
    data = {'percentile':  [i+1 for i in range(n_groups)],
            'res': score,
            'std': std
            }

    x = data["percentile"]
    y = data["res"]
    yerr = data["std"]
    width = 10
    height = 8

    plt.figure(figsize=(width, height))
    plt.bar(x, y, color='blue')
    plt.xlabel('Subgroup')
    plt.ylabel('Delta error')
    plt.errorbar(x, y, yerr, fmt='.', color='Black')
    plt.title(" ".join(name.split("_")))
    plt.savefig('Images/'+name+'.png', dpi=400,
                transparent=True, pad_inches=0.02)


def plot_ie_data_multiple(preds_old, preds_dense, preds_multiple, labels, n_groups=20, softmax=False, sub_after=False, title="", name='', legend=[]):
    width = 10
    height = 8

    colors = {"20": '#ff595e',
              "50": '#ffca3a',
              "70": '#8ac926',
              "90": '#1982c4',
              "99": '#6a4c93'}

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.title(title)
    plt.xlabel('Bin')
    plt.ylabel("% Error ")
    plt.xticks([1, 5, 10, 15, 20])
    plt.figure(figsize=(width, height))

    x = [i+1 for i in range(n_groups)]

    preds_multiple.reverse()

    min_v = 1000
    max_v = -1000
    max_std = 0

    for (preds, pt) in preds_multiple:
        score, std, percs, std_perc = influential_examples(
            preds_old, preds_dense, preds, labels, number_of_groups=n_groups, softmax=softmax, sub_after=sub_after, return_losses=True)
        score = percs
        std = std_perc

        score = np.array(score)
        std = np.array(std)

        ax.plot(x, [1]*len(x), ":", color="black")

        ax.step(x, score*100, alpha=0.7, color=colors[pt], where='mid')
        ax.errorbar(x, score*100, std*100, fmt='.', color='Black')

        min_v = min(min_v, min(score)*100)
        max_v = max(max_v, max(score)*100)
        max_std = max(max_std, max(std)*100)

    ax.set_ylim(min_v-max_std, max_v+max_std)

    fig.savefig("image_name", dpi=400, transparent=True, pad_inches=0.02)


def el2n_seq(predictions, correct, multilabel=False):
    if multilabel:
        return np.mean(torch.linalg.norm(predictions - correct, ord=2, dim=2).numpy(), axis=0)
    return np.mean(torch.linalg.norm(predictions - F.one_hot(correct.to(torch.int64)), ord=2, dim=2).numpy(), axis=0)


def get_indexes(preds1, labels, number_of_groups=20, multilabel=False):
    if not multilabel:
        m = nn.Softmax(dim=2)
        el2n = el2n_seq(m(preds1), labels)
    else:
        m = nn.Sigmoid()
        el2n = el2n_seq(m(preds1), labels, True)

    rank = np.argsort(el2n)

    sorted_scores = el2n[rank]
    percentiles = []
    for i in range(1, number_of_groups+1):
        percentiles.append(np.percentile(
            sorted_scores, i*(100/number_of_groups)))

    indexes = [[]]

    count = 0
    for i in range(len(sorted_scores)):
        if sorted_scores[i] > percentiles[count]:
            count += 1
            indexes.append([])

        indexes[count].append(i)

    return rank, indexes


def influential_examples(preds1, preds_dense, preds_sparse, labels, number_of_groups=20, softmax=False, sub_after=False, return_losses=False, acc=False):
    multilabel = preds_sparse.shape[2] > 4
    if not multilabel:
        m = nn.Softmax(dim=2)
        el2n = el2n_seq(m(preds1), labels)
    else:
        m = nn.Sigmoid()
        el2n = el2n_seq(m(preds1), labels, True)

    rank = np.argsort(el2n)

    if softmax:
        preds_sparse = m(preds_sparse)
        preds_dense = m(preds_dense)

    sorted_scores = el2n[rank]
    percentiles = []
    for i in range(1, number_of_groups+1):
        percentiles.append(np.percentile(
            sorted_scores, i*(100/number_of_groups)))

    indexes = [[]]

    count = 0
    for i in range(len(sorted_scores)):
        if sorted_scores[i] > percentiles[count]:
            count += 1
            indexes.append([])

        indexes[count].append(i)

    if multilabel:
        loss = nn.BCEWithLogitsLoss(reduction='none')
        labels = labels.type(torch.float)
    else:
        loss = nn.CrossEntropyLoss(reduction='none')

    std = None
    if sub_after:
        o1 = []
        percs = []
        for i in range(len(preds1)):
            if acc:
                losdd_diff, losdd = None, None
                if multilabel:
                    losdd_diff = f1_score(
                        y_true=labels, y_pred=preds_sparse, average="macro")
                    losdd_diff -= f1_score(y_true=labels,
                                           y_pred=preds_dense, average="macro")

                else:
                    losdd_diff = []
                    for i in range(len(preds_sparse)):
                        _, predicted = preds_sparse[i].max(dim=1)
                        losdd_difft = (
                            predicted == labels).sum().item()/len(labels)
                        _, predicted = preds_dense[i].max(dim=1)
                        losdd_difft -= (predicted ==
                                        labels).sum().item()/len(labels)
                        losdd_diff.append(losdd_difft)

                    losdd_diff = np.mean(losdd_diff)

                losdd = losdd_diff
            else:
                losdd_diff = loss(preds_sparse[i], labels).numpy(
                ) - loss(preds_dense[i], labels).numpy()

                losdd = loss(preds_sparse[i], labels).numpy(
                ) / loss(preds_dense[i], labels).numpy()
                if multilabel:
                    losdd_diff = losdd_diff.mean(axis=1)

                    losdd = losdd.mean(axis=1)

            percs.append(losdd)
            o1.append(losdd_diff)

        percs = np.array(percs)
        out2 = percs.mean(axis=0)
        out2 = out2[rank]

        out2_stds = percs.std(axis=0)
        out2_stds = out2_stds[rank]

        o1 = np.array(o1)
        out1 = o1.mean(axis=0)
        out1 = out1[rank]

        avgs = []
        std = []

        perc = []
        perc_std = []

        for indexe in indexes:
            avgs.append(out1[indexe].mean())
            std.append(out1[indexe].std(ddof=1) /
                       np.sqrt(np.size(out1[indexe])))

            perc.append(out2[indexe].mean())
            perc_std.append(out2_stds[indexe].mean() /
                            np.sqrt(np.size(out2[indexe])))
    else:
        o1 = []
        o2 = []

        for i in range(len(preds1)):
            o1.append(loss(preds_sparse[i], labels).numpy())
            o2.append(loss(preds_dense[i], labels).numpy())
        o1 = np.array(o1)
        o2 = np.array(o2)
        out1 = o1.mean(axis=0)
        out2 = o2.mean(axis=0)
        out1 = out1[rank]
        out2 = out2[rank]

        avgs = []
        std = []
        for indexe in indexes:
            tmp = out1[indexe]-out2[indexe]
            avgs.append(tmp.mean())
            std.append(tmp.std(ddof=1) / np.sqrt(np.size(tmp)))

    if return_losses:
        return list(avgs), std, list(perc), perc_std

    return list(avgs), std


def retrieve_mask(l1, l2):
    tmp = ((l1-l2) != 0).numpy()
    mask = 1 - tmp
    return mask


def apply_sig(preds):
    m = nn.Sigmoid()
    for i in range(len(preds)):
        preds[i] = m(preds[i])
    return preds


def retrieve_pies(dense_preds, pruned_preds, n_classes=2, thresholds=None, fast=True):
    if thresholds is None:
        if fast:
            return dense_preds != pruned_preds

        tmp = torch.argmax(dense_preds, dim=2)
        tmp2 = torch.argmax(pruned_preds, dim=2)

        tmp = F.one_hot(tmp.to(torch.int64), num_classes=n_classes).numpy()
        tmp2 = F.one_hot(tmp2.to(torch.int64), num_classes=n_classes).numpy()

        tmp = np.sum(tmp, axis=0)
        tmp2 = np.sum(tmp2, axis=0)

        c1 = np.argmax(tmp, axis=1)
        c2 = np.argmax(tmp2, axis=1)
        return c1 != c2
    else:

        out = []

        dense_preds = dense_preds.numpy()
        pruned_preds = pruned_preds.numpy()

        # From confidence to prediction (for both dense and pruned)
        for i in range(len(thresholds[0])):
            dense_preds[i] = (dense_preds[i] > thresholds[0][i]).astype(int)
            pruned_preds[i] = (pruned_preds[i] > thresholds[1][i]).astype(int)

        # get threshold to exceed to have majority of agreement
        t = dense_preds.shape[0]/2

        # Sum over different initializations
        dense_sums = np.sum(dense_preds, axis=0)
        pruned_sums = np.sum(pruned_preds, axis=0)

        # Consider as majority only the classes where at least half of the models agree
        dense_sums = (dense_sums > t).astype(int)
        pruned_sums = (pruned_sums > t).astype(int)

        # for each column (class) check if dense and pruned agree, if not it is a pie
        for i in range(pruned_sums.shape[1]):
            tmp = dense_sums[:, i]-pruned_sums[:, i]
            out.append(np.nonzero(tmp)[0].tolist())
        return out


def get_distinct_pies(pies):
    distinct = set()

    for cl in pies:
        distinct = distinct.union(set(cl))
    distinct = list(distinct)
    return distinct


def compute_class_dist(pies, labels=None, n_classes=2, real=True, multilabel=False):
    if real:
        if not multilabel:
            classes = [0]*n_classes
            for ind in pies:
                classes[labels[ind]] += 1
            classes = np.array(classes)
            return classes, classes/sum(classes)
        else:

            distinct = get_distinct_pies(pies)

            classes = np.array([0]*len(pies))
            for ind in distinct:
                classes = classes+labels[ind].numpy()
            return classes, classes/sum(classes)

    else:
        if not multilabel:
            classes = np.array([0]*n_classes)
            for ind in pies:
                tmp = torch.argmax(labels[ind], dim=1)
                classes += F.one_hot(tmp.to(torch.int64),
                                     num_classes=n_classes).numpy()

            return classes, classes/sum(classes)

        else:
            dist = np.array([len(x) for x in pies])
            return dist, dist/sum(dist)


def retrieve_pred(preds, n_classes):
    tmp = torch.argmax(preds, dim=2)
    tmp = F.one_hot(tmp.to(torch.int64), num_classes=n_classes).numpy()
    tmp = np.sum(tmp, axis=0)
    c1 = np.argmax(tmp, axis=1)
    return c1


def retrieve_preds(dense_preds, pruned_preds, n_classes):
    tmp = torch.argmax(dense_preds, dim=2)
    tmp2 = torch.argmax(pruned_preds, dim=2)
    tmp = F.one_hot(tmp.to(torch.int64), num_classes=n_classes).numpy()
    tmp2 = F.one_hot(tmp2.to(torch.int64), num_classes=n_classes).numpy()
    tmp = np.sum(tmp, axis=0)
    tmp2 = np.sum(tmp2, axis=0)

    count = 0
    for a in tmp2:
        if a[0] == 15:
            count += 1
    print(count)

    c1 = np.argmax(tmp, axis=1)
    c2 = np.argmax(tmp2, axis=1)
    return c1, c2


def accuracy_pies(preds, labels, pies, thresholds=None):
    if thresholds is None:
        preds = np.array(preds)
        labels = labels.numpy()
        tmp = preds == labels
        tmp = tmp[pies]
        return tmp.sum()/len(pies)
    else:
        pies = get_distinct_pies(pies)

        labels = labels.numpy()
        labels = labels[pies]

        predictions = preds[:, pies, :]

        f = []
        for i in range(len(predictions)):
            _, _, fmac, _ = precision_recall_fscore_support(
                y_true=labels, y_pred=predictions[i], average="macro")
            f.append(fmac)

        f = np.array(f)

        return f.mean()


def accuracy_class(preds, labels, multilabel=False, tr=None):
    num_c = preds.shape
    num_c = num_c[2]
    if not multilabel:

        preds = torch.argmax(preds, dim=2)

        preds = np.array(preds)
        labels = labels.numpy()
        tmp = preds == labels

        dis = np.array([0]*num_c)
        tot_e = np.array([0]*num_c)

        for i in range(len(labels)):
            dis[labels[i]] += np.sum(tmp[:, i])
            tot_e[labels[i]] += 30

        return dis/tot_e, tot_e

    else:

        labels = labels.numpy()

        preds = np.array(preds)
        ac = np.array(preds, copy=True)
        m = nn.Sigmoid()

        f1_scores = []
        for i in range(len(preds)):
            preds[i] = (np.array(m(torch.tensor(preds[i]))) > tr[i])
            ac[i] = preds[i]*labels
            f1_scores.append(
                f1_score(y_true=labels, y_pred=preds[i], average=None))

        tot_app = np.array([0]*num_c)
        for i in range(num_c):
            tot_app[i] = sum(labels[:, i])

        return np.array(f1_scores).mean(axis=0), tot_app


def plot_anal_5(preds_multiple, labels, name='', legend=[]):
    width = 10
    height = 8

    colors = {"0": "#000000",
              "20": '#ff595e',
              "50": '#ffca3a',
              "70": '#8ac926',
              "90": '#1982c4',
              "99": '#6a4c93'}

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.title(" ".join(name.split("_")))

    plt.xlabel('Class')
    plt.ylabel('Accuracy/TP')

    plt.figure(figsize=(width, height))

    x = [i for i in range(len(preds_multiple[0][0]))]

    order = np.argsort(-preds_multiple[0][1])

    ax.bar(x, preds_multiple[0][1][order])
    for i, preds in enumerate(preds_multiple):

        score = np.array(preds[0])
        ax.plot(x, score[order], alpha=0.7, marker='v',
                linestyle='None', color=colors[legend[i]])

    labels = list(colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label])
               for label in labels]
    fig.legend(handles, labels)

    fig.savefig('Images/anal5/'+name+'.png', dpi=400,
                transparent=True, pad_inches=0.02)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.title(" ".join(name.split("_")))

    plt.xlabel('Class')
    plt.ylabel('Accuracy/TP')

    plt.figure(figsize=(width, height))

    x = [i for i in range(len(preds_multiple[0][0]))]
    for i, preds in enumerate(preds_multiple):

        score = np.array(preds[0])/np.array(preds[1])
        ax.plot(x, score[order], alpha=0.7, marker='v',
                linestyle='None', color=colors[legend[i]])

    labels = list(colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label])
               for label in labels]
    fig.legend(handles, labels)

    fig.savefig('Images/anal5/'+name+'_perc.png', dpi=400,
                transparent=True, pad_inches=0.02)


def plot_anal_5_2(preds_multiple, labels, name='', legend=[], title=""):

    sns.set()
    fig = plt.figure()

    x = [i for i in range(len(preds_multiple[0][0]))]
    x2 = x
    if len(preds_multiple[0][0]) >= 10:
        x2 = range(0, len(preds_multiple[0][0]), 5)

    ti = np.array([0, len(preds_multiple[0][0])-1])
    my_xticks = ['Most frequent\nclass', 'Least frequent\nclass']

    order = np.argsort(-preds_multiple[0][1])

    for i, preds in enumerate(preds_multiple):

        preds_multiple[i] = np.array(preds[0])
        preds_multiple[i] = preds_multiple[i][order]

    preds_multiple = np.array(preds_multiple)
    tmp = preds_multiple.mean(axis=1)
    for i in range(len(tmp)):
        tmp[i] = str(round(tmp[i], 3))

    ax = sns.heatmap(preds_multiple, vmin=0, vmax=1, xticklabels=x,
                     yticklabels=legend, cmap="RdYlGn", cbar=False)

    plt.xticks(rotation=0)

    ax.set_xticks(ti)
    ax.set_xticklabels(my_xticks)

    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('F1 score')

    ax2 = plt.twinx()
    ax2 = sns.heatmap(preds_multiple, vmin=0, vmax=1,
                      xticklabels=x, yticklabels=tmp, cmap="RdYlGn", cbar=False)
    ax2.set_xticks(ti)
    ax2.set_xticklabels(my_xticks)

    plt.tight_layout()
    plt.show()

    fig.savefig("image_name", dpi=400,
                transparent=True, pad_inches=0.02)
