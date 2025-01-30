import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import json


def pred_to_file(pred, name):
    arr = pred.data.detach().cpu().numpy()
    np.savetxt(name+'.csv', arr)


def file_to_pred(name, pickled=False):
    if pickled:
        return np.load(name+".npy", allow_pickle=True)
    return np.genfromtxt(name+".csv", delimiter=' ')


def retrieve_predictions(folder, name, how_many, epoch=None, pickled=True):
    out = []
    count = 0
    for i in range(how_many):
        myFile = file_to_pred(folder+str(i)+name, pickled)

        mname = folder+str(i)+"_measures.json"
        if "train" in name and os.path.isfile(mname):
            f = open(mname)
            data = json.load(f)
            if "train_indexes" in data:

                if epoch is None:
                    epoch = len(data["accuracy"][0])-1

                d = data["train_indexes"][str(epoch)]
                d = np.array(d)
                a = [0]*len(myFile)
                for ii, el in enumerate(d):
                    a[el] = ii
                myFile = myFile[a]
                count += 1

        out.append(myFile)
    return torch.tensor(np.array(out))


def plot_data(Xs, Ys, stds=None, title=None, legend=None, xlabel="x",
              ylabel="y", style="-", name=None, legend_space=1, xticks=None, yticks=None, xy_ranges=None, colors=None):
    fig, ax = plt.subplots()
    if title:
        fig.suptitle(title)

    if type(style) != list:
        style = [style] * len(Xs)

    plots = []

    if stds is None:
        for i in range(len(Xs)):
            p, = ax.plot(Xs[i], Ys[i], style[i])
            plots.append(p)
    else:
        stds = np.array(stds)
        Ys = np.array(Ys)
        for i in range(len(Xs)):

            if colors:
                p, = ax.plot(Xs[i], Ys[i], style[i], color=colors[legend[i]])
                plots.append(p)

                ax.fill_between(Xs[i], Ys[i]-stds[i], Ys[i] +
                                stds[i], alpha=0.3, color=colors[legend[i]])
            else:
                p, = ax.plot(Xs[i], Ys[i], style[i])
                plots.append(p)

                ax.fill_between(Xs[i], Ys[i]-stds[i], Ys[i]+stds[i], alpha=0.3)

    if legend is not None:
        box = ax.get_position()
        if legend_space == 1:
            ax.legend(plots, legend)
        else:
            ax.set_position(
                [box.x0, box.y0, box.width * legend_space, box.height])
            ax.legend(plots, legend, loc='center left',
                      bbox_to_anchor=(1, 0.5))
    if xy_ranges:
        if xy_ranges[0] is not None:
            ax.set_xlim(xy_ranges[0])
        if xy_ranges[1] is not None:
            ax.set_ylim(xy_ranges[1])

    if xticks:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)
    if yticks:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if name != None:
        plt.savefig(name, dpi=600)
        plt.close()
    else:
        plt.show()


def plot_data_tokens(Xs, Ys, stds=None, title=None, legend=None, xlabel="x",
                     ylabel="y", style="-", name=None, legend_space=1, xticks=None, yticks=None, xy_ranges=None, colors=None):
    fig, ax = plt.subplots()
    if title:
        fig.suptitle(title, fontsize=20)

    if type(style) != list:
        style = [style] * len(Xs)

    plots = []

    if stds is None:
        for i in range(len(Xs)-2):
            p, = ax.plot(Xs[i], Ys[i], style[i], linewidth=1, alpha=0.4)
            plots.append(p)
        p, = ax.plot(Xs[len(Xs)-2], Ys[len(Xs)-2],
                     style[len(Xs)-2], color='black', linewidth=3)
        p, = ax.plot(Xs[len(Xs)-1], Ys[len(Xs)-1],
                     style[len(Xs)-1], color='#00a7b0', linewidth=3)
        plots.append(p)

    else:
        stds = np.array(stds)
        Ys = np.array(Ys)
        for i in range(len(Xs)):

            if colors:
                p, = ax.plot(Xs[i], Ys[i], style[i],
                             color=colors[legend[i]], linewidth=2)
                plots.append(p)

                ax.fill_between(Xs[i], Ys[i]-stds[i], Ys[i] +
                                stds[i], alpha=0.3, color=colors[legend[i]])
            else:
                p, = ax.plot(Xs[i], Ys[i], style[i], linewidth=2)
                plots.append(p)

                ax.fill_between(Xs[i], Ys[i]-stds[i], Ys[i]+stds[i], alpha=0.3)

    if legend is not None:
        box = ax.get_position()
        if legend_space == 1:
            ax.legend(plots, legend)
        else:
            ax.set_position(
                [box.x0, box.y0, box.width * legend_space, box.height])
            ax.legend(plots, legend, loc='center left',
                      bbox_to_anchor=(1, 0.5))
    if xy_ranges:
        if xy_ranges[0] is not None:
            ax.set_xlim(xy_ranges[0])
        if xy_ranges[1] is not None:
            ax.set_ylim(xy_ranges[1])

    if xticks:
        ax.set_xticks(xticks)
        plt.xlim(20, 99)
        ax.set_xticklabels(xticks, fontsize=13)

    if yticks:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks, fontsize=13)
    ax.tick_params(axis='y', labelsize=13)

    ax.set_xlabel(xlabel, fontsize=17)
    ax.set_ylabel(ylabel, fontsize=17)

    if name != None:
        fig.tight_layout()
        plt.savefig(name, dpi=400)
        plt.close()
    else:
        plt.show()
