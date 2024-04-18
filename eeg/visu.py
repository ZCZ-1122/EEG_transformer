import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import seaborn as sn
from pathlib import Path
import os


def plot_confusion_matrix(confusion,title='confusion matrix'):
    plt.figure(figsize=(8,6))
    group_counts = ['{0:0.0f}'.format(value) for value in confusion.flatten()]
    group_percentages = ['{0:.2%}'.format(value)
                        for value in confusion.flatten()/np.sum(confusion)]
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(confusion.shape)
    ax = sn.heatmap(confusion, annot_kws={"size": 10},
                    fmt='', cmap='Blues', annot=labels )
    plt.title(title)
    print('row is Ground truth, column is prediction')


def plot_signals_from_loader(loader, nb_show_per_cls=2, sharey=True):
    sample, label = next(iter(loader))
    sample = sample.detach().cpu().numpy().squeeze()
    label = label.detach().cpu().numpy()

    # randomly some signals of each class
    ixs = []
    for l in range(5):
        ixs.append(np.random.choice(np.where(label==l)[0], nb_show_per_cls))
    ixs = np.concatenate(ixs)
    
    fig, ax = plt.subplots((len(ixs)+1)//2, 2,
                           figsize=(5 * 2, 2 * (len(ixs)+1)//2), sharey=sharey)
    ax = ax.ravel()

    for a, ix in zip(ax, ixs):
        a.plot(sample[ix])
        a.set_title(f'label = {label[ix]}')
        a.grid()
    fig.tight_layout()
    plt.show()


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds,
                                       ax=None,figsize=(10,8)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(thresholds, precisions[:-1], "b--", label="Precision")
    ax.plot(thresholds, recalls[:-1], "g-", label="Recall")
    ax.grid()
    ax.legend()
    ax.set_xlabel('thresholds')
    ax.set_title('precision recall vs threshold')



def plot_precision_vs_recall(precisions, recalls,
                             ax=None,figsize=(10,8)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(recalls[:-1], precisions[:-1],'m-')
    ax.grid()
    ax.set_xlabel('recalls')
    ax.set_ylabel('precisions')
    ax.set_title('precision vs recall')


def plot_attributions(data_arr, importance_scores, preds, labels, ax=None,
                      figsize=(12,3), save=False, path=None, mrsize=60):
    set_fig = False
    if ax is None:
        fig, ax = plt.subplots(len(data_arr),1, figsize=(figsize[0],figsize[1] * len(data_arr)))
        set_fig = True
        
    ax = ax.ravel()

    for i, a in tqdm(enumerate(ax)):
        c = None
        if preds[i] != labels[i]:
            c = 'r'
        a.scatter(range(len(data_arr[i])), data_arr[i], 
                  s=mrsize * importance_scores[i], marker='o', color=c)
        a.plot(data_arr[i], '--', lw=0.5)
        a.set_title(f'pred = {preds[i]}, gt = {labels[i]}')
        a.grid()
    
    if set_fig:
        fig.tight_layout()
    
    if save:
        if path is None:
            dir_path = "./figures/"
            dir_path_obj = Path(dir_path)
            dir_path_obj.mkdir(parents=True, exist_ok=True)
            path = os.path.join(dir_path, 'attribution_plot.png')
        fig.savefig(path, dpi=300)
        print(f'figure saved to {path}')

