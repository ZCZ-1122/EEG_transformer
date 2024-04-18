"""Evaluates the model"""

# import argparse
import logging
import os

import numpy as np

import torch
from eeg import utils, net

from torcheval.metrics.classification import BinaryRecall, BinaryPrecision, \
                                             BinaryAccuracy
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, \
                             MulticlassRecall


def predict_v2(data_loader, model, params, 
                return_gt=False, return_data=False):
    """
    return : `all_scores` if return_gt=False, else (`all_scores`, `all_gt`)
    """
    # torch.manual_seed(230)
    # if params.device == torch.device('cuda'):
    #     torch.cuda.manual_seed(230)

    # Prediction
    model.eval()

    out_dict = {}
    out_dict['all_scores'] = []
    if return_gt:
        out_dict['all_gt'] = []
    if return_data:
        out_dict['all_data'] = []

    with torch.no_grad(): # all computation will have requires_grad = False
        for data_batch, labels_batch in data_loader:
            data_batch   = data_batch.to(device=params.device, dtype=params.dtype)
            labels_batch = labels_batch.to(device=params.device, dtype=params.label_dtype)

            # compute model output
            scores_batch = model(data_batch).squeeze()

            out_dict['all_scores'].extend(scores_batch)
            if return_gt:
                out_dict['all_gt'].extend(labels_batch)
            if return_data:
                out_dict['all_data'].extend(data_batch)

    # Evaluation (accuracy, precision and recall)
    # all_scores = torch.tensor(all_scores).detach().cpu().numpy()
    for k,v in out_dict.items():
        out_dict[k] = torch.stack(v, dim=0).detach().cpu().numpy()

    # all_scores = torch.stack(all_scores, dim=0).detach().cpu().numpy()
    # if return_gt:
    #     all_gt = torch.stack(all_gt, dim=0).detach().cpu().numpy()
    #     return all_scores, all_gt
    # else:
    #     return all_scores
    return out_dict
            

def evaluate(model, loss_fn, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    acc_multi = MulticlassAccuracy(num_classes=params.num_classes)
    precision = MulticlassPrecision(num_classes=params.num_classes, average="weighted")
    recall = MulticlassRecall(num_classes=params.num_classes, average="weighted")
    f1score = MulticlassF1Score(num_classes=params.num_classes, average="weighted")

    
    with torch.no_grad():
        # compute metrics over the dataset
        for data_batch, labels_batch in dataloader:
            
            data_batch = data_batch.to(device=params.device, dtype=params.dtype)
            labels_batch = labels_batch.to(device=params.device, dtype=params.label_dtype)
            
            # compute model output
            output_batch = model(data_batch).squeeze()
            loss = loss_fn(output_batch, labels_batch)

            # update the streaming accuracy for this batch
            # stream_acc.update(output_batch, labels_batch)
            pred_batch = torch.argmax(output_batch, dim=1) # first convert logits to predicted labels
            acc_multi.update(pred_batch, labels_batch)
            precision.update(pred_batch, labels_batch)
            recall.update(pred_batch, labels_batch)
            f1score.update(pred_batch, labels_batch)

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.detach().cpu().numpy()
            labels_batch = labels_batch.detach().cpu().numpy()

            # compute all non-streaming metrics on this batch
            summary_batch = {m: metrics[m](output_batch, labels_batch)
                            for m in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)

        # compute and log mean of all metrics in summary
        metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
        metrics_mean['acc'] = acc_multi.compute().item()
        metrics_mean['precision'] = precision.compute().item()
        metrics_mean['recall'] = recall.compute().item()
        metrics_mean['f1score'] = f1score.compute().item()

        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
        logging.info("- Eval metrics : " + metrics_string)

    return metrics_mean


def evaluate_v2(data_loader, model, params, decision_thre=.0):

    # Init metrics
    recalls = BinaryRecall(threshold=decision_thre, device=params.device)
    precisions = BinaryPrecision(threshold=decision_thre, device=params.device)
    accuracy = BinaryAccuracy(threshold=decision_thre, device=params.device)

    recalls.reset()
    precisions.reset()
    accuracy.reset()

    # Prediction
    model.eval()
    with torch.no_grad():
        # compute metrics over the dataset
        for data_batch, labels_batch in data_loader:
            data_batch   = data_batch.to(device=params.device, dtype=params.dtype)
            labels_batch = labels_batch.to(device=params.device, dtype=torch.long)

            # compute model output
            scores_batch = model(data_batch).squeeze()

            # Metric update
            recalls.update(scores_batch, labels_batch)
            precisions.update(scores_batch, labels_batch)
            accuracy.update(scores_batch, labels_batch)

        return {'acc' : accuracy.compute().item(),
                'precision' : precisions.compute().item(),
                'recall' : recalls.compute().item()}
