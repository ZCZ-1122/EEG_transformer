"""Train the model"""

# import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from torcheval.metrics import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall,\
                              MulticlassF1Score
from tqdm import tqdm

from eeg import utils, net
from eeg.evaluate import evaluate


def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """
    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []

    loss_avg = utils.RunningAverage()
    acc_multi = MulticlassAccuracy()
    precision = MulticlassPrecision(num_classes=params.num_classes, average="weighted")
    recall = MulticlassRecall(num_classes=params.num_classes, average="weighted")
    f1score = MulticlassF1Score(num_classes=params.num_classes, average="weighted")


    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if available
            train_batch  = train_batch.to(device=params.device, dtype=params.dtype)
            labels_batch = labels_batch.to(device=params.device, dtype=params.label_dtype)

            # compute model output and loss
            output_batch = model(train_batch).squeeze()
            loss = loss_fn(output_batch, labels_batch)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # update the average loss and streaming accuracy for this batch
            loss_avg.update(loss.item())

            #  convert logits to predicted labels (0,1,2...,C-1)
            pred_batch = torch.argmax(output_batch, dim=1)
            acc_multi.update(pred_batch, labels_batch)
            precision.update(pred_batch, labels_batch)
            recall.update(pred_batch, labels_batch)
            f1score.update(pred_batch, labels_batch)

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.detach().cpu().numpy()
                labels_batch = labels_batch.detach().cpu().numpy()

                # compute all non-streaming metrics on this batch
                summary_batch = {m: metrics[m](output_batch, labels_batch)
                                 for m in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # Compute and log mean of all metrics in summary
    metrics_mean = {metric: np.mean([ x[metric]
                                     for x in summ ]) for metric in summ[0]}
    metrics_mean['acc'] = acc_multi.compute().item()
    metrics_mean['precision'] = precision.compute().item()
    metrics_mean['recall'] = recall.compute().item()
    metrics_mean['f1score'] = f1score.compute().item()

    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)

    tr_epoch_hist = {'loss': loss_avg(),
                      'acc' : acc_multi.compute().item(),
                      'precision' : precision.compute().item(),
                      'recall' : recall.compute().item(),
                      'f1score' : f1score.compute().item(),
    }
    return tr_epoch_hist


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir,
                       restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # Display parameters
    print('\n\nConfiguration :\n------------------')
    params.display()
    print('\n')

    # Metric names
    metric_names = ['acc', 'precision', 'recall', 'f1score']
    assert params.best_metric in metric_names, f"{params.best_metric} not in {metric_names}"
    best_val_metric = 0.0

    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file +'.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    # Initialization history dictionary saving loss and metrics values after each epoch
    history = {'train_loss':[], 'val_loss' : []}
    for m in metric_names:
        history[f'train_{m}'] = []
        history[f'val_{m}'] = []

    # Create a Tensorboard writer if specified
    if params.write_tensorboard:
        writer = SummaryWriter()

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # Compute number of batches in one epoch (one full pass over the training set)
        tr_epoch_hist = train(model, optimizer, loss_fn, train_dataloader, metrics, params)

        # Evaluate for one epoch on validation set
        val_epoch_hist = evaluate(model, loss_fn, val_dataloader, metrics, params)

        val_metric = val_epoch_hist[params.best_metric]
        is_best = val_metric >= best_val_metric

        # Log to Tensorboard
        if params.write_tensorboard:
            writer.add_scalar("Loss/train", tr_epoch_hist['loss'], epoch)
            writer.add_scalar("Loss/val", val_epoch_hist['loss'], epoch)
            writer.add_scalar(f"{params.best_metric}/train", tr_epoch_hist[params.best_metric], epoch)
            writer.add_scalar(f"{params.best_metric}/val", val_epoch_hist[params.best_metric], epoch)

        # Save train, val loss and metrics on epoch end
        for (k,v) in tr_epoch_hist.items():
            history[f'train_{k}'].append(v) 
        for (k,v) in val_epoch_hist.items():
            history[f'val_{k}'].append(v)

        # Save weights
        if params.save_checkpoint:
            utils.save_checkpoint({'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                                'optim_dict': optimizer.state_dict()},
                                is_best=is_best,
                                checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info(f"- Found new best {params.best_metric}")
            best_val_metric = val_metric

            if params.save_best:
                best_model_path = os.path.join(model_dir, "so_far_best_model.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f'checkpoint saved to {best_model_path}')

    if params.write_tensorboard:
        writer.flush()
        writer.close()

    # restore the best model
    if params.restore_best : 
        model.load_state_dict(torch.load(best_model_path))
        print(f'\n*** Best model according to {params.best_metric} reloaded from {best_model_path} ***\n')

    return history
