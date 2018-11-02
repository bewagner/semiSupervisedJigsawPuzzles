from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from pathlib import Path
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import collections
import os
import sys
import logger
import helpers
import preprocessing


# TODO Move this to a more fitting place
class CyclicLR(object):
    """Sets the learning rate of each parameter group according to
    cyclical learning rate policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.
    Cyclical learning rate policy changes the learning rate after every batch.
    `batch_step` should be called after a batch has been used for training.
    To resume training, save `last_batch_iteration` and use it to instantiate `CycleLR`.
    This class has three built-in policies, as put forth in the paper:
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    This implementation was adapted from the github repo: `bckenstler/CLR`_
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for eachparam groups.
            Default: 0.001
        max_lr (float or list): Upper boundaries in the cycle for
            each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function. Default: 0.006
        step_size (int): Number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch. Default: 2000
        mode (str): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: 'triangular'
        gamma (float): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
            Default: 1.0
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
            Default: None
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: 'cycle'
        last_batch_iteration (int): The index of the last batch. Default: -1
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = pipeline.CyclicLR(optimizer)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         scheduler.batch_step()
        >>>         train_batch(...)
    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """

    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            print("Setting learning rate of {} to {}.".format(param_group, lr))
            param_group['lr'] = lr

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma ** (x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs


def logger_from_params(use_visdom=True, dataset=None, learning_rate=None, number_of_epochs=None, batch_size=None,
                       optimizer=None, model=None, number_of_permutations=None, momentum=None, transforms=None,
                       piece_crop_percentage=None, batch_norm=None, train_type=None):
    """
    Create a logger from given experiment parameters
    :param batch_norm:
    :param piece_crop_percentage: Percentage of cropping the images
    :param transforms: Training transforms
    :param use_visdom: Wether to visualize with visdom
    :param dataset: Name or path of the dataset used
    :param learning_rate: Learning rate
    :param number_of_epochs: Number of training epochs
    :param batch_size: Training batch size
    :param optimizer: Optimizer
    :param model: The CNN model
    :param number_of_permutations: Number of tile permutations
    :param momentum: Optimizer momentum
    :return: Logger
    """

    if isinstance(dataset, Path):
        dataset = dataset.parts[-1]

    optimizer_name = type(optimizer).__name__ if optimizer is not None else None
    model_name = type(model).__name__ if model is not None else None

    config = {'Dataset': dataset, 'Learning rate': learning_rate, 'Number of epochs': number_of_epochs,
              'Batch size': batch_size,
              'Optimizer': optimizer_name, 'Model': model_name, 'Number_of_Permutations': number_of_permutations,
              'Momentum': momentum, 'Transforms': [str(t) for t in transforms.transforms],
              'piece_crop_percentage': piece_crop_percentage, 'batch_norm': batch_norm, 'train_type': train_type}

    return logger_from_dict(config, use_visdom=use_visdom)


def logger_from_dict(config, use_visdom=True):
    """
    Create a logger from a dictionary
    :param config: Dictionary containing the experiment hyper-parameters
    :param use_visdom: Wether to use Visdom
    :return: Logger
    """
    experiment_name = datetime.datetime.now().strftime("%Y-%m-%d-")
    ordered_config = collections.OrderedDict(sorted(config.items()))

    for key, value in ordered_config.items():
        key = helpers.to_camel_case(key.replace(' ', '_'))

        if value is str:
            value = helpers.to_camel_case(value.replace(' ', '_'))

        if key == 'Transforms' or key == 'trainType':
            continue

        if value is not None and key is not None:
            experiment_name += "{}_{}-".format(key, value)

    # Delete last dash
    experiment_name = experiment_name[:-1]

    # Create logger
    log = logger.Experiment(name=experiment_name, use_visdom=use_visdom,
                            visdom_opts={'server': 'http://localhost', 'port': 8097},
                            time_indexing=False,
                            xlabel='Epoch')

    log.log_config(config)

    # create parent metric for training metrics (easier interface)
    log.ParentWrapper(tag='train', name='parent',
                      children=[log.AvgMetric(name='loss'),
                                log.AvgMetric(name='acc')])
    # same for validation metrics (note all children inherit tag from parent)
    log.ParentWrapper(tag='val', name='parent',
                      children=[log.AvgMetric(name='loss'),
                                log.AvgMetric(name='acc')])
    # Add a best metric for the validation accuracy
    log.ParentWrapper(tag='best', name='parent', children=[log.BestMetric(name='acc')])

    return log


def write_experiment_description(description: str, log: logger.Experiment):
    # Create directory for experiment
    experiment_dir = Path("experiments").joinpath(log.name)

    # Create the experiment directory if it does not already exist
    experiment_dir.mkdir(mode=0o777, parents=True, exist_ok=True)

    file_path = Path("experiments").joinpath(log.name).joinpath("description.txt")
    file_path.touch()
    with file_path.open(mode='w') as file:
        file.write(description)


def maybe_save_checkpoint(accuracy: float, model: nn.Sequential, epoch: int, log: logger.Experiment = None,
                          logging_dir: Path = Path("experiments")):
    """
    Save a model checkpoint.
    :param epoch:
    :param experiments_path:
    :param accuracy:
    :param model: The current model state
    :param log: The logger for the experiment
    """
    if log is None:
        if accuracy < 30:
            print("\nNo log was given so model will NOT BE SAVED.\n")
        return

    # Generate experiment name
    if log.acc_best is not None and accuracy < log.acc_best:
        return

    # Log the current accuracy value
    log.Parent_Best.update(acc=accuracy)
    log.Parent_Best.log_and_reset()

    # Create directory for experiment
    experiment_dir = logging_dir.joinpath(log.name)

    # Create the experiment directory if it does not already exist
    experiment_dir.mkdir(mode=0o777, parents=True, exist_ok=True)

    # Save the model
    model_path = experiment_dir.joinpath('model_' + log.config['train_type'] + '.pth.tar')
    print("Saved checkpoint")
    print(str(model_path))
    torch.save(model.state_dict(), model_path)

    # Save the logger
    logger_path = experiment_dir.joinpath('logger_' + log.config['train_type'] + '.json')
    log.to_json(logger_path.resolve())
    log.path = str(experiment_dir)

    # Save the logger parameters to a file
    file_path = experiment_dir.joinpath('params_' + log.config['train_type'] + '.txt')
    file_path.touch()
    with file_path.open(mode='w') as file:

        file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M\n\n"))
        file.write("{0:20}\t{1:.2}\n".format("Accuracy", log.acc_best))
        file.write("{0:20}\t{1}\n".format("Epoch", epoch))

        for key, value in log.config.items():
            if 'git' not in key and key is not None and value is not None:
                file.write('{0:20}\t{1}\n'.format(key, value))

        # Write model architecture to file
        file.write("\n\n" + str(model) + "\n")


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        max_k = max(topk)
        batch_size = target.size(0)

        _, prediction = output.topk(max_k, 1, True, True)
        prediction = prediction.t()
        correct = prediction.eq(target.view(1, -1).expand_as(prediction))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(train_loader, model, optimizer, device, epoch, log=None, criterion=nn.CrossEntropyLoss(), print_frequency=20,
          scheduler=None):
    log_is_valid = log is not None

    # switch to train mode
    model.train()

    start_time_epoch = time.time()
    for batch_index, (data, labels) in enumerate(train_loader):
        start_time_batch = time.time()

        if scheduler is not None:
            scheduler.batch_step()

        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()

        # compute output
        output = model(data)
        loss = criterion(output, labels)

        # Calculate accuracy and log data

        acc = accuracy(output, labels)

        if log_is_valid:
            log.Parent_Train.update(loss=loss.item(), acc=acc[0], n=output.size(0))

        # compute gradient and do optimizer step
        loss.backward()
        optimizer.step()

        if batch_index % print_frequency == 0:
            print('Epoch {0} [{1}/{2}]\t\t'
                  'Loss {loss_val:.4f}\t'
                  'Acc {acc_val:.3f}\t'
                  'Time {batch_time:.2f}'.format(
                epoch, batch_index, len(train_loader), loss_val=loss.item(),
                acc_val=acc[0].item(), batch_time=time.time() - start_time_batch))

    if log_is_valid:
        log.Parent_Train.log_and_reset()
        print('\nFinished epoch {0} in {time:.2f}s\t'
              'Avg. Loss {loss:.4f}\t'
              'Avg. Accuracy {acc:.3f}'.format(epoch, loss=log.loss_train, acc=log.acc_train,
                                               time=time.time() - start_time_epoch))


def validate(val_loader, model, device, epoch, log=None, criterion=nn.CrossEntropyLoss()):
    # switch to evaluate mode
    model.eval()

    log_is_valid = log is not None

    start_time_epoch = time.time()

    with torch.no_grad():
        for i, (tiles, permutations) in enumerate(val_loader):
            tiles, permutations = tiles.to(device), permutations.to(device)

            # compute output
            output = model(tiles)
            loss = criterion(output, permutations)

            # # measure accuracy and record loss
            acc = accuracy(output, permutations)

            if log_is_valid:
                log.Parent_Val.update(loss=loss.item(), acc=acc[0], n=output.size(0))

    if log_is_valid:
        log.Parent_Val.log_and_reset()
        print('Validated epoch {0} in {time:.2f}s'
              '\tLoss {loss:.4f}'
              '\tAccuracy {acc:.3f}\n'.format(epoch, loss=log.loss_val, acc=log.acc_val,
                                              time=time.time() - start_time_epoch))
    if log_is_valid:
        return log.acc_val
    else:
        return None


def load_weights(model: nn.Sequential, path: Path, train_type: str) -> nn.Sequential:
    """
    Function to load the weights from a given path into a model
    :param train_type:
    :param model: Model
    :param path: Wether the directory containing the weights or the path to the weight file itself
    :return: The model with the weights loaded
    """
    if path.is_dir():
        path = path.joinpath("model_" + train_type + ".pth.tar")

    elif not str(path.resolve()).endswith(".pth.tar"):
        raise ValueError("Given weight path could not be found.")

    model.load_state_dict(torch.load(path))
    return model


class LearningRateFinder:
    def __init__(self, model, data_loader, criterion, optimizer, device, start_cutoff=5, end_cutoff=10, init_value=1e-8,
                 final_value=10.0, beta=0.98):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.start_cutoff = start_cutoff
        self.end_cutoff = end_cutoff
        self.init_value = init_value
        self.final_value = final_value
        self.beta = beta

        self.learning_rates, self.losses = self.search_run(self.model, self.data_loader, self.criterion, self.optimizer,
                                                           self.device, self.init_value, self.final_value,
                                                           self.beta)

    def plot_learning_rate(self):
        """
        Plot the calulated losses vs. the learning rates
        :return: Plot
        """
        learning_rates = self.learning_rates[self.start_cutoff:-self.end_cutoff]
        losses = self.losses[self.start_cutoff:-self.end_cutoff]

        p = plt.plot(learning_rates, losses)[0]
        plt.xscale('log')

        plt.xlabel('learning rate (log scale)')
        plt.ylabel('losses')

        p.axes.spines['right'].set_visible(False)
        p.axes.spines['top'].set_visible(False)

        plt.show()
        return p

    @staticmethod
    def search_run(model, data_loader, criterion, optimizer, device, init_value=1e-8, final_value=10., beta=0.98):
        """
        Run a search train_classify_stl10 of the algorithm described in "No More Pesky Learning Rate Guessing Games (Smith, 2015)".

        The result is an array of losses and the corresponding learning rates.
        :param model: Model to use
        :param data_loader: Data loader for the model
        :param criterion: Criterion for the loss
        :param optimizer: Optimizer to us
        :param device: Device on which calculations are made (CPU or CUDA)
        :param init_value: Initial learning rate value (default=1e-8)
        :param final_value: Final learning rate value (default=10)
        :param beta: Parameter for exponential smoothing of the learning rate curve (default=0.98)
        :return: Learning rates and losses as lists
        """
        model.to(device)

        training_examples_n = len(data_loader) - 1

        multiplication_factor = (final_value / init_value) ** (1 / training_examples_n)

        lr = init_value
        optimizer.param_groups[0]['lr'] = lr

        average_loss = 0.
        best_loss = 0.
        batch_n = 0
        losses = []
        learning_rates = []

        for data in data_loader:
            batch_n += 1

            if type(data) is dict:
                inputs, labels = [e.to(device) for e in data.values()]
            else:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Compute the smoothed loss
            average_loss = beta * average_loss + (1 - beta) * loss.item()
            smoothed_loss = average_loss / (1 - beta ** batch_n)

            # Stop if the loss is exploding
            if batch_n > 1 and smoothed_loss > 4 * best_loss:
                return learning_rates, losses

            # Record the best loss
            if smoothed_loss < best_loss or batch_n == 1:
                best_loss = smoothed_loss

            # Store the values
            losses.append(smoothed_loss)
            learning_rates.append(lr)

            # Do the SGD step
            loss.backward()
            optimizer.step()

            # Update the lr for the next step
            lr *= multiplication_factor
            optimizer.param_groups[0]['lr'] = lr

        return learning_rates, losses
