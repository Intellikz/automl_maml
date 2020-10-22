import numpy as np
import torch
import typing

from utils import SineGenerator, BaseLearner, TrainFromScratch, plot_history
from assignment import MAML

# Do not change this tests
def learning_curve(maml_steps=5, plot_tfs=False) -> None:
    """
    Plots the learning curve of MAML (and optionally TrainFromScratch)

    :param maml_steps: int
    Number of updates that MAML makes per task

    :param visualize: bool
    Whether to plot the learning curve of MAML and TrainFromScratch
    """
    
    task_loader = SineGenerator(k=10, k_test=40)
    m = MAML(num_steps = maml_steps)
    tfs = TrainFromScratch(num_steps = 10)
    for episode in task_loader.generator(episodes=40000):
        sine_function, task = episode
        m.train(task)
        if plot_tfs:
            tfs.train(task)

    if plot_tfs:
        plot_history(tfs.history, m.history)
    else:
        plot_history(None, m.history)

# Do not change these seeds!!!
np.random.seed(1337)
torch.manual_seed(42)

# Make sure to try out at least:
# 1-step MAML, and
# 3-step MAML
learning_curve(maml_steps=1, plot_tfs=True)
