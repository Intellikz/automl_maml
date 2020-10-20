import matplotlib.pyplot as plt
import numpy as np
import torch
import typing

#from assignment import SineGenerator, BaseLearner, TrainFromScratch, MAML
from solution import SineGenerator, BaseLearner, TrainFromScratch, MAML


def window(history: typing.List[float], ticks: int) -> typing.Union[typing.List[int], typing.List[float]]:
    """
    Plots a history of values using a window of size <ticks>
    
    :param history: typing.List[float]
    List of values to plot
    :param ticks: tick interval
    Window size over which we compute the average
    
    :return typing.Union[typing.List[int], typing.List[float]]
    Transformed x values and corresponding y values
    """
    
    windowed_values = [np.mean(history[i:i+ticks]) for i in range(0, len(history)-ticks, ticks)]
    xlabels = list(range(ticks, len(history), ticks))
    return xlabels, windowed_values

def plot_history(tfs_history: typing.List[float], maml_history: typing.List[float]):
    """
    Plots lists of loss values for TrainFromScratch and MAML
    
    :param tfs_history: typing.List[float]
    List of 'Train from Scratch' loss values to plot
    
    :param maml_history: typing.List[float]
    List of MAML loss values to plot
    """
    
    plt.figure(figsize=(10,8))
    plt.title("Sliding window of the MSE over time")
    plt.xlabel("Episode")
    plt.ylabel("Mean MSE loss in window")
    ticks = 300
    xmaml, maml_windowed = window(maml_history, ticks)
    xtfs, tfs_windowed = window(tfs_history, ticks)
    plt.plot(xtfs, tfs_windowed, label="Train from Scratch")
    plt.plot(xmaml, maml_windowed, label="MAML")
    plt.legend()
    plt.show()


# Do not change this tests
def test_maml(visualize=False) -> None:
    """
    Test case for MAML

    :param visualize: bool
    Whether to plot the learning curve of MAML and TrainFromScratch
    """
    
    task_loader = SineGenerator(k=10, k_test=40)
    m = MAML(num_steps = 1)
    tfs = TrainFromScratch(num_steps = 10)
    for episode in task_loader.generator(episodes=40000):
        sine_function, task = episode
        m.train(task)
        if visualize:
            tfs.train(task)
    
    _, task = task_loader.generate_task()
    maml_weights = m.initialization
    ground_truth= [2.0677974, -1.8574915, -1.6009828, -1.6887664, -1.391571, 
                   -1.6430949, -1.5081352, -0.62344056, 1.110019, 0.6495296]
    train_preds = np.array(BaseLearner.forward(task[0], maml_weights).detach()).flatten()

    assert np.all(np.isclose(train_preds, ground_truth)), "test_maml failed."
    print("Gratz!!! MAML test succeeded :)!")
    if visualize:
        plot_history(tfs.history, m.history)

# Do not change these seeds!!!
np.random.seed(1337)
torch.manual_seed(42)

test_maml(visualize=False)