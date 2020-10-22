import numpy as np
import torch
import typing

from utils import SineGenerator, BaseLearner, TrainFromScratch
#from assignment import MAML
from solution import MAML

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
