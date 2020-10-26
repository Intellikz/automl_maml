import numpy as np
import torch
import unittest

from utils import SineGenerator, TrainFromScratch
from solution import MAML

#Unit tests ran in 183 seconds
class TestMAML(unittest.TestCase):

    def test_1step(self):
        np.random.seed(1337)
        torch.manual_seed(42)

        task_loader = SineGenerator(k=10, k_test=40)
        m = MAML(num_steps = 1)
        for episode in task_loader.generator(episodes=40000):
            sine_function, task = episode
            m.train(task)
        
        performance = np.mean(m.history[-500:])
        self.assertTrue(performance < 0.7) # Precise performance: 0.5334421900827437

    def test_5step(self):
        np.random.seed(1337)
        torch.manual_seed(42)

        task_loader = SineGenerator(k=5, k_test=40)
        m = MAML(num_steps = 5)
        for episode in task_loader.generator(episodes=40000):
            sine_function, task = episode
            m.train(task)

        performance = np.mean(m.history[-500:])
        # Note performance of test_1step is better because it sees 10 data points
        # per task
        self.assertTrue(performance < 0.8) #0.6509916101358831