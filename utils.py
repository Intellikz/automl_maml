import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import typing

########################################################################
# DONT CHANGE ANYTHING IN THIS FILE
########################################################################
class SineGenerator(object):
    """
    Class that is responsible for generating sine wave data. 
    """
    
    def __init__(self, k: int, k_test: int) -> None:
        """
        Initialize the SineGenerator object

        
        :param k: int
        Number of data points in the support set of a task
        :param k_test: int
        Number of data points in the query set of a task
        """
        
        # Check argument types
        error_message = "Argument '%s' in constructor of SineGenerator object should be an integer"
        assert isinstance(k, int), error_message % 'k'
        assert isinstance(k_test, int), error_message % 'k_test'
        
        # Set parameters
        self.k = k
        self.k_test = k_test
        
    
    def __generate_random_function(self) -> typing.Callable[[float], float]:
        """
        Generates a random sine wave function with amplitude and phase chosen
        uniformly at random between [0.1, 5.0) and [0, np.pi)
        
        :return: Callable[[float], float]
        Function that takes an x-coordinate (float) and returns the sine function 
        value (float) corresponding to that input 
        """
        
        # Sample amplitude and phase uniformly at random
        amplitude = np.random.uniform(0.1, 5.0)
        phase = np.random.uniform(0, np.pi)
        # Create and return function with the obtained amplitude and phase
        fn = lambda x: amplitude * np.sin(x + phase)
        return fn
    
    def __to_torch(self, iterable: typing.Sequence[np.array]) -> typing.Sequence[torch.Tensor]:
        """
        Convert all numpy input arrays to torch.Tensors
        
        :param iterable: typing.Sequence[np.array]
        The numpy input arrays to be converted
        
        :return typing.Sequence[torch.Tensor]
        The converted numpy array (as sequence of torch.Tensor variables)
        """
        
        return [torch.from_numpy(el) for el in iterable]
    
    def generate_task(self) -> typing.Union[typing.Callable[[float], float], 
                                              typing.Sequence[np.array]]:
        """
        Generate a single task in 3 steps:
         1. Pick a random sine wave function
         2. Generate the support set 
         3. Generate the query set
         
        :return:  typing.Union[typing.Callable[[float], float], 
                               typing.Sequence[np.array]]
        The sine wave function object, and the support (train_x, train_y) and query set
        (test_x, test_y)
        """
        
        # Generate a sine function
        fn = self.__generate_random_function()
        # Generate support set
        train_x = np.random.uniform(-5.0, +5.0, size=(self.k,1)).astype("float32")
        train_y = fn(train_x)
        # Generate query set
        test_x = np.random.uniform(-5.0, +5.0, size=(self.k_test,1)).astype("float32")
        test_y = fn(test_x)
        return fn, self.__to_torch([train_x, train_y, test_x, test_y])
        
    def generator(self, episodes : int) -> typing.Union[typing.Callable[[float], float], 
                                           typing.Sequence[np.array]]:
        """
        Generator object that yields <episodes> tasks = (sine function, support set, 
        query set), or equivalently, (sine function, train_x, train_y, test_x, test_y)
        
        :param episodes: int
        The number of tasks to generate
        
        :return: typing.Union[typing.Callable[[float], float], 
                              typing.Sequence[np.array]]
        Tasks
        """
        
        assert isinstance(episodes, int), "'episodes' should be of integer type"
        # Generate tasks!
        for _ in range(episodes):
            yield self.generate_task()

class BaseLearner(object):
    """
    Neural network base-learner for the sine wave regression task.
    The architecture is as follows: 
    (1 input -> 40 hidden nodes -> ReLU -> 40 hidden nodes -> ReLU -> 1 output node) 
    """
        
    def random_initial_weights() -> typing.List[torch.Tensor]:
        """
        Returns a set of random initial parameters for the base-learner network.

        :return typing.List[torch.Tensor]:
        A list of random initialization parameters in the form:
        initialization[i]: 
          - Weight tensor of the (i+1)-th layer if i is even
          - Bias tensor of the (i)-th layer if i is uneven
        """
        
        # Shape of the base-learner network
        # (1 input -> 40 hidden nodes -> 40 hidden nodes -> 1 output node) 
        shape = [1,40,40,1]
        initialization = []
        for i in range(len(shape) - 1):
            # Compute the input and output shapes
            input_shape = shape[i]
            output_shape = shape[i+1]
            layer = nn.Linear(in_features=input_shape, out_features=output_shape)
            layer.bias = nn.Parameter(torch.zeros(output_shape))
            params = list(layer.parameters())
            initialization += params
        return initialization
    
    def forward(x: torch.Tensor, weights: typing.List[torch.Tensor]) -> torch.Tensor:
        """
        Feedforward pass of the neural network. Takes input tensor x
        and network weights (theta) to produce a prediction tensor y. 
        
        This forward pass assumes the following structure of the weights tensor:
        weights[i]: 
          - Weight tensor of the (i+1)-th layer if i is even
          - Bias tensor of the (i)-th layer if i is uneven
          
        The network architecture that is presumed is: 
        1 input node (x-coordinate) -> 40 hidden nodes -> ReLU 
        -> 40 hidden nodes -> ReLU -> 1 output node (predicted y-coordinate)
    
        :param x: torch.Tensor
        Input tensor of size (batch_size, 1)
        
        :param weights: typing.List[torch.Tensor]
        List of base-learner weights (theta) that should be used to compute
        a prediction for input x
        
        :return: torch.Tensor
        Tensor of predictions y of shape (batch_size, 1)
        """
        
        x = F.linear(x, weight=weights[0], bias=weights[1])
        x = F.relu(x)
        x = F.linear(x, weight=weights[2], bias=weights[3])
        x = F.relu(x)
        x = F.linear(x, weight=weights[4], bias=weights[5])
        return x

class TrainFromScratch(object):
    """
    Naive strategy that learns a new model from scratch for every new task
    """
    
    def __init__(self, num_steps: int) -> None:
        """
        Initialize the model with the: 
         - Number of update steps it is allowed to make per task
         - Random initialization from which we start over every task
         - Optimization function (Adam)
         - Empty history of MSE losses over time
        """
        
        self.num_steps = num_steps
        self.loss_fn = nn.MSELoss()
        self.initialization = BaseLearner.random_initial_weights()
        self.opt_fn = optim.Adam
        
        self.history = []
                
    def train(self, task: typing.Sequence[torch.Tensor]) -> None:
        """
        Train from scratch for <num_steps> epochs on the support set of the task
        Evaluate on the query set
        
        :param task: typing.Sequence[torch.Tensor]
        Task consisting of the suport set (train_x, train_y) 
        and the query set (test_x, test_y)
        """
        
        train_x, train_y, test_x, test_y = task  
        # Create task-specific weights in order to maintain the random initilization 
        # for new tasks afterwards
        weights = [p.clone().detach() for p in self.initialization]
        for p in weights:
            p.requires_grad = True
        # Initialize the optimizer we use (Adam)
        optimizer = self.opt_fn(weights)
        # Make <num_steps> updates to the weights 
        for step in range(self.num_steps):
            predictions = BaseLearner.forward(train_x, weights)
            loss = self.loss_fn(predictions, train_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Evaluate on the query set
        predictions = BaseLearner.forward(test_x, weights)
        loss = self.loss_fn(predictions, test_y)
        self.history.append(loss.item())

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
    
    plt.figure()
    plt.title("Sliding window of the MSE over time")
    plt.xlabel("Episode")
    plt.ylabel("Mean MSE loss in window")
    ticks = 300
    xmaml, maml_windowed = window(maml_history, ticks)
    if not tfs_history is None:
        xtfs, tfs_windowed = window(tfs_history, ticks)
        plt.plot(xtfs, tfs_windowed, label="Train from Scratch")
    plt.plot(xmaml, maml_windowed, label="MAML")
    plt.legend()
    plt.show()
