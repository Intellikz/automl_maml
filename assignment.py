import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import typing

from utils import BaseLearner

class MAML:
    """
    Implementation of model-agnostic meta-learning by Finn et al. (2017)
    """
    
    def __init__(self, num_steps: int) -> None:
        """
        Randomly initialize the base-learner initialization parameters, 
        number of update steps, and meta-optimizer (from the outer-loop)
        
        :param num_steps: int
        Number of updates to make to the initialization parameters in the inner-loop
        """
        
        # Number of updates to make in the inner-loop (on a given task)
        self.num_steps = num_steps
        # Loss function to optimize
        self.loss_fn = nn.MSELoss()
        # Shape of the base-learner network 
        self.initialization = BaseLearner.random_initial_weights()
        # Initialize the meta-optimizer, which will update the initialization point
        self.meta_optimizer = optim.Adam(self.initialization)
        
        # Keep a history of losses on the query set
        self.history = [] # DO NOT DELETE THIS. 
        
    def copy_params(self, params: typing.List[torch.Tensor]) -> None:
        """
        Makes and returns a copy of params. Use this to make a copy of fast_weights. 
        You can use this copy to compute gradients on the support set, which can be used to update the original fast weights.

        :param params: 
        """
        copy = [p.clone().detach() for p in params]
        for p  in copy:
            p.requires_grad = True
        return copy
        
    def train(self, task: typing.Sequence[torch.Tensor]) -> None:
        """
        Perform a single training iteration of MAML on the given task.
        1. Compute task-specific weights by applying <num_steps> regular gradient descent
           updates to the initialization parameters
        2. Update the initialization parameters using the meta_optimizer
        3. Update the moving average of the observed losses on the query sets
           with the task-specific weights
           
        :param task: typing.Sequence[torch.Tensor]]
        The given task to train on. It has the form: 
        (train_x, train_y, test_x, test_y). 
        The support set is given by (train_x, train_y)
        The query set is given by (test_x, test_y)
        """
        
        # Unpack the task to the support set (train_x, train_y) and 
        # query set (test_x, test_y)
        train_x, train_y, test_x, test_y = task        
        
        # Initialize fast weights to our initialization (specific to this task)
        fast_weights = [p.clone() for p in self.initialization]

        # Make S (self.num_steps) updates to the fast weights using the support set 
        for step in range(self.num_steps):
            # Make predictions with a copy of the fast weights on the input of the support set
            copy_fast_weights = self.copy_params(fast_weights)
            


        # -- End loop
        # Evaluate the loss on the query set and update the initialization parameters accordingly
        
        
        # Empty gradient buffers
        self.meta_optimizer.zero_grad()
        # Add the loss on the query set to the history
        self.history.append(loss.item()) # DO NOT DELETE THIS
