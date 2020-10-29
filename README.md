# Model-Agnostic Meta-Learning

In this assignment, you are going to implement a very popular meta-learning algorithm for deep neural networks, namely [Model-Agnostic Meta-Learning (MAML)](https://arxiv.org/pdf/1703.03400.pdf). 

In the assignment, we will apply MAML to the Sine Wave Regression task which was proposed by the authors of MAML. The goal of this problem setup is to learn new sine wave functions *quickly* (from very few examples). 
Every task corresponds to a sine function f(x) = amplitude * sin(x + phase), where amplitude and phase are drawn uniformly at random from the ranges [0.1, 5.0) and [0, pi) respectively. Moreover, every task consists of a *support set* and a *query set*. For an example of a task, take a close look at Figure 2 in the [MAML paper](https://arxiv.org/pdf/1703.03400.pdf). When presented with a task, the base-learner neural network is allowed to learn from the support set. The degree of success of this learning process is then measured on the examples query set. By attempting to minimize the loss on the query sets after a learning on the support set, an algorithm can learn to learn.

MAML does this in the following manner. It aims to learn a good set of initialization parameters for a base-learner from which we can quickly learn new tasks within just a few gradient descent updates, as shown in the image below. ![Model-Agnostic Meta-Learning](maml.png)

Here, theta denotes the parameters of our base-learner neural network, which attempts to model sine wave curves. This network has 1 input node, 2 hidden layers of 40 ReLU nodes, and a final output layer of 1 node. MAML attempts to *meta-learn* (find) parameters for this network, from which we can quickly *learn* new sine waves (represented by tasks 1, 2, and 3 in the image). In our case, learning corresponds to minimizing the MSE loss on the query set (after learning on the support set). 

Pseudocode for MAML, with a small improvement by [Antoniou et al. (2019)](https://arxiv.org/pdf/1810.09502.pdf) is shown below:
```
1. Randomly choose initialization weights theta
2. For every task T:
3.   Compute fast weights for T using S steps of gradient descent with base-learner learning rate LR_base on the support set
4.   Compute the loss of the resulting weights on the query set
5.   Make a single update step on the initialization parameters according to this loss with learning rate LR_meta 
```

LR_base, LR_meta, and S are the only hyperparameters that MAML has. The approach is simple, effective, and much better than common transfer learning methods (which rely on pre-training and fine-tuning). **We will use LR_base = 0.01, and LR_meta = 0.001, in similar fashion to Finn et al. (2017). We keep S as a variable.** 

It is your task to implement this algorithm in `assignment.py` with the help of PyTorch. All requirements to run the script can be found in `requirements.txt`. 

To help you with the implementation, here are a few tips:
- We have already initialized the base-learner parameters for you in `self.initialization = [tensor1, tensor2, ..., tensorN]`. This is a list of tensors, which completely specifies the base-learner network, which aims to learn sine-wave functions. In this list, tensors are weights for a specific layer of the network. 
- The base-learner network, which attempts to learn sine wave functions, has already been implemented for you in `utils.py`. You will only need to use the `forward` function to make predictions on input data. Note that `BaseLearner` is not an object, so to call its functions, you could use `BaseLearner.forward(input_data, network_weights)`. 
- We will use the MSE loss function to measure the performance of the base-learner. This function is already instantiated (`self.loss_fn`) in the `__init__` function in the MAML class. Use this [link](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss) to see how to use the MSE function. 
- After computing the loss, you have to perform back-propagation, to obtain gradients. You have to explicitly demand pytorch to do this. 
- After back-propagation, you can access the gradients of tensors through the `.grad` attribute. Note that `self.initialization` is a list of tensors (base-learner parameters), and you can thus call `.grad` on every element. Use this to perform manual gradient descent in the inner-loop!
- Here is a good [blog post](https://towardsdatascience.com/paper-repro-deep-metalearning-using-maml-and-reptile-fd1df1cc81b0) on some meta-learning techniques. This is extra material, and is not required at all to complete this assignment (they use a completely different approach), but it is interesting! 

In the `assignment.py` file, you should only change the code in the `train` function of the MAML class, and **nothing else.**

You can run `test.py` to see how well your MAML implementation is learning. You can also compare it to a naive approach for learning new tasks (set `plot_tfs=True`), namely one that learns every task from scratch. **Note that the program becomes significantly slower when `plot_tfs=True`**. 

This is an accessible state-of-the-art method in meta-learning, so good luck, and have a lot of fun! 
