# Deep Reinforcement Learning Nanodgree 
## Project on Collaboration and Competition

### Project Description

![Environment Image](figures/tennis.png)

In this project, we will work with [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment, in which two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play. 

It is to be noted that while there are two agents in the environment, they are not competing against each other (i.e., it is not like a real tennis game where a player aims to hit the ball out of bound of the other player). Instead, the two agents will learn from each other to keep the ball in play as long as possible. 

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

 * After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
 * This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

In this project, we use an actor-critic algorithm, the Deep Deterministic Policy Gradients (DDPG) algorithm to train the agents. We implement the algorithm with a multi-agent approach to achieve the goal.

### Environment Description

The observation (state) space consists of 8 variables corresponding to the position and velocity of the ball and racket. The position is represented by a two values, leading to a total of 24 values in the state vector. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

### Code Overview

The repository consists of the following files:

* `Tennis.ipynb` - the main notebook which will be used to run and train the agent.
* `agent.py` - defines the Agent that is being trained
* `model.py` - defines the PyTorch model for the Actor and the Critic network
* `checkpoint_actor.pth` - stores the weights of the trained Actor network when the environment is solved 
* `checkpoint_critic.pth` - stores the weights of the trained Critic network when the environment is solved 
* `Report.md` - The report presenting the details of the DDPG algorithm and experimental results.

### Getting Start

#### Download the Environment

To be able to run this project, you need to install dependencies by following the instructions in this [link](https://github.com/udacity/deep-reinforcement-learning#dependencies).

A prebuilt environment has to be installed. You need to download the corresponding environment depending on your operating system.
* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Unzip the file to a folder, which you need to refer at the second code cell in `Tennis.ipynb`. 

#### Dependencies

If you are running this project in your local desktop, you need to install the following libraries.
* `tensorflow==1.7.1`
* `Pillow>=4.2.1`
* `matplotlib`
*` numpy>=1.11.0`
* `jupyter`
* ` pytest>=3.2.2`
* `docopt`
* `pyyaml`
* `protobuf==3.5.2`
* `grpcio==1.11.0`
* `torch==0.4.0`
* `pandas`
* `scipy`
* `ipykernel`

You can now start exploring the environment and implement your own training algorithm! Open `Tennis.ipynb` and follow step by step to train the agent and see its working.

