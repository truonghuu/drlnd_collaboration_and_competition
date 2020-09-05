# Deep Reinforcement Learning Nanodgree 
## Project on Collaboration and Competition

### Project Description

![Environment Image](figures/tennis.png)

In this project, we will work with [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment, in which two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play. 

It is to be noted that while there are two agents in the environment, they are not competing against each other (i.e., it is not like a real tennis game where a player aims to hit the ball out of bound of the other player). Instead, the two agents will learn from each other to keep the ball in play as long as possible. 

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

 * Item: After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
 * Item: This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

In this project, we use an actor-critic algorithm, the Deep Deterministic Policy Gradients (DDPG) algorithm to train the agents. We implement the algorithm with a multi-agent approach to achieve the goal.

### Environment Description

The observation (state) space consists of 8 variables corresponding to the position and velocity of the ball and racket. The position is represented by a two values, leading to a total of 24 values in the state vector. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

### Code Overview

The repository consists of the following files:

* Item: Tennis.ipynb` - the main notebook which will be used to run and train the agent.
* Item:`agent.py` - defines the Agent that is being trained
* Item:`model.py` - defines the PyTorch model for the Actor and the Critic network
* Item:`checkpoint_actor.pth` - stores the weights of the trained Actor network when the environment is solved 
* Item:`checkpoint_critic.pth` - stores the weights of the trained Critic network when the environment is solved 
* Item:`Report.md` - The report presenting the details of the DDPG algorithm and experimental results.

### Getting Start

