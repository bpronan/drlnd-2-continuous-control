[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

# Continuous Reach

This project covers the solution for Project 2 of the Udacity Deep Reinforcement Learning Nanodegree. The goal of the project was to train an agent in an environment with a continuous action space.

## Project Details

![Trained Agent](assets/ddpg_reacher_agent.gif)

This project required training an agent to manipulate articulated arms to reach a moving goal. A reward of +0.04 is provided for each step that the agent's hand is within the goal.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector is a number between -1 and 1.

## Getting Started

### Prerequisites (Conda)

1. Setup conda environment `conda create -n reacher python=3.6` and `conda activate reacher`.
1. Install [PyTorch version 0.4.1](https://pytorch.org/get-started/previous-versions/) for the version of CuDA you have installed.
2. Run `pip -q install ./python`

### Unity Environment Setup
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
2. Place the file in the `environments/` folder, and unzip (or decompress) the file.

### Instructions

Run `jupyter notebook` from this directory.

Open `Continuous_Control.ipynb` and run the cells to see how the agent was trained!
