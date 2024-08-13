# Physics-Informed Neural Network (PINN) with LBFGS Optimization

This repository contains code for a Physics-Informed Neural Network (PINN) using PyTorch. The network is optimized using the LBFGS optimizer, and the model includes both observed loss and physical consistency constraints. The code is designed to predict physical parameters like the drag coefficient (`cd`) while ensuring that the predictions adhere to certain physical laws.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Key Considerations](#key-considerations)
- [Comparison Experiment](#comparison-experiment)

## Overview

This project implements a PINN model that is trained using both observational data and physical constraints. The primary focus is on accurately predicting the drag coefficient (`cd`) and other physical parameters by minimizing a loss function that includes multiple components:

1. **Observed Loss (`loss_obs`)**: The difference between the model's predictions and the observed data.
2. **Equation Loss (`loss_eqn`)**: A loss term that enforces physical consistency based on a given physical equation.
3. **Optional Gradient-Based Losses (`loss_u`, `loss_v`)**: These additional losses enforce that the gradients of the predicted positions with respect to the inputs align with the predicted velocities.

## Installation

To run this code, ensure you have Python installed along with the required dependencies. You can install the dependencies using:

```bash
pip install -r requirements.txt
```

## Running the Model
You can run the training process by executing the main script:
```bash
python main.py
```

This will train the model using the LBFGS optimizer and will output the drag coefficient (cd) and loss values during training.

### Customizing the Code
You can customize various aspects of the model, including:

The neural network architecture.
The loss function components.
The physical equations used in equation_.

## Key Considerations
### CAUTION: Impact of Gradient-Based Loss Terms
The function lbfgs_closure includes commented-out code that calculates gradient-based losses (loss_u and loss_v). These losses enforce stricter physical constraints by aligning gradients with predicted velocities. Uncommenting these lines will significantly alter the training dynamics, particularly the predicted cd value. It's crucial to understand how these additional constraints affect model performance.

### Comparison Experiment
To understand the impact of including or excluding the gradient-based loss terms:

### Run Training Without loss_u and loss_v:

Comment out the additional loss terms and train the model.
Observe the predicted cd value.
### Run Training With loss_u and loss_v:

Uncomment the additional loss terms and include them in the total loss.
Train the model and compare the predicted cd value.
Analyze the Results:

### Compare how the inclusion of these terms affects the cd value and the overall performance.
