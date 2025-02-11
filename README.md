# Autodifferentiation Engine for Neural Networks

This project provides a step-by-step implementation of an **Autodifferentiation Engine** used in neural networks, focusing on the standard computational graph for forward and backward propagation. The engine handles automatic differentiation, which is essential for computing gradients during backpropagation and training neural networks.

## Files Overview

- **autodiff_engine.py**  
  This file contains the implementation of the autodifferentiation engine, which builds the computational graph, computes forward passes, and calculates gradients during the backward pass.

- **autodiff_engine.ipynb**  
  This Jupyter notebook demonstrates how to use the autodifferentiation engine, including examples of neural network training and gradient computation.

## Features

1. **Computational Graph**:  
   - Constructs a dynamic graph for each operation in the neural network, such as matrix multiplications, additions, and activations (e.g., ReLU, sigmoid).
   - Each node in the graph represents an operation, and edges represent dependencies (inputs/outputs).

2. **Forward Pass**:  
   - The engine computes the output of the network layer by layer.
   - Each node computes its output and stores intermediate results for later use in the backward pass.

3. **Backward Pass (Backpropagation)**:  
   - The engine computes gradients using backpropagation, iterating through the computational graph in reverse order.
   - Gradients are propagated backward from the output node to the input nodes, updating the weights of the network.

4. **Automatic Gradient Calculation**:  
   - The autodifferentiation engine automatically computes derivatives for each operation in the graph.
   - Simplifies the process of implementing training loops and gradient updates.

## How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/Subh-k9/autodiff_engine.git
cd autodiff_engine
