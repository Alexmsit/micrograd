# micrograd

## 1. Info

This repository contains code from the [spelled-out into to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0&t=1708s) by Andrej Karpathy as well as my adaptions to this code. The lecture shows how micrograd, a tiny autograd engine, is built from scratch.

<hr>

## 2. Micrograd introduction

Micrograd is an autograd engine which implements backpropagation by building a DAG (Directed Acyclic Graph) over a simple neural network with a PyTorch-like API, which allows to build and optimize deep neural networks.

<hr>

## 3. Implementation

The micrograd engine calculates the gradients of a mathematical expression with respect to the inputs of the expression (which are the weights in this case). The mathematical expression can always be viewed as a computational graph that combines the inputs through defined operations.

The engine is implemented through a Value class. This class is instanciated once with every single input to create Value objects. Since Python does not know how to do operations with Value objects, the class contains methodes for all basic operations. In this way, the mathematical expression can be calculated in the forward pass. The method of each operation also contains a backward function, which is used to calculate the gradient for this operation to construct the backward pass.

<hr>

## 4. Usage

The command below can be used to test the code and visualize the compuational graphs. The are four examples which can be shown can be chosen through arguments.

```
python main.py --example 0
```

Arguments:
- example: Select which example computational graph should be shown (eg. 0, 1, 2 or 3)


