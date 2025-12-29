# README

## Overview

This project contains two Jupyter notebooks used to study optimization dynamics and loss-landscape geometry of neural networks trained on the MNIST dataset.

- Notebook 1 implements a custom L-BFGS optimizer with GSS line search and trains multiple neural network architectures.
- Notebook 2 analyzes the connectivity of minima by learning a smooth Fourier-parameterized path between two trained weight configurations.

All experiments are implemented in PyTorch.

---

## Requirements

- Python = 3.10.18  
- PyTorch = 2.8  
- torchvision = 0.23.0
- NumPy = 2.1.2

The MNIST dataset is downloaded automatically.

---

## Supported Architectures

All architectures are defined in `NN_arch.py`:

- FCP – Fully connected perceptron  
- LeNet (LN) – Convolutional neural network  
- AE – Autoencoder  
- LSTM – Sequential MNIST LSTM (`LSTM_sMNIST`)  

---

## Notebook 1: L-BFGS with GSS Line Search

### Purpose

This notebook trains neural networks using a custom L-BFGS optimizer that replaces standard Wolfe line search with GSS.  
The goal is to precisely control step sizes and study deterministic optimization behavior.

---

### Key Features

- Limited-memory BFGS (L-BFGS)
- Two-loop recursion for inverse Hessian approximation
- Brent line search with fixed evaluation budget
- Automatic reset when numerical instability is detected
- Full-batch training for deterministic gradients

---

### Training Setup

- Dataset: MNIST (train split)
- Batch size: Entire dataset
- Loss:
  - CrossEntropyLoss (FCP, LeNet, LSTM)
  - MSELoss (Autoencoder)
- Epochs: 1000
- History size: 10
- Line search budget: 6 evaluations

---

### Output

- Trained network parameters (saved externally)
- Printed training loss per epoch
- Weight checkpoints used later for loss-landscape path analysis

---

## Notebook 2: Fourier-Parameterized Optimization Paths

### Purpose

This notebook studies loss-landscape connectivity by learning a smooth path in parameter space between two trained weight vectors.

Given two solutions **ω_1** and **ω_2**, the path is defined using a truncated Fourier series:

**ω(t) = (1 - t)ω_1 + tω_2 + \sum_{n=1}^N b_n \sin(n\pi t)**

The Fourier coefficients **b_n** are learned to minimize loss along the path while maintaining smoothness.

---

### Key Components

- FourierPathNN: Generates a continuous weight path as a function of **t ∈ [0,1]**
- Path loss: Sum of task loss evaluated at discrete points along the path
- Smoothness regularization: Penalizes sharp changes between consecutive weights
- Optimizer: Adam

---

### Required Inputs

This notebook requires pretrained weight files generated from Notebook 1:

- `FC_BFGS_Training_best48_weights.npy`
- `LN_BFGS_Training_best48_weights.npy`
- `AE_BFGS_Training_best48_weights.npy`
- `LSTM_BFGS_Training_best48_weights.npy`

Each file must contains 48 flattened parameter vectors.

---

### Path Optimization Setup

- Fourier terms: 10
- Path resolution: 51 points between **t=0** and **t=1**
- Dataset: MNIST (train or test)
- Loss:
  - CrossEntropyLoss (classification models)
  - MSELoss (autoencoder)

---

### Output

- Loss values along the minimum-loss path
- Total path length in parameter space
- Printed loss and path length during optimization
