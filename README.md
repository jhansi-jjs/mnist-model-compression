Neural Network Model Compression on MNIST
Overview

This project explores model compression techniques in deep learning using the MNIST handwritten digit dataset. The goal is to study how neural networks can be made smaller and more efficient while maintaining good accuracy. Experiments were implemented using PyTorch.

Dataset

The MNIST dataset contains grayscale images of handwritten digits from 0–9.

Training images: 60,000

Test images: 10,000

Image size: 28 × 28 pixels

Each image is flattened into 784 input features.

Model Architecture

A fully connected neural network was implemented with the following architecture:

784 → 128 → 10

Activation function: ReLU

Loss function: CrossEntropyLoss

Optimizer: Adam

Experiments

Baseline Model
The neural network was trained on the MNIST dataset and achieved 96.86% test accuracy.

Magnitude Pruning
L1 unstructured pruning was applied to remove 50% of the weights in the first layer while maintaining approximately 96% accuracy.

Knowledge Distillation
A smaller student network (784 → 32 → 10) was trained using predictions from the larger teacher model. The student model achieved 94.82% accuracy with significantly fewer parameters.

Technologies Used

Python

PyTorch

NumPy

Matplotlib

Conclusion

The experiments demonstrate that model compression techniques such as pruning and knowledge distillation can significantly reduce model size while maintaining strong performance.
