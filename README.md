# Neural Network Model Compression on MNIST

## Overview

This project explores **model compression techniques in deep learning** using the MNIST handwritten digit dataset. The objective is to reduce model size and computational cost while maintaining high classification performance.

The following techniques were implemented and compared:

- Baseline Neural Network
- Magnitude-based Pruning
- Random Pruning
- Response-based Knowledge Distillation
- Feature-based Knowledge Distillation

All experiments were implemented using **PyTorch**.

---

## Dataset

- Dataset: MNIST (handwritten digit classification)
- Training samples: 60,000
- Test samples: 10,000
- Image size: 28 × 28 (grayscale)
- Input features: 784 (flattened)

---

## Model Architecture
# Neural Network Model Compression on MNIST

## Overview

This project explores **model compression techniques in deep learning** using the MNIST handwritten digit dataset. The objective is to reduce model size and computational cost while maintaining high classification performance.

The following techniques were implemented and compared:

- Baseline Neural Network
- Magnitude-based Pruning
- Random Pruning
- Response-based Knowledge Distillation
- Feature-based Knowledge Distillation

All experiments were implemented using **PyTorch**.

---

## Dataset

- Dataset: MNIST (handwritten digit classification)
- Training samples: 60,000
- Test samples: 10,000
- Image size: 28 × 28 (grayscale)
- Input features: 784 (flattened)

---

## Model Architecture




### Baseline Model
784 → 128 → 10

- Activation: ReLU  
- Loss Function: CrossEntropyLoss  
- Optimizer: Adam  

---

## Experiments and Results

### 1. Baseline Model

| Metric | Value |
|------|------|
| Parameters | 101,770 |
| Accuracy | 97.04% |
| Macro Precision | 0.97037 |
| Macro Recall | 0.97009 |
| Avg Inference Time | 8.95 × 10⁻⁵ s |
| Avg + Std | 0.0003756 |
| Avg - Std | -0.0001967 |

---

### 2. Pruning (50% Weight Removal)

#### Magnitude Pruning (L1)

| Metric | Value |
|------|------|
| Parameters | 101,770 |
| Accuracy | 97.03% |
| Macro Precision | 0.97030 |
| Macro Recall | 0.96997 |
| Avg Inference Time | 0.000325 s |
| Sparsity | 50% |

#### Random Pruning

| Metric | Value |
|------|------|
| Parameters | 101,770 |
| Accuracy | 94.53% |
| Macro Precision | 0.94715 |
| Macro Recall | 0.94467 |
| Avg Inference Time | 0.000312 s |
| Sparsity | 50% |

**Observation:**  
Magnitude pruning removes redundant weights with minimal performance loss, while random pruning degrades accuracy.

---

### 3. Knowledge Distillation (Response-Based)

#### Teacher Model

- Activation: ReLU  
- Loss Function: CrossEntropyLoss  
- Optimizer: Adam  

---

## Experiments and Results

### 1. Baseline Model

| Metric | Value |
|------|------|
| Parameters | 101,770 |
| Accuracy | 97.04% |
| Macro Precision | 0.97037 |
| Macro Recall | 0.97009 |
| Avg Inference Time | 8.95 × 10⁻⁵ s |
| Avg + Std | 0.0003756 |
| Avg - Std | -0.0001967 |

---

### 2. Pruning (50% Weight Removal)

#### Magnitude Pruning (L1)

| Metric | Value |
|------|------|
| Parameters | 101,770 |
| Accuracy | 97.03% |
| Macro Precision | 0.97030 |
| Macro Recall | 0.96997 |
| Avg Inference Time | 0.000325 s |
| Sparsity | 50% |

#### Random Pruning

| Metric | Value |
|------|------|
| Parameters | 101,770 |
| Accuracy | 94.53% |
| Macro Precision | 0.94715 |
| Macro Recall | 0.94467 |
| Avg Inference Time | 0.000312 s |
| Sparsity | 50% |

**Observation:**  
Magnitude pruning removes redundant weights with minimal performance loss, while random pruning degrades accuracy.

---

### 3. Knowledge Distillation (Response-Based)

#### Teacher Model
784 → 128 → 10

| Metric | Value |
|------|------|
| Parameters | 101,770 |
| Accuracy | 96.99% |
| Macro Precision | 0.97004 |
| Macro Recall | 0.96951 |
| Avg Inference Time | 0.000121 s |

#### Student Model

| Metric | Value |
|------|------|
| Parameters | 101,770 |
| Accuracy | 96.99% |
| Macro Precision | 0.97004 |
| Macro Recall | 0.96951 |
| Avg Inference Time | 0.000121 s |

#### Student Model
784 → 32 → 10

| Metric | Value |
|------|------|
| Parameters | 25,450 |
| Accuracy | 94.82% |
| Macro Precision | 0.94827 |
| Macro Recall | 0.94744 |
| Avg Inference Time | 7.00 × 10⁻⁵ s |

**Observation:**  
Model size reduced by approximately **75%** with only a small drop in accuracy.

---

### 4. Feature-Based Knowledge Distillation

In this approach, the student model learns from **teacher hidden layer representations** in addition to ground truth labels.

#### Method

- Student feature dimension: 32  
- Teacher feature dimension: 128  
- A projection layer (Linear: 32 → 128) is used to align feature spaces  

#### Loss Function

| Metric | Value |
|------|------|
| Parameters | 25,450 |
| Accuracy | 94.82% |
| Macro Precision | 0.94827 |
| Macro Recall | 0.94744 |
| Avg Inference Time | 7.00 × 10⁻⁵ s |

**Observation:**  
Model size reduced by approximately **75%** with only a small drop in accuracy.

---

### 4. Feature-Based Knowledge Distillation

In this approach, the student model learns from **teacher hidden layer representations** in addition to ground truth labels.

#### Method

- Student feature dimension: 32  
- Teacher feature dimension: 128  
- A projection layer (Linear: 32 → 128) is used to align feature spaces  

#### Loss Function
L = L_classification + L_feature

Where:

- L_classification = CrossEntropyLoss  
- L_feature = Mean Squared Error between teacher and student features  

**Observation:**  
Feature-based distillation improves representation learning by transferring deeper knowledge from the teacher network.

---

## Key Insights

- Neural networks are **over-parameterized**, allowing pruning without significant accuracy loss  
- **Magnitude pruning** is more effective than random pruning  
- Pruning introduces **sparsity** without reducing parameter count structurally  
- **Knowledge distillation** enables smaller models with competitive performance  
- Feature-based distillation enhances the quality of learned representations  

---

## Technologies Used

- Python  
- PyTorch  
- NumPy  
- Matplotlib  
- Scikit-learn  

---

## Project Structure
mnist_experiment.py # Baseline + pruning experiments
distillation_experiment.py # Response-based knowledge distillation
feature_distillation_experiment.py # Feature-based knowledge distillation
README.md

---

## Conclusion

This project demonstrates that model compression techniques such as **pruning and knowledge distillation** can significantly reduce model size while maintaining strong performance.

These methods are essential for deploying deep learning models in **resource-constrained environments** such as mobile devices and embedded systems.

---
