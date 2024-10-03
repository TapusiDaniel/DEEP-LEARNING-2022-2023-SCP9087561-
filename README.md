# DEEP LEARNING 2022-2023 (SCP9087561)

This GitHub repository contains materials and projects from the Deep Learning course I attended at the University of Padova during the 2022/2023 academic year. It includes my homework assignments and other exercises that demonstrate practical applications of deep learning concepts.

# Course Overview

The Deep Learning course aimed to provide a comprehensive introduction to the fundamental principles and techniques of deep learning, focusing on neural networks. By the end of the course, students were expected to:
  * Understand the architecture and training of deep feedforward neural networks
  * Apply convolutional neural networks (CNNs) for image processing tasks
  * Implement recurrent neural networks (RNNs) and transformer models for sequence tasks such as time series or natural language processing
  * Develop and analyze autoencoders and variational autoencoders for data compression and generation
  * Gain practical experience using popular deep learning frameworks like PyTorch and TensorFlow for implementing neural networks and deep learning algorithms

The course also covered advanced topics such as optimization techniques, regularization, and generative models, providing a solid foundation for working with deep learning in research and industry settings.

# Examination and Assessment

The course evaluation consisted of the following components:
  * **Written Exam**: This closed-book exam assessed the students' understanding of the theoretical concepts in deep learning, including architectures, optimization methods, and neural network training
  * **Homework Assignments**: The homework assignments required students to apply deep learning models to various real-world problems. These were evaluated based on the correct implementation of models, analysis of results, and understanding of key concepts

# Homework Assignments

[Homework 1: From the Perceptron to Deep Neural Networks](https://github.com/TapusiDaniel/DEEP-LEARNING-2022-2023-SCP9087561-/blob/main/2065492_HW1.ipynb)

In this homework, we explored the fundamentals of neural networks, starting with the basic perceptron model and advancing to a multi-layer feedforward network. The progression involved understanding both the theoretical concepts and practical implementation in Python using NumPy.

### Key tasks:
  1. **Perceptron Implementation**:
     * Developed a simple perceptron to classify linearly separable data
     * Implemented key components like activation functions, weight initialization, and forward propagation
     * Added backpropagation and batch training to optimize weights
  2. **Analysis of Training**:
     * Explored how the perceptron learns over multiple iteration
     * Investigated the fourth sample in the dataset to understand misclassifications, discussing potential adjustments to improve its prediction
  3. **Two-Layer Neural Network**:
     * Implemented a two-layer neural network capable of solving more complex problems that a single-layer perceptron could not handle, such as non-linearly separable data
     * Extended the network to handle logical operations and experimented with adding hidden nodes to optimize the model's performance
  4. **Handwritten Digits Classification**:
     * Built a neural network for digit classification using the well-known MNIST dataset
     * Trained the model with backpropagation and tuned hyperparameters to enhance performance

[Homework 2: Optimization and Regularization in Deep Learning](https://github.com/TapusiDaniel/DEEP-LEARNING-2022-2023-SCP9087561-/blob/main/2065492_HW2.ipynb)

In this homework, we developed a deep neural network for text classification using PyTorch. The tasks focused on building, training, and evaluating a model while addressing common deep learning challenges such as overfitting and hyperparameter tuning.

### Key tasks:
  1. **Feedforward Neural Network**:
     * Implemented a 3-layer deep feedforward neural network using PyTorch for a text classification task
     * Loaded the AG News Subset dataset and performed preprocessing steps like tokenization
     * Defined the model architecture, including layers, activations, and weight initialization
  2. **Overfitting Analysis**:
     * Trained the model on a limited dataset and observed how overfitting occurs
     * Implemented strategies to deliberately overfit the model by adjusting training parameters and visualized the results
  3. **Regularization Techniques**:
     * Applied L1 and L2 regularization to mitigate overfitting
     * Explored how these regularization techniques influence the model's performance by adding norm penalties to the objective function
  4. **Early Stopping**:
     * Implemented early stopping to prevent overfitting by monitoring the validation error and halting training when generalization performance began to degrade
  5. **Model Selection via Grid Search**:
     * Conducted a grid search to find the optimal hyperparameters for the model, such as learning rate and regularization strength
     * Utilized scikit-learn's GridSearchCV to evaluate multiple combinations of hyperparameters and identify the best-performing model

[Homework 3: Convolutional Neural Networks (CNNs)](https://github.com/TapusiDaniel/DEEP-LEARNING-2022-2023-SCP9087561-/blob/main/2065492_HW3.ipynb)

In this homework, we developed Convolutional Neural Networks (CNNs) for image classification tasks using the CIFAR-10 dataset. The focus was on building various models, understanding the effects of different hyperparameters, and applying transfer learning techniques.

### Key tasks:
  1. **Simple CNN Development**:
     * Loaded the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes
     * Preprocessed the images by scaling pixel values and splitting the dataset into training, validation, and test sets
     * Defined a simple CNN architecture, including a convolutional layer, max pooling, flattening, and a dense output layer
  2. **Model Training and Visualization**:
     * Trained the simple CNN model and visualized the learned filters in the convolutional layer, gaining insights into feature extraction
     * Calculated the number of parameters in the Conv2D layers and explained the formula used for this calculation
  3. **Deep CNN Exploration**:
     * Developed a deeper CNN model with additional layers and filters, analyzing the impact on performance. Observed a drop in accuracy with a more complex model, prompting further exploration of hyperparameters
  4. **Hyperparameter Tuning**:
     * Experimented with various hyperparameter settings, such as the number of layers, filter sizes, activation functions, epochs, and batch sizes
     * Identified the best-performing model, which improved test accuracy through adjustments like increased layers and filters
  5. **Transfer Learning with ResNet18**:
     * Implemented transfer learning by loading a pre-trained ResNet18 model trained on the ImageNet dataset
     * Fine-tuned the model on the CIFAR-10 dataset by adding a fully connected layer, demonstrating the effectiveness of leveraging pre-trained models for new tasks

[Homework 4: *Recurrent Neural Networks & Transformer](https://github.com/TapusiDaniel/DEEP-LEARNING-2022-2023-SCP9087561-/blob/main/2065492_HW4.ipynb)
