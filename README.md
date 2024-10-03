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

[Homework 4: Recurrent Neural Networks & Transformer](https://github.com/TapusiDaniel/DEEP-LEARNING-2022-2023-SCP9087561-/blob/main/2065492_HW4.ipynb)

In this homework, we explored the development of Recurrent Neural Networks (RNNs), including LSTMs, GRUs, and Transformers for sentiment analysis using the IMDB dataset. The focus was on comparing the performance and computational efficiency of various models and understanding the underlying mechanisms of sequence-based architectures.

### Key tasks:
  1. **Simple RNN for Sentiment Analysis**:
     * Used the IMDB dataset, preprocessed reviews into word index sequences, and split the dataset into training, validation, and test sets
     * Implemented a basic RNN model with an embedding layer, followed by a recurrent layer and a dense output layer. The task was binary classification (positive/negative sentiment)
  2. **Comparison with LSTM and GRU**:
     * Replaced the simple RNN layer with LSTM and GRU layers to compare performance
     * Analyzed differences in training time, the number of parameters, and model accuracy between the three models
     * Noted that LSTMs and GRUs, with their ability to handle long-term dependencies, generally outperformed the basic RNN in terms of accuracy while increasing computational complexity
  3. **Bidirectional LSTM**:
     * Implemented a bidirectional LSTM to explore its ability to capture context from both directions (past and future) in a sequence
     * Compared the bidirectional LSTM to the regular LSTM, observing improvements in model performance at the cost of additional computation
  4. **Exploring Word Embeddings**:
     * Analyzed word embeddings learned by the model, exploring relationships between words using geometric properties
     * Checked whether word analogies, such as "king - man + woman = queen," were reflected in the learned embeddings, demonstrating how the model captures word relations
  5. **Transformer for Sentiment Analysis**:
     * Implemented a Transformer model to perform the same sentiment classification task
     * Compared the Transformer’s performance with that of the RNN, LSTM, and GRU models. Experimented with hyperparameters such as the number of heads in the multi-head attention mechanism, feedforward layer size, and dropout rates
     * Observed the advantages of the Transformer, particularly in terms of parallelization and capturing long-range dependencies more efficiently
  6. **Hyperparameter Tuning and Analysis**:
     * Experimented with various configurations for the RNN, LSTM, GRU, and Transformer models
     * Provided a detailed analysis of how changing hyperparameters (e.g., hidden units, learning rates, batch size, and dropout) impacted model performance in terms of accuracy, training time, and generalization

[Homework 5: Autoencoders](https://github.com/TapusiDaniel/DEEP-LEARNING-2022-2023-SCP9087561-/blob/main/2065492_HW5.ipynb)

In this homework, we focused on dimensionality reduction techniques using autoencoders. We developed shallow and deep autoencoders and experimented with their performance in reconstructing the CIFAR-10 dataset. Additionally, we explored denoising autoencoders by adding noise to the input data and training the model to reconstruct the original images.

### Key tasks:
  1. **Singular Value Decomposition (SVD)**:
     * Applied SVD, a linear dimensionality reduction technique, to the CIFAR-10 dataset to analyze its performance
     * Decomposed the data matrix using the torch.linalg.svd function and compared it with autoencoder results
     * Utilized a reduced version of the data matrix to optimize memory consumption and computational efficiency
  2. **Shallow Linear Autoencoder**:
     * Developed a shallow autoencoder consisting of one fully connected layer for encoding and one for decoding
     * Trained the autoencoder on grayscale CIFAR-10 images, preprocessed and flattened into vectors
     * Compared the reconstruction quality and loss with SVD results, noting that SVD achieved better reconstruction due to its ability to retain more intrinsic data properties
  3. **Shallow Non-linear Autoencoder**:
     * Modified the shallow autoencoder by incorporating non-linear activation functions (sigmoid) instead of linear ones
     * Observed improvements in test loss and reconstruction quality compared to the linear autoencoder
     * Highlighted the role of non-linearities in capturing more complex patterns in the data
  4. **Deep Autoencoder**:
     * Built a deep autoencoder with 5 encoding layers and 4 decoding layers, following an hourglass structure (i.e., the encoding layers reduce dimensionality progressively, and the decoding layers mirror this progression)
     * Experimented with various hyperparameters, including layer sizes, learning rates, and activation functions (e.g: leaky ReLU)
     * Reported on overfitting issues with higher epochs and explored the trade-offs between model depth and generalization
   5. **Shallow Denoising Autoencoder**:
      * Implemented a shallow denoising autoencoder by adding Gaussian noise to the input images and training the model to reconstruct the clean versions
      * Ran experiments with different noise levels, observing how increasing the noise factor impacted the model’s ability to denoise
      * Tuned hyperparameters such as noise factor, learning rate, and number of epochs to optimize the denoising performance

[Homework 6: Variational Autoencoders](https://github.com/TapusiDaniel/DEEP-LEARNING-2022-2023-SCP9087561-/blob/main/2065492_HW6.ipynb)

In this homework, we explored Variational Autoencoders (VAEs), implementing a custom VAE model in PyTorch. The primary goal was to learn the latent space representation of the MNIST dataset using both the reconstruction loss and the Kullback-Leibler Divergence (KL-Divergence). We also experimented with generating new samples from the learned latent space and visualized the results.

### Key tasks:
  1. **VAE Architecture**:
     * Implemented the VAE architecture using PyTorch, defining the Encoder and Decoder parts with dense layers
     * Introduced a Sampling layer using the reparameterization trick, which allows the VAE to generate samples from the learned latent space
     * Created the encoder to output the mean and log-variance for the latent space, and implemented the sampling function to draw latent space values
  2. **Loss Functions**:
     * Defined the two main losses for the VAE:
        * **Reconstruction loss**: Ensured the decoded samples matched the original inputs using binary cross-entropy
        * **KL-Divergence**: Regularized the latent space by comparing the learned distribution to a standard normal distribution
     * Implemented the KL-Divergence loss from scratch within the VAE model
  3. **Training Loop**:
     * Designed a training loop combining the reconstruction loss and KL-Divergence loss, with a tunable beta parameter to control the relative importance of each loss
     * Set the latent dimension to 2 for easy graphical representation of the learned latent space
     * Trained the VAE on the MNIST dataset, splitting it into training and test sets, and used appropriate hyperparameters to ensure convergence
  4. **Hyperparameter Tuning**:
     * Defined an hourglass architecture for the encoder and decoder, starting with 256 units and reducing the number of units in subsequent layers
     * Chose leaky ReLU as the activation function to handle non-linearities and sigmoid for the final layer to normalize the pixel values (ranging between 0 and 1)
     * Experimented with different layer sizes, activation functions, and learning rates to optimize the model’s performance
  5. **Generation of New Samples**:
     * Generated new images by sampling random values from the 2D latent space and feeding them into the decoder
     * Visualized the reconstructed images and observed the smooth transition between different digits in the latent space, which demonstrates how the VAE learns a meaningful representation
  6. **Visualization of Latent Space**:
     * Plotted the distribution of digit encodings in the 2D latent space, showing how the VAE clusters similar digits together (e.g., 8 and 9)
     * Explored the impact of different latent space values on the quality of the reconstructed images
  7. **Hyperparameter Exploration**:
     * Experimented with the number of layers and units, observing that reducing the model complexity resulted in poorer reconstructions, while increasing complexity led to overfitting
     * Compared activation functions like ReLU and leaky ReLU, finding no significant difference in performance for this dataset due to the absence of negative pixel values

# Final Remarks

This course provided me with a solid foundation in deep learning, both in terms of theoretical knowledge and practical implementation. Through these homework assignments, I gained valuable hands-on experience in building, training, and evaluating various deep learning models, ranging from simple feedforward neural networks to more advanced architectures like Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), Transformers, and Autoencoders. 

Each assignment challenged me to deepen my understanding of the intricacies involved in designing and optimizing neural networks. Working with real-world datasets such as CIFAR-10, IMDB, and MNIST, I developed a stronger grasp of how deep learning can be applied across different domains, from image processing to natural language processing and generative modeling.

Looking ahead, I am excited to further explore the world of deep learning, particularly in areas such as generative models, reinforcement learning, and applying deep learning techniques to time series data and unsupervised learning tasks. This course has been an essential step in my journey toward becoming proficient in deep learning, and I look forward to leveraging these skills in both academic research and industry projects.
