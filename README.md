# CIFAR-10 Image Classification

This project explores the development of a Convolutional Neural Network (CNN) model to classify images from the CIFAR-10 dataset into one of ten categories. It covers the full pipeline from data preparation to model evaluation and prediction on unseen data

## Project Overview

The CIFAR-10 dataset comprises 60,000 color images across ten classes, such as airplanes, automobiles, birds, cats, and more. The objective of this project is to build a robust CNN model capable of accurately identifying the class of each image.

## Key Components

- **Data Handling**: Loading, splitting, and preprocessing the CIFAR-10 dataset.
- **CNN Model Architecture**: Designing a CNN with multiple convolutional and fully connected layers.
- **Training and Validation**: Training the model while tracking performance through validation data.
- **Evaluation**: Assessing the model's accuracy and loss on the test dataset.
- **Prediction and Visualization**: Using the trained model to predict and visualize class labels for new images.

## Implementation Details

### 1. Dataset Preparation

- **Loading Data**: The CIFAR-10 dataset is imported directly using Keras. It is divided into training and test sets.
- **Visualizing Samples**: A few sample images are displayed to gain an understanding of the dataset's diversity.
- **Label Mapping**: A dictionary is set up to map numerical labels to their respective categories (e.g., 0 for airplanes, 1 for automobiles).

### 2. Preprocessing

- **Data Splitting**: The training data is split further into training and validation sets (80/20 split).
- **One-Hot Encoding**: Target labels are one-hot encoded to make them compatible with the categorical cross-entropy loss function.
- **Normalization**: All images are normalized by scaling pixel values to a range between 0 and 1, improving the model’s training efficiency.

### 3. CNN Model Architecture

- **Initial Convolutional Block**: The first block consists of a convolutional layer with 32 filters, which is followed by batch normalization and max pooling to downsample the feature maps.
- **Subsequent Convolutional Blocks**: Two additional convolutional blocks are added, each increasing the filter count, followed by batch normalization, max pooling, and dropout layers to prevent overfitting.
- **Fully Connected Layers**: The model includes two dense layers with ReLU activation, followed by dropout layers for further regularization.
- **Output Layer**: A softmax-activated dense layer outputs probabilities for the ten classes.

### 4. Model Compilation and Training

- **Optimizer and Loss Function**: The model is compiled using the Adam optimizer and categorical cross-entropy as the loss function.
- **Training Process**: The model is trained over 10 epochs with a batch size of 64, using the validation data to monitor its performance.
- **Performance Tracking**: Training and validation accuracy and loss are plotted to visualize the model’s learning process.

### 5. Model Evaluation

- **Test Set Evaluation**: The model's performance is assessed on the test dataset, yielding accuracy and loss metrics.
- **Classification Report**: A detailed classification report is generated, showing precision, recall, and F1-score for each class.

### 6. Making Predictions

- **Batch Predictions**: The model predicts the classes of all images in the test set, with results compared to actual labels.
- **Individual Image Prediction**: A specific image from the test set is selected, and the model's prediction is displayed alongside the actual image.


